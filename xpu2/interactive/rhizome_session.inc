// Rhizome in moonray_xpu2 (RHIZOME.md): a frame rendered by a swarm of machines, each a whole XPU2 (its processor groups,
// NUMA nodes and graphics cards), one the master integrator.
//
//   moonray_xpu2 --rhizome-master -in <scene> [-in ...] -out <image.pfm> --key <rhizome key> [--nodes N] [--wait s]
//                [--bind host] [--port p] [--advertise host] [--cell px] [--chunk samples] [--mode 0|1|2] [-threads n]
//       Listens for nodes, waits for N of them (or --wait seconds), plans the frame over them and itself, audits and
//       adds what comes back, and writes the picture (the beauty, a PFM in MoonRay's film order).
//   moonray_xpu2 --rhizome-node --master host:port -in <scene> [-in ...] --key <rhizome key> [--bind host] [--port p]
//                [--advertise host] [--mode 0|1|2] [-threads n] [--folder <its folder: node.key>]
//       Joins the master and renders the chunks it holds until the frame is done.
//
// Each machine reads the scene itself for now (the scene's pieces through the memory pool are to come). A chunk is a
// share of the frame (async::RenderOptions::share): a cell's pixels and a range of the frame's sample indices, the
// samples the whole frame would give them, so the master's picture is the frame one machine would render.
// Included by xpu2_session.cc.
#include "rhizome_node.h"
#include <future>

namespace {
namespace rhizomesession {
namespace rz = xpu2::interactive::rhizome;

struct RhizomeArgs {
    bool master = false;
    std::vector<std::string> scenes;
    std::string out, key, masterAddress, bind = "127.0.0.1", advertise, folder;
    unsigned nodes = 0, threads = 0, cell = 256, chunk = 16;
    uint16_t port = 0;
    int mode = 2;
    double wait = 30;
};
inline RhizomeArgs parseRhizome(int argc, char** argv) {
    RhizomeArgs a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
            return argv[++i];
        };
        if (k == "--rhizome-master") a.master = true;
        else if (k == "--rhizome-node") a.master = false;
        else if (k == "-in") a.scenes.push_back(next());
        else if (k == "-out") a.out = next();
        else if (k == "--key") a.key = next();
        else if (k == "--master") a.masterAddress = next();
        else if (k == "--bind") a.bind = next();
        else if (k == "--advertise") a.advertise = next();
        else if (k == "--port") a.port = uint16_t(std::stoul(next()));
        else if (k == "--nodes") a.nodes = unsigned(std::stoul(next()));
        else if (k == "--wait") a.wait = std::stod(next());
        else if (k == "--cell") a.cell = unsigned(std::stoul(next()));
        else if (k == "--chunk") a.chunk = unsigned(std::stoul(next()));
        else if (k == "--mode") a.mode = std::stoi(next());
        else if (k == "-threads") a.threads = unsigned(std::stoul(next()));
        else if (k == "--folder") a.folder = next();
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.scenes.empty()) throw std::runtime_error("no scene (-in)");
    if (a.key.empty()) throw std::runtime_error("no rhizome key (--key)");
    if (a.master && a.out.empty()) throw std::runtime_error("no picture to write (-out)");
    if (!a.master && a.masterAddress.empty()) throw std::runtime_error("no master (--master host:port)");
    if (a.cell == 0 || a.cell % 64) throw std::runtime_error("--cell must be a multiple of 64");
    return a;
}

// A machine's XPU2: the scene prepared once, its lanes, and a share of the frame rendered on request. Everything runs
// on one thread of its own, the renderer's feeder: a graphics card's lane is used only from the thread that made it.
class Local {
public:
    Local(const RhizomeArgs& a, const fs::path& staging) : mode_(a.mode) {
        std::promise<void> ready;
        auto made = ready.get_future();
        thread_ = std::thread([&, this] {
            try {
                setup(a, staging);
                ready.set_value();
            } catch (...) {
                ready.set_exception(std::current_exception());
                return;
            }
            serve();
        });
        try {
            made.get();
        } catch (...) {
            thread_.join();
            throw;
        }
    }
    ~Local() {
        {
            std::lock_guard<std::mutex> g(jobsMutex_);
            stopping_ = true;
        }
        jobsCv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    const cpu::Plan& plan() const { return *scene_.plan; }
    // A share of the frame: its film (sums and counts) for the rectangle, rendered on the feeder.
    rz::Film render(const uint32_t rect[4], uint32_t first, uint32_t count) {
        const uint32_t r[4] = {rect[0], rect[1], rect[2], rect[3]};
        return call([&] { return renderHere(r, first, count); });
    }
    // A pixel's sums for an audit: its 8-pixel tile's share rendered once, kept for the other pixels checked there.
    void pixel(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) {
        call([&] {
            pixelHere(x, y, first, count, sum);
            return 0;
        });
    }
    // Card and processor memory this machine gives the swarm's pool.
    static uint64_t ramBytes() {
        MEMORYSTATUSEX m{};
        m.dwLength = sizeof m;
        GlobalMemoryStatusEx(&m);
        return m.ullAvailPhys / 2;
    }

    // On the feeder (call), and what it runs there.
    template <class F>
    auto call(F f) -> decltype(f()) {
        using R = decltype(f());
        std::packaged_task<R()> task(std::move(f));
        auto result = task.get_future();
        {
            std::lock_guard<std::mutex> g(jobsMutex_);
            jobs_.push_back([&task] { task(); });
        }
        jobsCv_.notify_all();
        return result.get();
    }
    void serve() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> g(jobsMutex_);
                jobsCv_.wait(g, [&] { return stopping_ || !jobs_.empty(); });
                if (jobs_.empty()) break;
                job = std::move(jobs_.front());
                jobs_.pop_front();
            }
            job();
        }
        renderer_.reset();
        gpu_.reset();
    }
    void setup(const RhizomeArgs& a, const fs::path& staging) {
        threads_ = a.threads ? a.threads : multi::logicalProcessors();
        scene_.coarseFirst = false;
        scene_.dso = executableFolder();
        scene_.staging = staging;
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        fs::create_directories(staging);
        cpu::sequenceDirectory() = executableFolder();
        scene_.load(a.scenes);
        scene_.prepare();
        if (mode_ != 0) makeGpu();
        renderer_ = std::make_unique<async::Renderer>(scene_.plan, *scene_.cpuScene, gpu_.get());
        std::printf("MoonRay XPU2 Rhizome: scene ready (%ux%u, %u samples a pixel), %s, %u threads\n", plan().width, plan().height, plan().spp,
                    gpu_ ? (mode_ == 1 ? "the GPU alone" : "CPU and GPU") : "the CPU alone", threads_);
        std::fflush(stdout);
    }
    rz::Film renderHere(const uint32_t rect[4], uint32_t first, uint32_t count) {
        async::RenderOptions o;
        o.threads = threads_;
        o.mode = gpu_ ? mode_ : 0;
        o.share = true;
        for (int k = 0; k < 4; ++k) o.region[k] = int(rect[k]);
        o.sampleFirst = first, o.sampleCount = count;
        o.grid.cellSize = 64;
        o.grid.progressive = false;
        o.important = false;
        o.partial = false;  // every sample whole: the shares must add to the frame
        o.deviceChain = false;
        const async::RenderResult r = renderer_->render(o);
        if (r.failed) throw std::runtime_error(r.error.empty() ? "the render failed" : r.error);
        rz::Film f;
        for (int k = 0; k < 4; ++k) f.rect[k] = rect[k];
        f.first = first, f.count = count;
        const uint32_t W = plan().width;
        const size_t n = size_t(f.width()) * f.height();
        f.sums.resize(n * 4);
        f.counts.resize(n);
        // The AOV accumulators of the share's rows (RenderResult::aovRaw), and the odd-index sums where the scene has
        // aux outputs (RenderResult::odd holds their means).
        const aov::Tables& t = renderer_->aovTables();
        const bool aovs = t.channels && !r.aovRaw.empty();
        if (aovs) f.aovs.resize(n * t.channels), f.aovDepths.resize(n * t.closestCount);
        if (!r.odd.empty()) f.odd.resize(n * 4);
        size_t i = 0;
        for (uint32_t y = rect[1]; y < rect[3]; ++y)
            for (uint32_t x = rect[0]; x < rect[2]; ++x, ++i) {
                const size_t p = size_t(y) * W + x;
                const uint32_t k = r.counts[p];
                f.counts[i] = k;
                for (int c = 0; c < 4; ++c) f.sums[i * 4 + c] = r.rgba[p * 4 + c] * float(k);
                if (!f.odd.empty())
                    for (int c = 0; c < 4; ++c) f.odd[i * 4 + c] = r.odd[p * 4 + c] * float(k);
                if (aovs && p >= r.shareFrom && p < r.shareTo) {
                    const size_t b = p - r.shareFrom;
                    std::copy_n(r.aovRaw.begin() + b * t.channels, t.channels, f.aovs.begin() + i * t.channels);
                    std::copy_n(r.aovRawDepths.begin() + b * t.closestCount, t.closestCount, f.aovDepths.begin() + i * t.closestCount);
                }
            }
        return f;
    }
    // The master's picture made whole: the merged films' AOVs resolved, the display filters run, and the scene's
    // render outputs written as a frame writes them (aov_write.h). film: sums a pixel; odd: odd-index sums.
    void writeOutputs(const std::vector<double>& film, const std::vector<uint32_t>& counts, const std::vector<double>& odd,
                      const std::vector<double>& raw, const std::vector<float>& depths) {
        call([&] {
            if (!scene_.plan->aovs) return 0;
            const size_t pixels = size_t(plan().width) * plan().height;
            async::RenderResult result;
            result.rgba.resize(pixels * 4);
            result.counts.assign(counts.begin(), counts.end());
            if (!odd.empty()) result.odd.resize(pixels * 4);
            for (size_t p = 0; p < pixels; ++p)
                for (int c = 0; c < 4; ++c) {
                    result.rgba[p * 4 + c] = counts[p] ? float(film[p * 4 + c] / counts[p]) : 0.0f;
                    if (!odd.empty()) result.odd[p * 4 + c] = counts[p] ? float(odd[p * 4 + c] / counts[p]) : 0.0f;
                }
            if (!raw.empty()) renderer_->resolveMerged(raw, depths, result.counts, result.aov);
            if (scene_.plan->aovs->cryptomatte)
                std::printf("MoonRay XPU2 Rhizome: Cryptomatte is not merged across nodes yet: its layers are written empty\n");
            std::vector<std::string> warnings;
            if (!scene_.displayFilters.empty()) {
                std::vector<std::vector<float>> filtered;
                scene_.displayFilters.run(*scene_.plan->aovs, result, filtered);
                size_t k = 0;
                for (const aov::Output* out : scene_.plan->aovs->all)
                    if (out->result == aov::ResultDisplayFilter && k < filtered.size()) result.displayFilters[out->name] = std::move(filtered[k++]);
            }
            const std::string error = aov::writeOutputs(*scene_.plan->aovs, result, plan().width, plan().height, warnings);
            for (const auto& w : warnings) std::printf("MoonRay XPU2 Rhizome: %s\n", w.c_str());
            if (!error.empty()) throw std::runtime_error(error);
            std::printf("MoonRay XPU2 Rhizome: the scene's render outputs written\n");
            return 0;
        });
    }
    const aov::Tables& aovTables() const { return renderer_->aovTables(); }
    void pixelHere(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) {
        const uint32_t tx = x & ~7u, ty = y & ~7u;
        const auto key = std::make_tuple(tx, ty, first, count);
        if (auditKey_ != key) {
            const uint32_t rect[4] = {tx, ty, std::min(tx + 8, plan().width), std::min(ty + 8, plan().height)};
            auditFilm_ = renderHere(rect, first, count);
            auditKey_ = key;
        }
        const size_t i = size_t(y - ty) * auditFilm_.width() + (x - tx);
        for (int c = 0; c < 4; ++c) sum[c] = auditFilm_.sums[i * 4 + c];
    }
    void makeGpu() {
        const fs::path folder = executableFolder();
        const Programs p = programs(folder);
        if (!fs::exists(shaderMark(p))) {
            std::printf("MoonRay XPU2 Rhizome: the GPU programs are not compiled here (moonray_xpu2 --compile-shaders): the CPU alone\n");
            return;
        }
        const std::set<std::string> held = compiledKinds(shaderMark(p));
        std::vector<int> cards;
        for (int card : multi::cardOrdinals(gpu::cardCount()))
            if (held.count(cardKind(card))) cards.push_back(card);
        if (cards.empty()) return;
        gpu_ = gpuLane(p, {}, cards[0]);
        for (size_t k = 1; k < cards.size(); ++k) {
            try {
                gpu_->follow(gpuLane(p, {}, cards[k], int(k)));
            } catch (const std::exception& e) {
                gpu_->use();
                std::printf("MoonRay XPU2 Rhizome: graphics card %d left out: %s\n", cards[k], e.what());
            }
        }
        const std::string why = async::GpuLane::cpuOnly(scene_.plan);
        if (!why.empty()) {
            std::printf("MoonRay XPU2 Rhizome: the GPU leaves %s to the CPU; the CPU alone\n", why.c_str());
            gpu_.reset();
            return;
        }
        gpu_->prepare(scene_.plan);
    }
private:
    Scene scene_;
    std::unique_ptr<async::GpuLane> gpu_;
    std::unique_ptr<async::Renderer> renderer_;
    int mode_ = 2;
    unsigned threads_ = 1;
    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> auditKey_{~0u, ~0u, 0, 0};
    rz::Film auditFilm_;
    std::thread thread_;
    std::mutex jobsMutex_;
    std::condition_variable jobsCv_;
    std::deque<std::function<void()>> jobs_;
    bool stopping_ = false;
};

inline void writePfm(const fs::path& path, uint32_t w, uint32_t h, const std::vector<double>& film, const std::vector<uint32_t>& counts) {
    std::ofstream out(path, std::ios::binary);
    const std::string header = "PF\n" + std::to_string(w) + " " + std::to_string(h) + "\n-1.0\n";
    out.write(header.data(), std::streamsize(header.size()));
    std::vector<float> rgb(size_t(w) * h * 3);
    for (size_t i = 0; i < size_t(w) * h; ++i)
        for (int c = 0; c < 3; ++c) rgb[i * 3 + c] = counts[i] ? float(film[i * 4 + c] / counts[i]) : 0.0f;
    out.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size() * sizeof(float)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

// The scene as the swarm's memory holds it (RHIZOME.md, 4b): the master's scene files and every file they name (a quoted
// string in a scene that is a file there: textures, VDB grids, caches; a UDIM set's tiles), each cut into pieces named
// by their SHA-256 and placed in the pool. A node fetches the pieces, writes the files into its own folder, and points
// the scene files' strings at them. Binary scenes (.rdlb) go as they are: what they name is not rewritten.
constexpr size_t PieceBytes = 4u << 20;
struct SceneFile {
    std::string literal;  // as the scene writes it (a scene given with -in: its path), escapes and all
    std::string name;     // the file's own name (a UDIM set: one entry a tile, its literal the set's)
    bool scene = false;
    uint64_t bytes = 0;
    std::vector<rz::Hash> pieces;
};
// A string literal of a scene file unescaped as Lua reads it (backslash escapes; enough for paths).
inline std::string unescape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[++i];
            out += c == 'n' ? '\n' : c == 't' ? '\t' : c;
        } else {
            out += s[i];
        }
    }
    return out;
}
inline std::string escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    return out;
}
inline bool textScene(const fs::path& p) {
    std::string e = p.extension().string();
    for (auto& c : e) c = char(std::tolower(static_cast<unsigned char>(c)));
    return e == ".rdla" || e == ".lua";
}
inline std::vector<std::string> literals(const std::string& text) {
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '"') continue;
        size_t j = i + 1;
        while (j < text.size() && text[j] != '"' && text[j] != '\n') j += text[j] == '\\' ? 2 : 1;
        if (j < text.size() && text[j] == '"') out.push_back(text.substr(i + 1, j - i - 1));
        i = j;
    }
    return out;
}
// The master: the files of the scene, read and placed in the pool. Returns the manifest.
inline rz::Bytes publishScene(rz::RhizomeNode& node, const std::vector<std::string>& scenes, uint64_t& total) {
    std::vector<SceneFile> files;
    std::set<std::string> seen;
    auto add = [&](const std::string& literal, const fs::path& path, bool scene) {
        SceneFile f;
        f.literal = literal, f.name = path.filename().u8string(), f.scene = scene;
        std::ifstream in(path, std::ios::binary);
        rz::Bytes block(PieceBytes);
        for (;;) {
            in.read(reinterpret_cast<char*>(block.data()), std::streamsize(block.size()));
            const size_t got = size_t(in.gcount());
            if (!got) break;
            rz::Bytes data(block.begin(), block.begin() + got);
            const rz::Hash h = rz::sha256(data);
            node.addPiece(rz::Piece{h, data.size(), rz::Tier::Ram}, data);
            f.pieces.push_back(h);
            f.bytes += got;
        }
        total += f.bytes;
        files.push_back(std::move(f));
    };
    for (const auto& s : scenes) {
        const fs::path p = fs::u8path(s);
        add(s, p, true);
        if (!textScene(p)) continue;
        std::ifstream in(p, std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        for (const std::string& lit : literals(text)) {
            if (lit.size() < 3 || seen.count(lit)) continue;
            const std::string path = unescape(lit);
            std::error_code e;
            const auto udim = path.find("<UDIM>") != std::string::npos ? path.find("<UDIM>") : path.find("<udim>");
            if (udim != std::string::npos) {
                bool any = false;
                for (int tile = 1001; tile < 2000; ++tile) {
                    std::string one = path;
                    one.replace(udim, 6, std::to_string(tile));
                    if (fs::is_regular_file(fs::u8path(one), e)) add(lit, fs::u8path(one), false), any = true;
                }
                if (any) seen.insert(lit);
                continue;
            }
            if (path.find_first_of("/\\") == std::string::npos || !fs::is_regular_file(fs::u8path(path), e)) continue;
            seen.insert(lit);
            add(lit, fs::u8path(path), false);
        }
    }
    rz::Writer w;
    w.u32(uint32_t(files.size()));
    for (const auto& f : files) {
        w.str(f.literal), w.str(f.name), w.u8(f.scene), w.u64(f.bytes);
        w.u32(uint32_t(f.pieces.size()));
        for (const auto& h : f.pieces) w.hash(h);
    }
    std::printf("MoonRay XPU2 Rhizome: the scene in the swarm's memory: %zu files, %.1f MB\n", files.size(), double(total) / (1 << 20));
    std::fflush(stdout);
    return w.b;
}
// A node: the scene's files fetched from the pool and written into folder, the scene files' strings pointed at them.
// Returns the scene files to load.
inline std::vector<std::string> receiveScene(rz::RhizomeNode& node, const rz::Bytes& manifest, const fs::path& folder) {
    rz::Reader r(manifest);
    std::vector<SceneFile> files(r.u32());
    std::vector<rz::Hash> all;
    for (auto& f : files) {
        f.literal = r.str(), f.name = r.str(), f.scene = r.u8() != 0, f.bytes = r.u64();
        f.pieces.resize(r.u32());
        for (auto& h : f.pieces) h = r.hash();
        all.insert(all.end(), f.pieces.begin(), f.pieces.end());
    }
    if (r.bad) throw std::runtime_error("the master's manifest does not read");
    const auto began = Clock::now();
    if (!node.fetch(all, 3600)) throw std::runtime_error("the scene's pieces did not all arrive");
    // Each literal's files in a folder of their own (a UDIM set's tiles together, under their own names).
    std::map<std::string, fs::path> where;
    std::vector<std::pair<fs::path, bool>> written;
    uint64_t total = 0;
    for (const auto& f : files) {
        auto it = where.find(f.literal);
        if (it == where.end()) it = where.emplace(f.literal, folder / ("f" + std::to_string(where.size()))).first;
        fs::create_directories(it->second);
        const fs::path path = it->second / fs::u8path(f.name);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        for (const auto& h : f.pieces) {
            const rz::Bytes data = node.piece(h);
            out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        }
        if (!out) throw std::runtime_error("cannot write " + path.string());
        total += f.bytes;
        if (f.scene) written.push_back({path, textScene(path)});
    }
    std::vector<std::string> scenes;
    for (const auto& [path, text] : written) {
        if (text) {
            std::ifstream in(path, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            in.close();
            for (const auto& f : files) {
                if (f.scene) continue;
                const std::string from = "\"" + f.literal + "\"";
                std::string name = unescape(f.literal);
                name = name.substr(name.find_last_of("/\\") + 1);
                const std::string to = "\"" + escape((where[f.literal] / fs::u8path(name)).u8string()) + "\"";
                for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
            }
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(s.data(), std::streamsize(s.size()));
        }
        scenes.push_back(path.u8string());
    }
    std::printf("MoonRay XPU2 Rhizome: the scene from the swarm: %zu files, %.1f MB in %.1f s\n", files.size(), double(total) / (1 << 20),
                std::chrono::duration<double>(Clock::now() - began).count());
    std::fflush(stdout);
    return scenes;
}

inline int run(int argc, char** argv) {
    RhizomeArgs a = parseRhizome(argc, argv);
    const fs::path base = fs::temp_directory_path() / ("xpu2-rhizome-" + std::to_string(GetCurrentProcessId()));
    // The master reads the scene where it is; a node makes it from the swarm's memory once it has joined.
    std::unique_ptr<Local> held;
    if (a.master) held = std::make_unique<Local>(a, base / "staging");
    Local* local = held.get();
    std::atomic<Local*> made{local};
    // The master's merge of the AOVs and aux outputs (each film as it is taken in, by each output's filter).
    std::vector<double> mergedAovs, odd;
    std::vector<float> mergedDepths;
    rz::NodeOptions o;
    o.readyAtJoin = false;
    std::promise<rz::Bytes> manifestPromise;
    auto manifest = manifestPromise.get_future();
    std::atomic<bool> manifestSet{false};
    o.onManifest = [&](const rz::Bytes& m) {
        if (!manifestSet.exchange(true)) manifestPromise.set_value(m);
    };
    if (a.master) {
        const aov::Tables& t = local->aovTables();
        const size_t pixels = size_t(local->plan().width) * local->plan().height;
        if (t.channels) {
            mergedAovs.resize(pixels * t.channels);
            mergedDepths.resize(pixels * std::max<U32>(t.closestCount, 1));
            for (size_t p = 0; p < pixels; ++p) async::initAovPixel(t, mergedAovs.data() + p * t.channels, mergedDepths.data() + p * t.closestCount);
        }
        const bool aux = local->plan().aovs && (local->plan().aovs->beautyAux || local->plan().aovs->alphaAux);
        if (aux) odd.assign(pixels * 4, 0.0);
        o.onFilm = [&, pixels, t](const rz::Film& f) {
            const uint32_t W = local->plan().width;
            size_t i = 0;
            for (uint32_t y = f.rect[1]; y < f.rect[3]; ++y)
                for (uint32_t x = f.rect[0]; x < f.rect[2]; ++x, ++i) {
                    const size_t p = size_t(y) * W + x;
                    if (p >= pixels) continue;
                    if (!odd.empty() && !f.odd.empty())
                        for (int c = 0; c < 4; ++c) odd[p * 4 + c] += f.odd[i * 4 + c];
                    if (t.channels && f.aovs.size() >= (i + 1) * t.channels)
                        async::mergeAovPixel(t, mergedAovs.data() + p * t.channels, mergedDepths.data() + p * t.closestCount,
                                             f.aovs.data() + i * t.channels, f.aovDepths.data() + i * t.closestCount);
                }
        };
    }
    o.rhizomeKey = a.key;
    o.master = a.master;
    o.masterAddress = a.masterAddress;
    o.bind = a.bind, o.port = a.port, o.advertise = a.advertise;
    o.caps.cpuThreads = multi::logicalProcessors();
    o.caps.numaNodes = unsigned(std::max<size_t>(1, multi::nodes().size()));
    o.caps.cards = unsigned(gpu::cardCount());
    o.ramBytes = Local::ramBytes();
    o.render = [&](const uint32_t r[4], uint32_t first, uint32_t count) {
        Local* l = made.load();
        if (!l) throw std::runtime_error("the scene is not made here yet");
        return l->render(r, first, count);
    };
    o.renderPixel = [&](uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) { made.load()->pixel(x, y, first, count, sum); };
    o.log = [](const std::string& s) {
        std::printf("MoonRay XPU2 Rhizome: %s\n", s.c_str());
        std::fflush(stdout);
    };
    // The node's hash is kept in its folder (node.key), the same each time it runs from it.
    const fs::path folder = a.folder.empty() ? (executableFolder() / "rhizome" / (a.master ? "master" : "node")) : fs::u8path(a.folder);
    rz::RhizomeNode node(o, rz::Identity::load(folder));
    node.start();
    std::printf("MoonRay XPU2 Rhizome: %s %s at %s\n", a.master ? "master" : "node", rz::hex(node.hash()).c_str(), node.address().c_str());
    std::fflush(stdout);
    if (!a.master) {
        // A node: the scene from the swarm's memory, made here, then ready for cells; it renders until it is stopped.
        if (manifest.wait_for(std::chrono::duration<double>(std::max(a.wait, 60.0))) != std::future_status::ready)
            throw std::runtime_error("no manifest from the master");
        a.scenes = receiveScene(node, manifest.get(), base / "scene");
        held = std::make_unique<Local>(a, base / "staging");
        made.store(held.get());
        node.markReady();
        std::printf("MoonRay XPU2 Rhizome: node ready\n");
        std::fflush(stdout);
        for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    uint64_t sceneBytes = 0;
    node.publishManifest(publishScene(node, a.scenes, sceneBytes));
    const auto began = Clock::now();
    while (node.ready() < a.nodes && std::chrono::duration<double>(Clock::now() - began).count() < a.wait)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));  // the nodes meet each other
    const cpu::Plan& p = local->plan();
    rz::FrameShape shape;
    shape.width = p.width, shape.height = p.height, shape.cellSize = a.cell, shape.spp = p.spp, shape.chunkSamples = std::min(a.chunk, p.spp);
    std::printf("MoonRay XPU2 Rhizome: %zu nodes joined, %zu ready; the frame in %u cells of %u px, chunks of %u samples\n", node.joined(),
                node.ready(), shape.cells(), shape.cellSize, shape.chunkSamples);
    const auto frameBegan = Clock::now();
    node.beginFrame(shape, {});
    if (!node.waitFrame(24 * 3600.0)) throw std::runtime_error("the frame did not complete");
    const double seconds = std::chrono::duration<double>(Clock::now() - frameBegan).count();
    const std::vector<double> film = node.film();
    const std::vector<uint32_t> counts = node.counts();
    writePfm(fs::u8path(a.out), shape.width, shape.height, film, counts);
    local->writeOutputs(film, counts, odd, mergedAovs, mergedDepths);
    const auto by = node.chunksByNode();
    std::string share;
    for (size_t i = 0; i < by.size(); ++i)
        share += (i ? ", " : "") + rz::shortHex(node.plan()->nodes[i].hash) + " " + std::to_string(by[i]) + (node.plan()->nodes[i].fast ? " (fast)" : " (slow)");
    std::printf("MoonRay XPU2 Rhizome: the frame in %.2f s, %llu films audited; chunks by node: %s\n", seconds,
                (unsigned long long)node.audits(), share.c_str());
    std::printf("MoonRay XPU2 Rhizome: wrote %s\n", a.out.c_str());
    std::fflush(stdout);
    node.stop();
    return 0;
}

}  // namespace rhizomesession
}  // namespace
