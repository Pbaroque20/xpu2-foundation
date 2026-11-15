// Rhizome in moonray_xpu2 (RHIZOME.md): a frame rendered by a swarm of machines, each a whole XPU2 (its processor groups,
// NUMA nodes and graphics cards), one the master integrator. Every machine has the same command line; what it may do
// follows its keys and endorsements (rhizome_trust.h).
//
//   moonray_xpu2 rhizome node    --key <k> [--master host:port] [--trust <approver hash> ...] [--mode 0|1|2] [-threads n]
//                                [--folder f] [--bind host] [--port p] [--advertise host] [--no-discovery]
//                                [--when-idle s] [--hours from-to] [--busy-threads n] [--shaders ask|compile|skip]
//                                [--whole-textures]
//   moonray_xpu2 rhizome master  -in <scene> [-in ...] -out <image.pfm> --key <k> [--nodes N] [--node host:port ...]
//                                [--wait s] [--approve] [--trust-first] [--trust <hash> ...] [--cell px] [--chunk n]
//                                [--mode 0|1|2] [-threads n] [--folder f] [--bind host] [--port p] [--no-discovery]
//                                [--frames 1-24] [--delta <frame.####.rdla> ...] [--queue folder [--queue-watch]]
//                                [--checkpoint s] [--fresh] [--shaders ask|compile|skip] [--scan [ranges]] [--no-broadcast]
//   moonray_xpu2 rhizome submit  --queue folder -in <scene> [-in ...] -out <image> [--frames 1-24] [--delta ...] [--name n]
//   moonray_xpu2 rhizome id      [--folder f]
//   moonray_xpu2 rhizome trust   <approver hash> [--folder f]
//   moonray_xpu2 rhizome approve <hash> [--approver] [--folder f]
//   moonray_xpu2 rhizome revoke  <hash> [--folder f]
//   moonray_xpu2 rhizome status  [--folder f] [--watch]
//   moonray_xpu2 rhizome firewall [--check] [--print] [--remove] [--public]
//   moonray_xpu2 rhizome <verb> --help
// (--rhizome-node and --rhizome-master: node and master.)
//
// A node needs no scene and no master: it makes its hash the first time it runs (node.key in its folder), listens,
// answers discovery on every network adapter (a direct cable too), and renders each job a master gives it, then waits
// for the next. A master finds the nodes on its networks (or is given them), sends them the scene through the swarm's
// memory, audits and adds what comes back, and writes the picture and the scene's outputs. A chunk is a share of the
// frame (async::RenderOptions::share): a cell's pixels and a range of the frame's sample indices, the samples the whole
// frame would give them, so the master's picture is the frame one machine would render. Included by xpu2_session.cc.
#include "rhizome_node.h"
#include "rhizome_discovery.h"
#include "rhizome_importance.h"
#include "rhizome_jobs.h"
#include <future>
#include <shellapi.h>

namespace {
namespace rhizomesession {
namespace rz = xpu2::interactive::rhizome;

constexpr uint16_t NodePort = 47100, MasterPort = 47110;

enum class Verb { Node, Master, Submit, Id, Trust, Approve, Revoke, Status, Firewall, Help };
struct RhizomeArgs {
    Verb verb = Verb::Help;
    bool master = false, help = false;
    std::vector<std::string> scenes, nodeAddresses, trust;
    std::string out, key, masterAddress, bind = "0.0.0.0", advertise, folder, subject;
    // Jobs (plan 13): the frames, each frame's changes (read on top of the frame before), the queue folder.
    std::vector<std::string> deltas;
    std::string frames, queue, name;
    bool queueWatch = false;
    // Checkpoints (plan 14): every so many seconds (0: none), and whether a job's checkpoint is left unused.
    double checkpointSeconds = 60;
    bool fresh = false;
    // Plan 16: a node's textures copied whole before it renders, instead of read from the swarm's memory as it renders.
    bool wholeTextures = false;
    // GPU programs not compiled here (plan 15): ask (on first start, the answer kept), compile, skip.
    std::string shaders = "ask";
    // Discovery beyond broadcast (plan 18): sweeps of ranges or of each adapter's subnet, broadcast left out.
    std::vector<std::string> scan;
    bool scanAdapters = false, broadcast = true;
    // The firewall verb: look only, print the commands, take the rule away, public networks too; (the elevated copy).
    bool firewallCheck = false, firewallPrint = false, firewallRemove = false, firewallPublic = false, firewallElevated = false;
    unsigned nodes = 0, threads = 0, cell = 256, chunk = 16;
    // A workstation: renders only once no one has used it for whenIdle seconds (0: always), within hours [from, to) of
    // the day (to below from: across midnight; -1: any hour); in use, busyThreads threads instead of stepping back.
    double whenIdle = 0;
    int hoursFrom = -1, hoursTo = -1;
    unsigned busyThreads = 0;
    int port = -1;  // -1: the verb's own (NodePort, MasterPort)
    int mode = 2;
    double wait = 30;
    bool approve = false, trustFirst = false, approver = false, watch = false, discovery = true, json = false;
};

inline const char* usage(Verb v) {
    switch (v) {
    case Verb::Node:
        return "moonray_xpu2 rhizome node --key <rhizome key> [--master host:port] [--trust <approver hash> ...] [--mode 0|1|2]\n"
               "    [-threads n] [--folder f] [--bind host] [--port p] [--advertise host] [--no-discovery]\n"
               "    [--when-idle seconds] [--hours from-to] [--busy-threads n] [--shaders ask|compile|skip] [--whole-textures]\n"
               "  A headless node: makes or reads its hash, listens and answers discovery, renders each job a master gives it\n"
               "  and waits for the next. --master: reach that master instead of waiting to be found. --trust: approvers this\n"
               "  node trusts (else it trusts the master that first endorses it). Port 47100 by default.\n"
               "  A workstation: --when-idle: render only once no one has touched the keyboard or mouse for that many seconds;\n"
               "  --hours 19-8: only in those hours (across midnight too); --busy-threads: while the machine is in use, render\n"
               "  on that many threads instead of stepping back. Stepped back, its work goes to the other machines.\n"
               "  GPU programs: a machine whose cards have none compiled renders on its processor alone; on its first start it\n"
               "  asks whether to compile them (--shaders ask; compile or skip decide it beforehand). The compile runs in the\n"
               "  background (its lines in shaders-compile.txt in the folder), the cards rendering from the next job after;\n"
               "  rhizome status shows each machine's GPU programs.\n"
               "  Textures: a scene's textures (.tx) stay in the swarm's memory, their tiles fetched as the render reads them\n"
               "  (those it never reads never come); --whole-textures: each copied whole before the node renders.";
    case Verb::Master:
        return "moonray_xpu2 rhizome master -in <scene> [-in ...] -out <image.pfm> --key <rhizome key> [--nodes N]\n"
               "    [--node host:port ...] [--wait s] [--approve] [--trust-first] [--trust <approver hash> ...] [--cell px]\n"
               "    [--chunk samples] [--mode 0|1|2] [-threads n] [--folder f] [--bind host] [--port p] [--no-discovery]\n"
               "    [--shaders ask|compile|skip] [--scan [ranges]] [--no-broadcast]\n"
               "  Finds the nodes on its networks (and those given with --node), waits for N of them ready (or --wait seconds),\n"
               "  renders the frame with them, and writes the picture and the scene's outputs. --approve: only approved\n"
               "  machines join (the others wait, their hash printed with the command to approve them); --trust-first:\n"
               "  approve every machine met on this run. Port 47110 by default.\n"
               "  Animation: --frames 1-24 (1-24:2 every second, 1,5,9 a list); a path with a run of # is a file a frame\n"
               "  (-in shot.####.rdla: the scene made each frame, its unchanged files not sent again). --delta shot.####.rdla:\n"
               "  each frame's changes read on top of the frame before, the scene kept on every machine between frames.\n"
               "  -out beauty.####.pfm (without #, .0001 and so on is added for more than one frame); the scene's outputs\n"
               "  likewise, unless their names already carry the frame. --queue folder: then the jobs in that folder\n"
               "  (moonray_xpu2 rhizome submit), one after another; --queue-watch: wait for more when it is empty.\n"
               "  Checkpoints: the job's frames done and the frame under way are saved in the master's folder every 60 s\n"
               "  (--checkpoint s; 0: never) and after each frame; started again on the same job (every machine stopped,\n"
               "  the queue's too), the master takes it up where it was, and skips a job it finished; a job whose scene or\n"
               "  delta files were written again since begins anew. --fresh: begin it anew anyway.\n"
               "  --shaders: as a node's (rhizome node --help).\n"
               "  Beyond broadcast (a VPN, a routed network): the machines reached before are remembered (known.txt in the\n"
               "  folder) and probed one by one at each start; --scan 10.8.0.0/24,10.9.0.0/24 probes those ranges address by\n"
               "  address (at most /16 each), --scan alone each adapter's subnet; --no-broadcast: no broadcasts. A machine\n"
               "  probes cannot reach: --node host:port. Other machines reach this one through the firewall's rule\n"
               "  (moonray_xpu2 rhizome firewall).";
    case Verb::Submit:
        return "moonray_xpu2 rhizome submit --queue folder -in <scene> [-in ...] -out <image> [--frames 1-24]\n"
               "    [--delta <frame.####.rdla> ...] [--name n]\n"
               "  Adds a job to a master's queue folder (rhizome master --queue folder renders them in turn). Its paths are\n"
               "  made whole here: the master must reach them the same way.";
    case Verb::Id: return "moonray_xpu2 rhizome id [--folder f]\n  This machine's hash (made the first time).";
    case Verb::Trust: return "moonray_xpu2 rhizome trust <approver hash> [--folder f]\n  Trust an approver: its endorsements and revocations count here.";
    case Verb::Approve:
        return "moonray_xpu2 rhizome approve <hash> [--approver] [--folder f]\n"
               "  Endorse a machine with this machine's key (it must be an approver). --approver: it may approve in its turn,\n"
               "  as a master's successor must. A master running from the same folder admits it within a second.";
    case Verb::Revoke:
        return "moonray_xpu2 rhizome revoke <hash> [--folder f]\n"
               "  An approver: revoke a machine, signed, passed through the swarm. Another machine: block it here only.";
    case Verb::Status:
        return "moonray_xpu2 rhizome status [--folder f] [--watch] [--json]\n"
               "  The swarm as this machine sees it: the table a running node or master writes each second. --json: its\n"
               "  status.json (RHIZOME_API.md) instead. Exit 0 while one runs from the folder, 3 when none does.";
    case Verb::Firewall:
        return "moonray_xpu2 rhizome firewall [--check] [--print] [--remove] [--public]\n"
               "  Lets other machines reach this one's node or master through Windows Firewall: one inbound rule for this\n"
               "  moonray_xpu2.exe on private and domain networks (--public: public ones too), added once. Adding or removing it\n"
               "  asks for an administrator (Windows' own prompt). --check: whether it is there, nothing changed; --print: the\n"
               "  commands to run by hand; --remove: the rule taken away.";
    default:
        return "moonray_xpu2 rhizome node | master | submit | id | trust | approve | revoke | status | firewall   (each with --help)";
    }
}

inline RhizomeArgs parseRhizome(int argc, char** argv) {
    RhizomeArgs a;
    int i = 1;
    const std::string first = argc > 1 ? argv[1] : "";
    if (first == "--rhizome-master") a.verb = Verb::Master, i = 2;
    else if (first == "--rhizome-node") a.verb = Verb::Node, i = 2;
    else if (first == "rhizome") {
        const std::string v = argc > 2 ? argv[2] : "help";
        a.verb = v == "node" ? Verb::Node : v == "master" ? Verb::Master : v == "submit" ? Verb::Submit : v == "id" ? Verb::Id : v == "trust" ? Verb::Trust
                 : v == "approve" ? Verb::Approve : v == "revoke" ? Verb::Revoke : v == "status" ? Verb::Status
                 : v == "firewall" ? Verb::Firewall : Verb::Help;
        if (a.verb == Verb::Help && v != "help" && v != "--help") throw std::runtime_error("unknown verb " + v + "\n" + usage(Verb::Help));
        i = 3;
    }
    a.master = a.verb == Verb::Master;
    for (; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
            return argv[++i];
        };
        if (k == "--help" || k == "-h") a.help = true;
        else if (k == "-in") a.scenes.push_back(next());
        else if (k == "-out") a.out = next();
        else if (k == "--key") a.key = next();
        else if (k == "--master") a.masterAddress = next();
        else if (k == "--node") a.nodeAddresses.push_back(next());
        else if (k == "--trust") a.trust.push_back(next());
        else if (k == "--bind") a.bind = next();
        else if (k == "--advertise") a.advertise = next();
        else if (k == "--port") a.port = std::stoi(next());
        else if (k == "--nodes") a.nodes = unsigned(std::stoul(next()));
        else if (k == "--wait") a.wait = std::stod(next());
        else if (k == "--cell") a.cell = unsigned(std::stoul(next()));
        else if (k == "--chunk") a.chunk = unsigned(std::stoul(next()));
        else if (k == "--when-idle") a.whenIdle = std::stod(next());
        else if (k == "--busy-threads") a.busyThreads = unsigned(std::stoul(next()));
        else if (k == "--hours") {
            const std::string h = next();
            const auto dash = h.find('-');
            if (dash == std::string::npos) throw std::runtime_error("--hours wants from-to, as 19-8");
            a.hoursFrom = std::stoi(h.substr(0, dash)) % 24, a.hoursTo = std::stoi(h.substr(dash + 1)) % 24;
        }
        else if (k == "--mode") a.mode = std::stoi(next());
        else if (k == "-threads") a.threads = unsigned(std::stoul(next()));
        else if (k == "--folder") a.folder = next();
        else if (k == "--approve") a.approve = true;
        else if (k == "--trust-first") a.trustFirst = true;
        else if (k == "--approver") a.approver = true;
        else if (k == "--watch") a.watch = true;
        else if (k == "--json") a.json = true;
        else if (k == "--no-discovery") a.discovery = false;
        else if (k == "--frames") a.frames = next();
        else if (k == "--delta") a.deltas.push_back(next());
        else if (k == "--queue") a.queue = next();
        else if (k == "--queue-watch") a.queueWatch = true;
        else if (k == "--name") a.name = next();
        else if (k == "--checkpoint") a.checkpointSeconds = std::stod(next());
        else if (k == "--fresh") a.fresh = true;
        else if (k == "--whole-textures") a.wholeTextures = true;
        else if (k == "--scan") {
            // --scan alone: each adapter's subnet; --scan 10.8.0.0/24,192.168.5.0/24: those.
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string list = next();
                for (size_t at = 0; at <= list.size();) {
                    const size_t comma = std::min(list.find(',', at), list.size());
                    if (comma > at) a.scan.push_back(list.substr(at, comma - at));
                    at = comma + 1;
                }
                for (const auto& s : a.scan) {
                    rz::Range r;
                    if (!rz::parseRange(s, r)) throw std::runtime_error("--scan takes IPv4 ranges like 10.8.0.0/24 (at most /16): " + s);
                }
            } else {
                a.scanAdapters = true;
            }
        } else if (k == "--no-broadcast") a.broadcast = false;
        else if (k == "--check") a.firewallCheck = true;
        else if (k == "--print") a.firewallPrint = true;
        else if (k == "--remove") a.firewallRemove = true;
        else if (k == "--public") a.firewallPublic = true;
        else if (k == "--elevated") a.firewallElevated = true;
        else if (k == "--shaders") {
            a.shaders = next();
            if (a.shaders != "ask" && a.shaders != "compile" && a.shaders != "skip")
                throw std::runtime_error("--shaders takes ask, compile or skip");
        }
        else if (!k.empty() && k[0] != '-' && a.subject.empty() &&
                 (a.verb == Verb::Trust || a.verb == Verb::Approve || a.verb == Verb::Revoke)) a.subject = k;
        else throw std::runtime_error("unknown argument " + k + "\n" + usage(a.verb));
    }
    if (a.help || a.verb == Verb::Help) return a;
    if (a.verb == Verb::Master || a.verb == Verb::Node) {
        if (a.key.empty()) throw std::runtime_error("no rhizome key (--key)\n" + std::string(usage(a.verb)));
        if (a.cell == 0 || a.cell % 64) throw std::runtime_error("--cell must be a multiple of 64");
    }
    if (a.verb == Verb::Master) {
        if (a.scenes.empty() && a.queue.empty()) throw std::runtime_error("no scene (-in) and no queue (--queue)\n" + std::string(usage(a.verb)));
        if (!a.scenes.empty() && a.out.empty()) throw std::runtime_error("no picture to write (-out)\n" + std::string(usage(a.verb)));
    }
    if (a.verb == Verb::Submit && a.queue.empty()) throw std::runtime_error("no queue folder (--queue)\n" + std::string(usage(a.verb)));
    if (!a.scenes.empty() || a.verb == Verb::Submit) {
        rz::Job j;
        j.scenes = a.scenes, j.deltas = a.deltas, j.frames = a.frames, j.out = a.out;
        j.check();  // the frames read, the deltas named a frame each
    }
    if ((a.verb == Verb::Trust || a.verb == Verb::Approve || a.verb == Verb::Revoke) && !rz::parseHash(a.subject))
        throw std::runtime_error("a machine's hash, 64 hex digits, is needed\n" + std::string(usage(a.verb)));
    return a;
}
// A machine's folder for a verb: --folder, else rhizome\master or rhizome\node beside the executable (the identity
// commands take the master's when there is one, else the node's).
inline fs::path roleFolder(const RhizomeArgs& a) {
    if (!a.folder.empty()) return fs::u8path(a.folder);
    const fs::path base = executableFolder() / "rhizome";
    if (a.verb == Verb::Master) return base / "master";
    if (a.verb == Verb::Node) return base / "node";
    return fs::exists(base / "master" / "node.key") ? base / "master" : base / "node";
}
// The XPU2 build this machine runs: its executable and its GPU programs, hashed (machines whose samples would differ
// refuse each other, rhizome_node.h decide).
inline std::string buildId() {
    const fs::path folder = executableFolder();
    std::vector<unsigned char> all;
    for (const fs::path& p : {folder / "moonray_xpu2.exe", folder / "async_gpu.ptx"}) {
        std::ifstream in(p, std::ios::binary);
        std::vector<unsigned char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const rz::Hash h = rz::sha256(b);
        all.insert(all.end(), h.begin(), h.end());
    }
    return rz::hex(rz::sha256(all));
}

// A machine's XPU2: the scene prepared once, its lanes, and a share of the frame rendered on request. Everything runs
// on one thread of its own, the renderer's feeder: a graphics card's lane is used only from the thread that made it.
class Local {
public:
    // deltas: a frame's changes read on top of the scene before it is prepared (a node joining a job's later frame).
    Local(const RhizomeArgs& a, const fs::path& staging, const std::vector<std::string>& deltas = {}) : mode_(a.mode) {
        std::promise<void> ready;
        auto made = ready.get_future();
        thread_ = std::thread([&, this] {
            try {
                setup(a, staging, deltas);
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
    // A frame's changes (plan 13): its delta files read on top of the scene as it stands and the plan made again (the
    // camera alone, or the placements alone, where that is all they changed, as Modo's session does: the card keeps
    // its meshes and textures), the renderer made anew. Returns the seconds it took.
    double apply(const std::vector<std::string>& deltas) {
        return call([&] {
            const auto began = std::chrono::steady_clock::now();
            renderer_.reset();
            auditKey_ = {~0u, ~0u, 0, 0};
            for (const auto& d : deltas) scene_.apply(d);
            scene_.prepare();
            const char* how = "made again";
            if (gpu_) {
                if (scene_.cameraFrom && gpuPlan_ == scene_.cameraFrom) {
                    gpu_->updateCamera(scene_.plan);
                    how = "the camera alone";
                } else if (scene_.placedFrom && gpuPlan_ == scene_.placedFrom && gpu_->updatePlacements(scene_.plan)) {
                    how = "the placements alone";
                } else {
                    const std::string why = async::GpuLane::cpuOnly(scene_.plan);
                    if (!why.empty()) {
                        std::printf("MoonRay XPU2 Rhizome: the GPU leaves %s to the CPU; the CPU alone\n", why.c_str());
                        gpu_.reset();
                    } else {
                        gpu_->prepare(scene_.plan);
                    }
                }
            } else if (scene_.cameraFrom) {
                how = "the camera alone";
            }
            gpuPlan_ = scene_.plan;
            renderer_ = std::make_unique<async::Renderer>(scene_.plan, *scene_.cpuScene, gpu_.get());
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            std::printf("MoonRay XPU2 Rhizome: the frame's changes read (%zu files, %s) in %.2f s\n", deltas.size(), how, s);
            std::fflush(stdout);
            return s;
        });
    }
    // The scene's output files' names, as it stands.
    std::vector<std::string> outputFiles() {
        return call([&] {
            std::vector<std::string> v;
            if (scene_.plan->aovs)
                for (const auto& f : scene_.plan->aovs->files) v.push_back(f.name);
            return v;
        });
    }
    // A workstation in use: the processor threads its shares render on (0: all it was given).
    void setThreads(unsigned n) { busyThreads_ = n; }
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
    // The master's importance of each rhizome cell (rhizome_importance.h): the indexer's first-hit probe, a camera ray
    // every `spacing` pixels on the processor, what each hit scored. On the feeder, as the rest.
    std::vector<float> importance(const rz::FrameShape& shape, uint32_t spacing, double* seconds = nullptr) {
        return call([&] {
            const auto began = std::chrono::steady_clock::now();
            const cpu::Plan& p = *scene_.plan;
            const auto config = mixed::config(scene_.plan);
            async::CpuBackend backend{*scene_.cpuScene};
            backend.setTime(0.5f);
            const uint32_t w = (p.width + spacing - 1) / spacing, h = (p.height + spacing - 1) / spacing;
            std::vector<rz::ProbeHit> lattice(size_t(w) * h);
            volume::Crossing x[volume::MaxCrossings];
            for (uint32_t j = 0; j < h; ++j)
                for (uint32_t i = 0; i < w; ++i) {
                    rz::ProbeHit& out = lattice[size_t(j) * w + i];
                    const float px = std::min(float(p.width) - 0.5f, float(i * spacing + spacing / 2) + 0.5f);
                    const float py = std::min(float(p.height) - 0.5f, float(j * spacing + spacing / 2) + 0.5f);
                    surface::Ray ray;
                    if (surface::cameraSample(config.camera, px, py, 0.5f, 0.5f, 0.5f, ray) != surface::CameraResult::Made) continue;
                    ray.mask = 1u;  // a camera ray
                    const auto hit = backend.trace(ray);
                    if (hit.geom != ~0u && hit.geom < p.meshes.size()) {
                        out.object = hit.geom;
                        out.depth = hit.t;
                        out.families = p.meshes[hit.geom].families;
                    }
                    const float until = out.object != ~0u ? hit.t : 3.402823466e38f;
                    const U32 n = backend.volumeCrossings(ray.origin, ray.direction, ray.nearT, x, volume::MaxCrossings, 1u);
                    for (U32 k = 0; k < n && x[k].t < until; ++k) {
                        if (!(x[k].flags & volume::Entry) || x[k].volume >= p.volumes.size()) continue;
                        const volume::Shader& v = p.volumes[x[k].volume];
                        out.volumes = uint8_t(std::min(255, out.volumes + 1));
                        out.glow = out.glow || v.emissionGrid != volume::NoGrid || v.emission.x + v.emission.y + v.emission.z > 0.0f;
                    }
                }
            if (seconds) *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
            return rz::importanceFromProbe(lattice, w, h, spacing, shape);
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
    void setup(const RhizomeArgs& a, const fs::path& staging, const std::vector<std::string>& deltas) {
        threads_ = a.threads ? a.threads : multi::logicalProcessors();
        scene_.coarseFirst = false;
        scene_.dso = executableFolder();
        scene_.staging = staging;
        std::error_code ignored;
        fs::remove_all(staging, ignored);
        fs::create_directories(staging);
        cpu::sequenceDirectory() = executableFolder();
        scene_.load(a.scenes);
        for (const auto& d : deltas) scene_.apply(d);
        scene_.prepare();
        // A card that cannot take the job (its programs would not load, a driver refusing them): the node renders it on
        // its processor rather than sitting the job out, as it does when the programs are missing or still compiling.
        if (mode_ != 0) {
            try {
                makeGpu();
            } catch (const std::exception& e) {
                gpu_.reset();
                std::printf("MoonRay XPU2 Rhizome: the graphics card could not take this job (%s): the CPU alone\n", e.what());
            }
        }
        renderer_ = std::make_unique<async::Renderer>(scene_.plan, *scene_.cpuScene, gpu_.get());
        std::printf("MoonRay XPU2 Rhizome: scene ready (%ux%u, %u samples a pixel), %s, %u threads\n", plan().width, plan().height, plan().spp,
                    gpu_ ? (mode_ == 1 ? "the GPU alone" : "CPU and GPU") : "the CPU alone", threads_);
        std::fflush(stdout);
    }
    rz::Film renderHere(const uint32_t rect[4], uint32_t first, uint32_t count) {
        async::RenderOptions o;
        const unsigned busy = busyThreads_;
        o.threads = busy ? std::min(busy, threads_) : threads_;
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
        if (r.crypto && !r.crypto->empty()) {
            rz::Writer cw;
            const U32 cats = r.crypto->categories();
            cw.u32(cats);
            for (uint32_t y = rect[1]; y < rect[3]; ++y)
                for (uint32_t x = rect[0]; x < rect[2]; ++x)
                    for (U32 c = 0; c < cats; ++c) {
                        const auto& list = r.crypto->fragments(c, size_t(y) * W + x);
                        cw.u32(uint32_t(list.size()));
                        cw.raw(list.data(), list.size() * sizeof(aov::CryptoBuffer::Fragment));
                    }
            f.crypto = std::move(cw.b);
        }
        return f;
    }
    // The master's picture made whole: the merged films' AOVs resolved, the display filters run, and the scene's
    // render outputs written as a frame writes them (aov_write.h). film: sums a pixel; odd: odd-index sums.
    void writeOutputs(const std::vector<double>& film, const std::vector<uint32_t>& counts, const std::vector<double>& odd,
                      const std::vector<double>& raw, const std::vector<float>& depths, const aov::CryptoBuffer* crypto = nullptr,
                      const std::function<std::string(const std::string&)>& rename = {}) {
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
            // Cryptomatte: every node's fragments merged at the master, finalized once over the pixels' samples.
            if (scene_.plan->aovs->cryptomatte && crypto && !crypto->empty()) {
                result.crypto = std::make_shared<aov::CryptoBuffer>(*crypto);
                result.crypto->finalize(result.counts);
            }
            std::vector<std::string> warnings;
            if (!scene_.displayFilters.empty()) {
                std::vector<std::vector<float>> filtered;
                scene_.displayFilters.run(*scene_.plan->aovs, result, filtered);
                size_t k = 0;
                for (const aov::Output* out : scene_.plan->aovs->all)
                    if (out->result == aov::ResultDisplayFilter && k < filtered.size()) result.displayFilters[out->name] = std::move(filtered[k++]);
            }
            // A frame of a job: the files' names for it (rename) while they are written, then as they were. The plan
            // holds its outputs' state machine and cannot be copied; only this thread (the feeder) reads the names.
            auto& files = const_cast<std::vector<aov::File>&>(scene_.plan->aovs->files);
            std::vector<std::string> names;
            for (auto& f : files) names.push_back(f.name);
            if (rename)
                for (auto& f : files) f.name = rename(f.name);
            std::string error;
            try {
                error = aov::writeOutputs(*scene_.plan->aovs, result, plan().width, plan().height, warnings);
            } catch (...) {
                for (size_t i = 0; i < files.size(); ++i) files[i].name = names[i];
                throw;
            }
            for (size_t i = 0; i < files.size(); ++i) files[i].name = names[i];
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
        if (gpu::cardCount() > 0 && !gpu::anySupported()) {
            std::printf("MoonRay XPU2 Rhizome: %s is older than XPU2's GPU programs run on: the CPU alone\n", olderCards().c_str());
            return;
        }
        if (!fs::exists(shaderMark(p))) {
            std::printf("MoonRay XPU2 Rhizome: the GPU programs are not compiled here (moonray_xpu2 --compile-shaders): the CPU alone\n");
            return;
        }
        const std::set<std::string> held = compiledKinds(shaderMark(p));
        std::vector<int> cards;
        for (int card : multi::cardOrdinals(gpu::cardCount()))
            if (gpu::supported(card) && held.count(cardKind(card))) cards.push_back(card);  // a card too old: the CPU alone
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
        gpuPlan_ = scene_.plan;
    }
private:
    Scene scene_;
    cpu::PlanView gpuPlan_;  // the plan the card's scene was made for (a frame's changes: the camera or placements alone)
    std::unique_ptr<async::GpuLane> gpu_;
    std::unique_ptr<async::Renderer> renderer_;
    int mode_ = 2;
    unsigned threads_ = 1;
    std::atomic<unsigned> busyThreads_{0};
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
// A texture's pieces are smaller (plan 16): a node reads it from the pool as it renders, a piece for each tile read, so
// a smaller piece fetches less of what the render never reads.
constexpr size_t TexturePieceBytes = 256u << 10;
inline bool textureFile(const std::string& name) {
    std::string ext = fs::u8path(name).extension().u8string();
    for (char& c : ext) c = char(std::tolower((unsigned char)c));
    return ext == ".tx";
}
inline size_t pieceBytesFor(const std::string& name, bool scene) { return !scene && textureFile(name) ? TexturePieceBytes : PieceBytes; }
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
        rz::Bytes block(pieceBytesFor(f.name, scene));
        for (;;) {
            in.read(reinterpret_cast<char*>(block.data()), std::streamsize(block.size()));
            const size_t got = size_t(in.gcount());
            if (!got) break;
            rz::Bytes data(block.begin(), block.begin() + got);
            const rz::Hash h = rz::sha256(data);
            node.addPiece(rz::Piece{h, data.size(), rz::Tier::Ram}, data, false);
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
    node.announcePlacements();  // every piece placed, then the pool's list sent once
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
// Plan 16: whether a scene's file stays in the swarm's memory on a node, read as the renderer reads it (a texture: its
// tiles; remote_files.h), an empty file of its name in its place for whatever lists the folder (a UDIM set's tiles).
inline bool& wholeTextures() {
    static bool whole = false;
    return whole;
}
inline bool readFromPool(const SceneFile& f) {
    return !f.scene && !wholeTextures() && f.bytes != 0 && textureFile(f.name);
}
// The textures read from the swarm's memory so far: their blocks fetched and the files' bytes in all.
struct PoolTextures {
    std::mutex m;
    std::vector<std::shared_ptr<xpu2::remote::Source>> sources;
};
inline PoolTextures& poolTextures() {
    static PoolTextures p;
    return p;
}
inline std::string poolTexturesLine() {
    auto& p = poolTextures();
    std::lock_guard<std::mutex> g(p.m);
    uint64_t held = 0, total = 0;
    size_t blocks = 0, all = 0;
    for (const auto& s : p.sources) held += s->bytesHeld(), total += s->size(), blocks += s->blocksHeld(), all += s->blocks();
    if (p.sources.empty()) return {};
    char t[200];
    std::snprintf(t, sizeof t, "textures from the swarm's memory: %zu files, %.1f of %.1f MB fetched (%zu of %zu blocks)", p.sources.size(),
                  double(held) / (1 << 20), double(total) / (1 << 20), blocks, all);
    return t;
}
// A node: the scene's files fetched from the pool and written into folder, the scene files' strings pointed at them
// (its textures left in the pool, readFromPool). Returns the scene files to load.
inline std::vector<std::string> receiveScene(rz::RhizomeNode& node, const rz::Bytes& manifest, const fs::path& folder) {
    rz::Reader r(manifest);
    std::vector<SceneFile> files(r.u32());
    std::vector<rz::Hash> all;
    for (auto& f : files) {
        f.literal = r.str(), f.name = r.str(), f.scene = r.u8() != 0, f.bytes = r.u64();
        f.pieces.resize(r.u32());
        for (auto& h : f.pieces) h = r.hash();
        if (!readFromPool(f)) all.insert(all.end(), f.pieces.begin(), f.pieces.end());
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
        if (readFromPool(f)) {
            // Its blocks fetched from the pool the first time a read touches them (from a holder, else the master).
            auto pieces = std::make_shared<const std::vector<rz::Hash>>(f.pieces);
            rz::RhizomeNode* from = &node;
            auto source = std::make_shared<xpu2::remote::Source>(f.bytes, pieceBytesFor(f.name, false), [from, pieces](size_t i) -> xpu2::remote::Block {
                if (i >= pieces->size()) return nullptr;
                const rz::Hash& h = (*pieces)[i];
                if (!from->holds(h) && !from->fetch({h}, 300)) return nullptr;
                auto b = std::make_shared<std::vector<unsigned char>>(from->piece(h));
                return b->empty() ? nullptr : b;
            });
            xpu2::remote::add(path.string(), source);
            {
                std::lock_guard<std::mutex> g(poolTextures().m);
                poolTextures().sources.push_back(source);
            }
            if (!out) throw std::runtime_error("cannot write " + path.string());
            total += f.bytes;
            continue;
        }
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
    size_t pooled = 0;
    for (const auto& f : files) pooled += readFromPool(f) ? 1 : 0;
    std::printf("MoonRay XPU2 Rhizome: the scene from the swarm: %zu files, %.1f MB in %.1f s%s\n", files.size(), double(total) / (1 << 20),
                std::chrono::duration<double>(Clock::now() - began).count(),
                pooled ? (", " + std::to_string(pooled) + " textures read from the swarm's memory as the render reads them").c_str() : "");
    std::fflush(stdout);
    return scenes;
}

// A frame's manifest (plan 13): the job, the frame, the scene's files (a manifest of publishScene) and the frame's
// changes so far in the job, one manifest each, in the order they are read. A node that holds the scene and the first of
// the changes reads the rest alone; one that does not makes the scene and reads them all.
constexpr uint32_t ManifestMagic = 0x324D5A52u;  // "RZM2"
struct JobManifest {
    std::string job;
    uint32_t frame = 0, frames = 0;  // the frame, and the job's count of frames (0: the scene as it is)
    rz::Bytes scene;
    std::vector<rz::Bytes> deltas;
};
inline rz::Bytes writeManifest(const JobManifest& m) {
    rz::Writer w;
    w.u32(ManifestMagic);
    w.str(m.job);
    w.u32(m.frame), w.u32(m.frames);
    w.bytes(m.scene);
    w.u32(uint32_t(m.deltas.size()));
    for (const auto& d : m.deltas) w.bytes(d);
    return w.b;
}
inline JobManifest readManifest(const rz::Bytes& b) {
    JobManifest m;
    rz::Reader r(b);
    if (r.u32() != ManifestMagic) {
        m.scene = b;  // a scene alone
        return m;
    }
    m.job = r.str();
    m.frame = r.u32(), m.frames = r.u32();
    m.scene = r.bytes();
    const uint32_t n = r.u32();
    if (n > 1000000) throw std::runtime_error("the master's manifest does not read");
    for (uint32_t i = 0; i < n && !r.bad; ++i) m.deltas.push_back(r.bytes());
    if (r.bad) throw std::runtime_error("the master's manifest does not read");
    return m;
}
// A job's checkpoint (plan 14): which job (the XPU2 build and the job, hashed), its frames done, and the frame under way
// so far: its number, its manifest's hash (the scene's files and changes as they were: another scene's partial frame
// is not used) and the swarm's state of it (RhizomeNode::checkpoint). Written whole, then named.
struct JobCheckpoint {
    rz::Hash job{};
    rz::Hash files{};       // the job's scene and delta files as they were (each one's path, size and time)
    bool finished = false;  // every frame done: the same job again is skipped
    std::vector<int> done;
    bool partial = false;
    int frame = 0;
    rz::Hash manifest{};
    rz::Bytes state;
};
constexpr uint32_t CheckpointMagic = 0x324A5A52u;  // "RZJ2"
// The job's files as they stand: every frame's scene and delta files, each one's path, size and last write. A file
// written again changes it (the files those name, textures and the like, are not looked at).
inline rz::Hash jobFiles(const rz::Job& job) {
    std::string all;
    auto add = [&](const std::string& path) {
        std::error_code e;
        const fs::path f = fs::u8path(path);
        const auto size = fs::file_size(f, e);
        const auto when = fs::last_write_time(f, e).time_since_epoch().count();
        all += path + "|" + std::to_string(e ? 0 : size) + "|" + std::to_string(e ? 0 : (long long)when) + "\n";
    };
    const std::vector<int> frames = job.frameList();
    if (frames.empty())
        for (const auto& s : job.scenes) add(s);
    for (int f : frames) {
        for (const auto& s : job.scenesFor(f)) add(s);
        for (const auto& d : job.deltasFor(f)) add(d);
    }
    return rz::sha256(rz::Bytes(all.begin(), all.end()));
}
inline void writeCheckpoint(const fs::path& path, const JobCheckpoint& c) {
    rz::Writer w;
    w.u32(CheckpointMagic);
    w.hash(c.job);
    w.hash(c.files);
    w.u8(c.finished);
    w.u32(uint32_t(c.done.size()));
    for (int f : c.done) w.u32(uint32_t(f));
    w.u8(c.partial);
    w.u32(uint32_t(c.frame));
    w.hash(c.manifest);
    w.bytes(c.state);
    std::error_code e;
    fs::create_directories(path.parent_path(), e);
    const fs::path part = path.u8string() + ".part";
    {
        std::ofstream out(part, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(w.b.data()), std::streamsize(w.b.size()));
        if (!out) {
            std::printf("MoonRay XPU2 Rhizome: the checkpoint could not be written: %s\n", part.u8string().c_str());
            return;
        }
    }
    fs::rename(part, path, e);
    if (e) std::printf("MoonRay XPU2 Rhizome: the checkpoint could not be named %s: %s\n", path.u8string().c_str(), e.message().c_str());
}
inline bool readCheckpoint(const fs::path& path, JobCheckpoint& c) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    const rz::Bytes b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    rz::Reader r(b);
    if (r.u32() != CheckpointMagic) return false;
    c.job = r.hash();
    c.files = r.hash();
    c.finished = r.u8() != 0;
    const uint32_t n = r.u32();
    if (n > 10000000) return false;
    c.done.clear();
    for (uint32_t i = 0; i < n && !r.bad; ++i) c.done.push_back(int(r.u32()));
    c.partial = r.u8() != 0;
    c.frame = int(r.u32());
    c.manifest = r.hash();
    c.state = r.bytes();
    return !r.bad;
}

// A scene's output file name for a frame of a job: its run of # filled; without one, in a job of more than one frame,
// .NNNN added, unless the name already has the frame's number in it (the scene names its frames itself).
inline std::string frameOutputName(const std::string& name, int frame, bool many) {
    if (rz::hasFrameSlot(name) || !many) return rz::frameOutput(name, frame, many);
    const size_t from = name.find_last_of("/\\") == std::string::npos ? 0 : name.find_last_of("/\\") + 1;
    for (size_t i = from; i < name.size();) {
        if (name[i] < '0' || name[i] > '9') {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < name.size() && name[j] >= '0' && name[j] <= '9') ++j;
        if (j - i <= 9 && std::stoll(name.substr(i, j - i)) == frame) return name;
        i = j;
    }
    return rz::frameOutput(name, frame, true);
}

// The merge of a frame's AOVs and aux outputs, as films are taken in (by each output's filter, async::mergeAovPixel):
// the master's, and its deputy's copy (failover), which a new deputy is given whole.
struct Merge {
    aov::Tables tables;
    size_t pixels = 0;
    uint32_t width = 0;
    std::vector<double> aovs, odd;
    std::vector<float> depths;
    std::shared_ptr<aov::CryptoBuffer> crypto;  // Cryptomatte's fragments as sums, every node's merged
    bool multiPresence = false;
    void init(Local& l) {
        tables = l.aovTables();
        width = l.plan().width;
        pixels = size_t(l.plan().width) * l.plan().height;
        aovs.clear(), depths.clear(), odd.clear();
        if (tables.channels) {
            aovs.resize(pixels * tables.channels);
            depths.resize(pixels * std::max<U32>(tables.closestCount, 1));
            for (size_t p = 0; p < pixels; ++p) async::initAovPixel(tables, aovs.data() + p * tables.channels, depths.data() + p * tables.closestCount);
        }
        if (l.plan().aovs && (l.plan().aovs->beautyAux || l.plan().aovs->alphaAux)) odd.assign(pixels * 4, 0.0);
        crypto.reset();
        if (l.plan().aovs && l.plan().aovs->cryptomatte) {
            crypto = std::make_shared<aov::CryptoBuffer>();
            multiPresence = l.plan().aovs->cryptoMultiPresence;
            crypto->init(l.plan().width, l.plan().height, l.plan().aovs->cryptoCategories, multiPresence);
        }
    }
    // A film's (or a saved merge's) fragments, pixels in the rectangle's rows: each category's list, merged by ID.
    bool addCrypto(const rz::Bytes& b, const uint32_t rect[4]) {
        if (!crypto || b.empty()) return true;
        rz::Reader r(b);
        const U32 cats = r.u32();
        if (cats != crypto->categories()) return false;
        for (uint32_t y = rect[1]; y < rect[3] && !r.bad; ++y)
            for (uint32_t x = rect[0]; x < rect[2] && !r.bad; ++x)
                for (U32 c = 0; c < cats && !r.bad; ++c) {
                    const uint32_t n = r.u32();
                    if (n > 4096) return false;
                    for (uint32_t k = 0; k < n && !r.bad; ++k) {
                        aov::CryptoBuffer::Fragment g;
                        if (!r.take(&g, sizeof g)) break;
                        crypto->merge(size_t(y) * width + x, c, g);
                    }
                }
        return !r.bad;
    }
    void add(const rz::Film& f) {
        size_t i = 0;
        for (uint32_t y = f.rect[1]; y < f.rect[3]; ++y)
            for (uint32_t x = f.rect[0]; x < f.rect[2]; ++x, ++i) {
                const size_t p = size_t(y) * width + x;
                if (p >= pixels) continue;
                if (!odd.empty() && !f.odd.empty())
                    for (int c = 0; c < 4; ++c) odd[p * 4 + c] += f.odd[i * 4 + c];
                if (tables.channels && f.aovs.size() >= (i + 1) * tables.channels)
                    async::mergeAovPixel(tables, aovs.data() + p * tables.channels, depths.data() + p * tables.closestCount,
                                         f.aovs.data() + i * tables.channels, f.aovDepths.data() + i * tables.closestCount);
            }
        if (!addCrypto(f.crypto, f.rect)) std::printf("MoonRay XPU2 Rhizome: a film's Cryptomatte did not read; its fragments are left out\n");
    }
    // The whole frame's fragments, as a film's (a deputy's copy).
    rz::Bytes saveCrypto() const {
        rz::Writer w;
        if (!crypto) return w.b;
        const U32 cats = crypto->categories();
        w.u32(cats);
        for (size_t p = 0; p < pixels; ++p)
            for (U32 c = 0; c < cats; ++c) {
                const auto& list = crypto->fragments(c, p);
                w.u32(uint32_t(list.size()));
                w.raw(list.data(), list.size() * sizeof(aov::CryptoBuffer::Fragment));
            }
        return w.b;
    }
    rz::Bytes save() const {
        rz::Writer w;
        w.array(aovs), w.array(depths), w.array(odd);
        w.bytes(saveCrypto());
        return w.b;
    }
    void load(const rz::Bytes& b) {
        rz::Reader r(b);
        auto a = r.array<double>();
        auto d = r.array<float>();
        auto o = r.array<double>();
        const rz::Bytes c = r.bytes();
        if (r.bad || a.size() != aovs.size() || d.size() != depths.size() || o.size() != odd.size()) return;
        aovs = std::move(a), depths = std::move(d), odd = std::move(o);
        if (crypto) {
            crypto->init(width, uint32_t(pixels / std::max<uint32_t>(1, width)), crypto->categories(), multiPresence);
            const uint32_t all[4] = {0, 0, width, uint32_t(pixels / std::max<uint32_t>(1, width))};
            addCrypto(c, all);
        }
    }
};

// The identity verbs: id, trust, approve, revoke, status. No renderer, no network.
inline int identityVerb(const RhizomeArgs& a) {
    const fs::path folder = roleFolder(a);
    if (a.verb == Verb::Status) {
        const fs::path text = folder / "status.txt", json = folder / "status.json";
        // Running: its status written within the last 5 s (it writes each second).
        auto running = [&] {
            std::error_code e;
            if (!fs::exists(json, e)) return false;
            return std::chrono::duration<double>(fs::file_time_type::clock::now() - fs::last_write_time(json, e)).count() <= 5.0;
        };
        if (a.json) {
            int code = 0;
            for (bool first = true; first || a.watch; first = false) {
                code = running() ? 0 : 3;
                std::string s;
                if (code == 0) {
                    std::ifstream in(json, std::ios::binary);
                    s.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
                } else {
                    // Nothing running: this folder's hash, when it has one.
                    std::string hash;
                    if (fs::exists(folder / "node.key")) hash = rz::hex(rz::Identity::load(folder)->hash());
                    s = "{\n  \"rhizome\": 1,\n  \"running\": false,\n  \"hash\": \"" + hash + "\"\n}\n";
                }
                std::fwrite(s.data(), 1, s.size(), stdout);
                std::fflush(stdout);
                if (a.watch) std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            return code;
        }
        for (bool first = true; first || a.watch; first = false) {
            if (a.watch) std::printf("\x1b[2J\x1b[H");
            std::error_code e;
            if (!fs::exists(text)) {
                const bool made = fs::exists(folder / "node.key");
                std::printf("No node or master is running from %s%s\n", folder.u8string().c_str(),
                            made ? "" : " (and this folder has no hash yet: rhizome id makes one)");
                if (made) {
                    auto id = rz::Identity::load(folder);
                    rz::Trust t(folder, *id);
                    std::printf("This machine  %s%s\n", rz::hex(id->hash()).c_str(), t.approver() ? "  approver" : "");
                    std::printf("Approvers     %zu trusted, %zu endorsement(s) held\n", t.approvers().size(), t.shown().size());
                }
            } else {
                const auto age = std::chrono::duration<double>(fs::file_time_type::clock::now() - fs::last_write_time(text, e)).count();
                std::ifstream in(text, std::ios::binary);
                const std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
                std::fwrite(s.data(), 1, s.size(), stdout);
                if (age > 5) std::printf("\n(not running now: written %.0f s ago)\n", age);
            }
            std::fflush(stdout);
            if (a.watch) std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return running() ? 0 : 3;
    }
    const bool made = !fs::exists(folder / "node.key");
    auto id = rz::Identity::load(folder);
    rz::Trust t(folder, *id);
    const rz::Hash subject = a.subject.empty() ? rz::Hash{} : *rz::parseHash(a.subject);
    switch (a.verb) {
    case Verb::Id:
        std::printf("%s\n", rz::hex(id->hash()).c_str());
        std::printf("(%s %s; %s; %zu approver(s) trusted, %zu endorsement(s) held)\n", made ? "made now in" : "kept in", folder.u8string().c_str(),
                    t.approver() ? "an approver" : "not an approver", t.approvers().size(), t.shown().size());
        return 0;
    case Verb::Trust:
        t.addApprover(subject);
        std::printf("trusted %s as an approver (%s)\n", rz::hex(subject).c_str(), folder.u8string().c_str());
        return 0;
    case Verb::Approve: {
        if (!t.approver()) {
            std::string who;
            for (const auto& h : t.approvers()) who += (who.empty() ? "" : ", ") + rz::shortHex(h);
            std::printf("this machine is not an approver: ask an approver (%s) to run: moonray_xpu2 rhizome approve %s\n",
                        who.empty() ? "none trusted yet" : who.c_str(), rz::hex(subject).c_str());
            return 2;
        }
        t.endorse(subject, a.approver ? rz::Role::Approver : rz::Role::Node);
        std::printf("approved %s%s, signed by %s; a master running from %s admits it within a second\n", rz::hex(subject).c_str(),
                    a.approver ? " as an approver" : "", rz::shortHex(id->hash()).c_str(), folder.u8string().c_str());
        return 0;
    }
    case Verb::Revoke:
        if (t.approver()) {
            t.revoke(subject);
            std::printf("revoked %s, signed by approver %s; passed through the swarm by a master running from %s\n", rz::hex(subject).c_str(),
                        rz::shortHex(id->hash()).c_str(), folder.u8string().c_str());
        } else {
            t.block(subject);
            std::printf("blocked %s on this machine only (it is not an approver: the others are not told)\n", rz::hex(subject).c_str());
        }
        return 0;
    default:
        return 0;
    }
}

// Whether a TCP port may be listened on here (a node's or master's own, else one the system picks).
inline bool portFree(const std::string& bind, uint16_t port) {
    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, bind.c_str(), &a.sin_addr);
    const bool ok = ::bind(s, (sockaddr*)&a, sizeof a) == 0;
    closesocket(s);
    return ok;
}

// Plan 15: the GPU programs on this machine. A machine whose cards have none compiled (moonray_xpu2 --compile-shaders,
// about an hour the first time) renders on its processor alone; on its first start it is asked whether to compile them
// (--shaders ask, the answer kept in its folder; no answer in 30 s: asked again the next start), --shaders compile and
// skip decide it beforehand. The compile runs as a process of its own at low priority, its lines in the folder's
// shaders-compile.txt, while the machine renders on its processor; its cards join from the next job. A start while
// an earlier start's compile still runs (the file held by it) does not start another. Its state is said in the swarm's
// heartbeats and shown in every machine's status.
class ShaderWork {
public:
    ShaderWork(const RhizomeArgs& a, const fs::path& folder) : folder_(folder) {
        if (a.mode == 0) {
            state_ = "not used (the processor alone)";
            return;
        }
        if (gpu::cardCount() == 0) {
            state_ = "no graphics card";
            return;
        }
        // Cards, but none XPU2's programs run on (a GTX 1080 Ti): nothing to compile, never asked; the processor alone.
        // Chosen by itself, for a node started by hand or as a service as for one started from the kit.
        if (!gpu::anySupported()) {
            state_ = "not used (" + olderCards() + " older than XPU2 runs on: the processor alone)";
            return;
        }
        const auto missing = missingCards();
        if (missing.empty()) {
            state_ = "compiled";
            return;
        }
        if (compileRunning()) {
            say("the GPU programs are being compiled by an earlier start (" + log().u8string() + "): the CPU alone until they are");
            compiling_ = true;
            state_ = "compiling";
            watch(nullptr);
            return;
        }
        std::string choice = a.shaders;
        const fs::path kept = folder_ / "shaders.txt";
        if (choice == "ask") {
            std::ifstream in(kept);
            std::string k;
            if (in >> k && (k == "compile" || k == "skip")) choice = k;
        }
        if (choice == "ask") {
            std::printf("MoonRay XPU2 Rhizome: the GPU programs are not compiled for %s here: this machine renders on its processor\n"
                        "  alone until they are. Compile them now (about an hour the first time; the processor renders meanwhile)?\n"
                        "  [y]es / [n]o (kept in %s; --shaders compile|skip decides it beforehand; no answer in 30 s: asked again\n"
                        "  next start) ",
                        missing.c_str(), kept.u8string().c_str());
            std::fflush(stdout);
            const std::string answer = askLine(30);
            if (!answer.empty() && (answer[0] == 'y' || answer[0] == 'Y')) choice = "compile";
            else if (!answer.empty() && (answer[0] == 'n' || answer[0] == 'N')) choice = "skip";
            if (choice != "ask") {
                std::ofstream out(kept, std::ios::trunc);
                out << choice << "\n";
            } else {
                std::printf("\n");
            }
        }
        if (choice == "compile") {
            begin();
        } else if (choice == "skip") {
            state_ = "missing: " + missing + " (declined here)";
            say("the GPU programs are not compiled for " + missing + " (declined; moonray_xpu2 --compile-shaders, or --shaders compile): the CPU alone");
        } else {
            state_ = "missing: " + missing;
            say("the GPU programs are not compiled for " + missing + " (moonray_xpu2 --compile-shaders, or --shaders compile): the CPU alone");
        }
    }
    ~ShaderWork() {
        {
            std::lock_guard<std::mutex> g(m_);
            stopping_ = true;
        }
        cv_.notify_all();
        if (watcher_.joinable()) watcher_.join();
        // The compile goes on without this run (its mark is left once it is done).
        if (process_) CloseHandle(process_);
    }
    std::string state() {
        std::lock_guard<std::mutex> g(m_);
        return state_;
    }
    // Reported each time it changes (the node's heartbeats).
    void follow(std::function<void(const std::string&)> f) {
        std::lock_guard<std::mutex> g(m_);
        report_ = std::move(f);
        if (report_) report_(state_);
    }

private:
    static void say(const std::string& s) {
        std::printf("MoonRay XPU2 Rhizome: %s\n", s.c_str());
        std::fflush(stdout);
    }
    fs::path log() const { return folder_ / "shaders-compile.txt"; }
    // The kinds of the cards this machine renders with that have no programs compiled, by name.
    static std::string missingCards() {
        std::string names;
        for (const auto& [card, kind] : uncompiledCards(shaderMark(programs(executableFolder())), false)) names += (names.empty() ? "" : ", ") + kind;
        return names;
    }
    // An earlier start's compile holds its file (written by it alone).
    bool compileRunning() const {
        if (!fs::exists(log())) return false;
        HANDLE h = CreateFileW(log().c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_SHARING_VIOLATION;
        CloseHandle(h);
        return false;
    }
    // A line from the console, or nothing in that many seconds (or no console: a service, a script).
    static std::string askLine(double seconds) {
        auto line = std::make_shared<std::promise<std::string>>();
        auto got = line->get_future();
        std::thread([line] {
            std::string s;
            if (!std::getline(std::cin, s)) s.clear();
            line->set_value(s);
        }).detach();
        if (got.wait_for(std::chrono::duration<double>(seconds)) != std::future_status::ready) return {};
        return got.get();
    }
    void begin() {
        SECURITY_ATTRIBUTES inherit{sizeof inherit, nullptr, TRUE};
        HANDLE out = CreateFileW(log().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inherit, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (out == INVALID_HANDLE_VALUE) {
            state_ = "missing: " + missingCards() + " (the compile did not start)";
            say("the GPU programs' compile did not start: " + log().u8string() + " cannot be written");
            return;
        }
        STARTUPINFOW si{};
        si.cb = sizeof si;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = INVALID_HANDLE_VALUE;
        si.hStdOutput = out;
        si.hStdError = out;
        PROCESS_INFORMATION pi{};
        std::wstring command = L"\"" + (executableFolder() / "moonray_xpu2.exe").wstring() + L"\" --compile-shaders";
        const BOOL made = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW, nullptr,
                                         executableFolder().c_str(), &si, &pi);
        CloseHandle(out);
        if (!made) {
            state_ = "missing: " + missingCards() + " (the compile did not start)";
            say("the GPU programs' compile did not start (error " + std::to_string(GetLastError()) + ")");
            return;
        }
        CloseHandle(pi.hThread);
        process_ = pi.hProcess;
        compiling_ = true;
        state_ = "compiling";
        say("compiling the GPU programs for " + missingCards() + " in the background (" + log().u8string() +
            "): the CPU alone meanwhile, the cards from the next job after");
        watch(process_);
    }
    // The compile's progress from its file every few seconds; its end (this start's process, or the file let go).
    void watch(HANDLE process) {
        watcher_ = std::thread([this, process] {
            rz::swarmThread();
            std::unique_lock<std::mutex> g(m_);
            while (!stopping_) {
                cv_.wait_for(g, std::chrono::seconds(3));
                if (stopping_) break;
                g.unlock();
                std::string last, line;
                {
                    std::ifstream in(log());
                    while (std::getline(in, line))
                        if (!line.empty()) last = line;
                }
                if (!last.empty() && last.back() == '\r') last.pop_back();
                bool ended = process ? WaitForSingleObject(process, 0) == WAIT_OBJECT_0 : !compileRunning();
                DWORD code = 0;
                if (ended && process) GetExitCodeProcess(process, &code);
                std::string next;
                if (ended) {
                    const std::string missing = missingCards();
                    next = missing.empty() ? "compiled" : "missing: " + missing + " (the compile failed: " + last.substr(0, 160) + ")";
                    say(missing.empty() ? "the GPU programs are compiled: the cards render from the next job"
                                        : "the GPU programs' compile failed (exit " + std::to_string(code) + "): " + last);
                } else {
                    // "MoonRay XPU2 shaders: 12 compile tasks done, 30 waiting, 120 s"
                    unsigned done = 0, waiting = 0, s = 0;
                    const size_t at = last.find(": ");
                    if (at != std::string::npos && std::sscanf(last.c_str() + at + 2, "%u compile tasks done, %u waiting, %u s", &done, &waiting, &s) == 3)
                        next = "compiling: " + std::to_string(done) + " of " + std::to_string(done + waiting) + " tasks, " + std::to_string(s) + " s";
                    else
                        next = "compiling";
                }
                g.lock();
                if (next != state_) {
                    state_ = next;
                    if (report_) report_(state_);
                }
                if (ended) break;
            }
        });
    }

    fs::path folder_;
    std::mutex m_;
    std::condition_variable cv_;
    std::string state_;
    std::function<void(const std::string&)> report_;
    std::thread watcher_;
    HANDLE process_ = nullptr;
    bool compiling_ = false, stopping_ = false;
};

// Plan 18: the firewall rule its user adds once. Windows Firewall lets another machine reach this one's node or master
// only through an inbound rule; a node runs headless, where Windows' own question when a program first listens is not
// seen. One rule, named for Rhizome, lets moonray_xpu2.exe (this one, by its path) take connections on private and
// domain networks (--public: public ones too). Adding it needs an administrator: asked for (Windows' own prompt) when
// this is not one. --check: whether it is there (read from the firewall's own list, nothing changed); --print: the
// commands, run by hand; --remove: the rule taken away.
constexpr const wchar_t* FirewallRuleName = L"MoonRay XPU2 Rhizome";
// The programs the inbound allowing rules of that name are for (the firewall's list in the registry, read only).
inline std::vector<std::wstring> firewallRulePrograms() {
    std::vector<std::wstring> out;
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Services\\SharedAccess\\Parameters\\FirewallPolicy\\FirewallRules", 0,
                      KEY_READ, &key) != ERROR_SUCCESS)
        return out;
    const std::wstring name = std::wstring(L"|Name=") + FirewallRuleName + L"|";
    for (DWORD i = 0;; ++i) {
        wchar_t value[512];
        DWORD valueLen = 512, type = 0, bytes = 0;
        LONG got = RegEnumValueW(key, i, value, &valueLen, nullptr, &type, nullptr, &bytes);
        if (got == ERROR_NO_MORE_ITEMS) break;
        if (got != ERROR_SUCCESS || type != REG_SZ || bytes == 0) continue;
        std::wstring data(bytes / sizeof(wchar_t) + 1, L'\0');
        valueLen = 512;
        DWORD size = DWORD(data.size() * sizeof(wchar_t));
        if (RegEnumValueW(key, i, value, &valueLen, nullptr, &type, reinterpret_cast<BYTE*>(data.data()), &size) != ERROR_SUCCESS) continue;
        data.resize(wcslen(data.c_str()));
        if (data.find(name) == std::wstring::npos || data.find(L"|Dir=In|") == std::wstring::npos ||
            data.find(L"|Action=Allow|") == std::wstring::npos || data.find(L"|Active=TRUE|") == std::wstring::npos)
            continue;
        const size_t app = data.find(L"|App=");
        if (app == std::wstring::npos) continue;
        const size_t end = data.find(L'|', app + 5);
        out.push_back(data.substr(app + 5, end == std::wstring::npos ? std::wstring::npos : end - app - 5));
    }
    RegCloseKey(key);
    return out;
}
inline fs::path selfExecutable() {
    wchar_t self[32768];
    const DWORD n = GetModuleFileNameW(nullptr, self, 32768);
    return fs::path(std::wstring(self, n));
}
// Whether the firewall lets this executable take connections from other machines (a rule of Rhizome's for its path).
inline bool firewallAllowsThis() {
    const std::wstring self = selfExecutable().wstring();
    for (const auto& p : firewallRulePrograms())
        if (_wcsicmp(p.c_str(), self.c_str()) == 0) return true;
    return false;
}
inline bool runningElevated() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    TOKEN_ELEVATION e{};
    DWORD got = 0;
    const bool ok = GetTokenInformation(token, TokenElevation, &e, sizeof e, &got) && e.TokenIsElevated;
    CloseHandle(token);
    return ok;
}
// A command run without a window; its exit code (-1: it did not start).
inline int runQuietly(std::wstring command) {
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return -1;
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    return int(code);
}
inline std::string narrow(const std::wstring& w) { return fs::path(w).u8string(); }
inline int firewallVerb(const RhizomeArgs& a) {
    const std::wstring exe = selfExecutable().wstring();
    const std::wstring remove = std::wstring(L"netsh advfirewall firewall delete rule name=\"") + FirewallRuleName + L"\"";
    const std::wstring add = std::wstring(L"netsh advfirewall firewall add rule name=\"") + FirewallRuleName +
                             L"\" dir=in action=allow enable=yes profile=private,domain" + (a.firewallPublic ? L",public" : L"") + L" program=\"" + exe + L"\"";
    auto state = [&] {
        if (firewallAllowsThis()) {
            std::printf("MoonRay XPU2 Rhizome: the firewall lets other machines reach this one (%s)\n", narrow(exe).c_str());
            return 0;
        }
        const auto others = firewallRulePrograms();
        if (others.empty()) std::printf("MoonRay XPU2 Rhizome: no firewall rule for Rhizome here: other machines may not reach this one\n");
        else std::printf("MoonRay XPU2 Rhizome: the firewall's Rhizome rule is for another copy (%s), not this one\n", narrow(others.front()).c_str());
        std::printf("  moonray_xpu2 rhizome firewall adds it (asking for an administrator)\n");
        return 3;
    };
    if (a.firewallCheck) return state();
    if (a.firewallPrint) {
        std::printf("Run these in a command prompt opened as an administrator:\n  %s\n", narrow(remove).c_str());
        if (!a.firewallRemove) std::printf("  %s\n", narrow(add).c_str());
        return 0;
    }
    if (!runningElevated()) {
        if (a.firewallElevated) {
            std::printf("MoonRay XPU2 Rhizome: not an administrator: the rule is not changed\n");
            return 5;
        }
        // Windows' own prompt for an administrator; this same command again behind it.
        std::wstring params = L"rhizome firewall --elevated";
        if (a.firewallRemove) params += L" --remove";
        if (a.firewallPublic) params += L" --public";
        SHELLEXECUTEINFOW sei{};
        sei.cbSize = sizeof sei;
        sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NO_CONSOLE;
        sei.lpVerb = L"runas";
        sei.lpFile = exe.c_str();
        sei.lpParameters = params.c_str();
        sei.nShow = SW_HIDE;
        if (!ShellExecuteExW(&sei) || !sei.hProcess) {
            std::printf("MoonRay XPU2 Rhizome: %s: the rule is not changed\n",
                        GetLastError() == ERROR_CANCELLED ? "the administrator prompt was declined" : "the administrator prompt did not open");
            return 4;
        }
        WaitForSingleObject(sei.hProcess, INFINITE);
        CloseHandle(sei.hProcess);
    } else {
        runQuietly(remove);  // the rule of an earlier copy (another path) too
        if (!a.firewallRemove && runQuietly(add) != 0) {
            std::printf("MoonRay XPU2 Rhizome: netsh did not add the rule\n");
            return 1;
        }
        if (a.firewallElevated) return 0;
    }
    if (a.firewallRemove) {
        const bool gone = firewallRulePrograms().empty();
        std::printf("MoonRay XPU2 Rhizome: %s\n", gone ? "the firewall rule is removed" : "the firewall rule is still there");
        return gone ? 0 : 1;
    }
    return state();
}

// Plan 18: the machines a master reached before, remembered in its folder (known.txt, an IPv4 address a line, the
// newest first, 256 at most) and probed one by one at its next start, where broadcast does not reach (a VPN, a routed
// network): found once by address or a sweep, found again by itself.
inline std::vector<std::string> readKnown(const fs::path& folder) {
    std::vector<std::string> out;
    std::ifstream in(folder / "known.txt");
    for (std::string line; std::getline(in, line);) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        in_addr x{};
        if (!line.empty() && inet_pton(AF_INET, line.c_str(), &x) == 1 && std::find(out.begin(), out.end(), line) == out.end()) out.push_back(line);
    }
    return out;
}
inline void remember(const fs::path& folder, const std::string& host) {
    in_addr x{};
    if (inet_pton(AF_INET, host.c_str(), &x) != 1) return;
    std::vector<std::string> known = readKnown(folder);
    known.erase(std::remove(known.begin(), known.end(), host), known.end());
    known.insert(known.begin(), host);
    if (known.size() > 256) known.resize(256);
    std::error_code e;
    const fs::path temp = folder / "known.txt.part";
    {
        std::ofstream out(temp, std::ios::trunc);
        for (const auto& k : known) out << k << "\n";
    }
    fs::rename(temp, folder / "known.txt", e);
}

inline int run(int argc, char** argv) {
    RhizomeArgs a = parseRhizome(argc, argv);
    wholeTextures() = a.wholeTextures;
    // This thread takes the node's lock too (checkpoints, status, the frame's start): above the renderer's, as the
    // swarm's own (a thread of normal priority holding the lock while the cores render holds up the heartbeats).
    rz::swarmThread();
    if (a.help || a.verb == Verb::Help) {
        std::printf("%s\n", usage(a.verb));
        return 0;
    }
    if (a.verb == Verb::Submit) {
        rz::Job j;
        for (const auto& s : a.scenes) j.scenes.push_back(fs::absolute(fs::u8path(s)).u8string());
        for (const auto& d : a.deltas) j.deltas.push_back(fs::absolute(fs::u8path(d)).u8string());
        j.frames = a.frames;
        j.out = fs::absolute(fs::u8path(a.out)).u8string();
        j.name = a.name.empty() ? fs::u8path(a.scenes.front()).filename().u8string() : a.name;
        const fs::path at = rz::queue::submit(fs::u8path(a.queue), j);
        std::printf("MoonRay XPU2 Rhizome: job %s queued: %s\n", j.name.c_str(), at.u8string().c_str());
        return 0;
    }
    if (a.verb == Verb::Firewall) return firewallVerb(a);
    if (a.verb != Verb::Node && a.verb != Verb::Master) return identityVerb(a);
    rz::Winsock winsock;
    const fs::path folder = roleFolder(a);
    const bool fresh = !fs::exists(folder / "node.key");
    auto identity = rz::Identity::load(folder);
    auto trust = std::make_shared<rz::Trust>(folder, *identity);
    for (const auto& t : a.trust) {
        const auto h = rz::parseHash(t);
        if (!h) throw std::runtime_error("--trust takes an approver's hash (64 hex digits): " + t);
        trust->addApprover(*h);
    }
    if (a.master && (a.approve || a.trustFirst)) {
        trust->require(true);
        trust->trustSelf();
    }
    std::printf("MoonRay XPU2 Rhizome: this machine is %s (%s %s)\n", rz::hex(identity->hash()).c_str(), fresh ? "made now in" : "kept in",
                folder.u8string().c_str());
    std::fflush(stdout);
    // Its GPU programs: compiled, or offered to be (plan 15).
    ShaderWork shaders(a, folder);
    const fs::path base = fs::temp_directory_path() / ("xpu2-rhizome-" + std::to_string(GetCurrentProcessId()));
    // The master reads each job's scene where it is (in the frame loop below); a node makes each job's from the swarm's
    // memory.
    std::shared_ptr<Local> current;
    std::mutex currentMutex;
    auto local = [&] {
        std::lock_guard<std::mutex> g(currentMutex);
        return current;
    };
    // The merge of the AOVs and aux outputs: the master's, or a deputy's copy of it (failover).
    Merge merge;
    rz::NodeOptions o;
    o.readyAtJoin = false;
    // A node's jobs: each manifest as it comes (the same one again is not a new job).
    std::mutex jobsMutex;
    std::condition_variable jobsCv;
    std::deque<rz::Bytes> manifests;
    o.onManifest = [&](const rz::Bytes& m) {
        {
            std::lock_guard<std::mutex> g(jobsMutex);
            manifests.push_back(m);
        }
        jobsCv.notify_all();
    };
    o.onFilm = [&](const rz::Film& f) { merge.add(f); };
    o.exportMerge = [&] { return merge.save(); };
    o.importMerge = [&](const rz::Bytes& b) { merge.load(b); };
    if (a.master) o.job = a.scenes.empty() ? std::string("the queue") : fs::u8path(a.scenes.front()).filename().u8string();
    // A node that takes over as master (its master failed): it waits for the picture, then writes it and the scene's
    // outputs as the master would have (the beauty beside its folder, output\<scene>.pfm).
    std::thread finisher;
    std::atomic<bool> finishing{false};
    // The frame a node holds the scene of, and its job's count of frames (0: no frames): a takeover's outputs named as
    // the master names them.
    std::atomic<int> jobFrame{0};
    std::atomic<uint32_t> jobFrames{0};
    rz::RhizomeNode* nodeRef = nullptr;
    o.onTakeover = [&] {
        if (finishing.exchange(true)) return;
        if (finisher.joinable()) finisher.join();  // the last takeover's, done (finishing was false)
        finisher = std::thread([&] {
            rz::RhizomeNode& n = *nodeRef;
            if (!n.waitFrame(24 * 3600.0)) return;
            const std::shared_ptr<Local> l = local();
            if (!l) return;
            const std::vector<double> film = n.film();
            const std::vector<uint32_t> counts = n.counts();
            const int frame = jobFrame;
            const uint32_t frames = jobFrames;
            const std::string stem = fs::u8path(o.job).stem().u8string();
            const fs::path out = folder / "output" / fs::u8path(frames ? rz::frameOutput(stem + ".pfm", frame, frames > 1) : stem + ".pfm");
            std::error_code e;
            fs::create_directories(out.parent_path(), e);
            writePfm(out, l->plan().width, l->plan().height, film, counts);
            std::function<std::string(const std::string&)> rename;
            if (frames) rename = [frame, frames](const std::string& n) { return frameOutputName(n, frame, frames > 1); };
            l->writeOutputs(film, counts, merge.odd, merge.aovs, merge.depths, merge.crypto.get(), rename);
            std::printf("MoonRay XPU2 Rhizome: took over as master and finished the frame: wrote %s\n", out.u8string().c_str());
            std::fflush(stdout);
            finishing = false;
        });
    };
    // Silence a real machine is allowed: a busy machine's (every core rendering, a large piece being sealed) can last a
    // second or two, and taking a live master for gone splits the swarm. A failed one is still noticed in seconds.
    o.suspectAfter = 2.0, o.goneAfter = 6.0;
    o.rhizomeKey = a.key;
    o.master = a.master;
    o.masterAddress = a.masterAddress;
    o.trust = trust;
    o.trustFirst = a.trustFirst;
    o.build = buildId();
    o.statusPath = (folder / "status.json").string();
    // Its own port, else (taken: another node or master on this machine) one the system picks.
    const uint16_t wanted = a.port >= 0 ? uint16_t(a.port) : a.master ? MasterPort : NodePort;
    o.bind = a.bind, o.advertise = a.advertise;
    o.port = wanted == 0 || portFree(a.bind, wanted) ? wanted : 0;
    if (o.port != wanted) std::printf("MoonRay XPU2 Rhizome: port %u is taken here: another one is used\n", unsigned(wanted));
    o.portFallback = a.port < 0;  // the default port: another one if it is taken as the node starts
    o.caps.cpuThreads = multi::logicalProcessors();
    o.caps.numaNodes = unsigned(std::max<size_t>(1, multi::nodes().size()));
    o.caps.cards = unsigned(gpu::cardCount());
    o.ramBytes = Local::ramBytes();
    o.render = [&](const uint32_t r[4], uint32_t first, uint32_t count) {
        const std::shared_ptr<Local> l = local();
        if (!l) throw std::runtime_error("the scene is not made here yet");
        const uint64_t fetchedBefore = xpu2::remote::fetchedBlocks().load();
        rz::Film f = l->render(r, first, count);
        f.waited = xpu2::remote::fetchedBlocks().load() != fetchedBefore;
        // The textures read from the swarm's memory so far (plan 16): said every 10 s while it renders, when it changed.
        static double said = 0;
        static std::string last;
        if (const double now = rz::nowSeconds(); now - said > 10) {
            said = now;
            if (std::string t = poolTexturesLine(); !t.empty() && t != last) {
                std::printf("MoonRay XPU2 Rhizome: %s\n", t.c_str());
                std::fflush(stdout);
                last = std::move(t);
            }
        }
        return f;
    };
    o.renderPixel = [&](uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) { local()->pixel(x, y, first, count, sum); };
    // A workstation: steps back while its machine is in use (or renders on fewer threads) and outside its hours.
    if (!a.master && (a.whenIdle > 0 || a.hoursFrom >= 0)) {
        o.mayWork = [&a, &local] {
            bool inHours = true;
            if (a.hoursFrom >= 0) {
                SYSTEMTIME t;
                GetLocalTime(&t);
                const int h = t.wHour;
                inHours = a.hoursFrom <= a.hoursTo ? (h >= a.hoursFrom && h < a.hoursTo) : (h >= a.hoursFrom || h < a.hoursTo);
            }
            bool idle = true;
            if (a.whenIdle > 0) {
                LASTINPUTINFO li{sizeof li, 0};
                if (GetLastInputInfo(&li)) idle = double(GetTickCount() - li.dwTime) / 1000.0 >= a.whenIdle;
            }
            const std::shared_ptr<Local> l = local();
            if (a.busyThreads && inHours) {
                if (l) l->setThreads(idle ? 0 : a.busyThreads);
                return true;
            }
            return inHours && idle;
        };
        std::printf("MoonRay XPU2 Rhizome: a workstation: %s%s%s\n",
                    a.whenIdle > 0 ? ("renders once idle for " + std::to_string(int(a.whenIdle)) + " s").c_str() : "renders whenever it may",
                    a.hoursFrom >= 0 ? (", from " + std::to_string(a.hoursFrom) + ":00 to " + std::to_string(a.hoursTo) + ":00").c_str() : "",
                    a.busyThreads ? (", on " + std::to_string(a.busyThreads) + " threads while in use").c_str() : "");
    }
    // The node's log lines: queued (they are said under the node's lock) and written by a thread of their own, so a
    // write that waits (the disk busy, a file being scanned) never holds up the heartbeats. RHIZOME_TRACE=1: each line
    // with the seconds since the start (who waited for whom).
    struct Logger {
        std::mutex m;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool stopping = false;
        std::thread writer;
        Logger() {
            writer = std::thread([this] {
                std::unique_lock<std::mutex> g(m);
                for (;;) {
                    cv.wait(g, [&] { return stopping || !lines.empty(); });
                    std::deque<std::string> out;
                    out.swap(lines);
                    const bool last = stopping;
                    g.unlock();
                    for (const auto& s : out) std::fputs(s.c_str(), stdout);
                    std::fflush(stdout);
                    g.lock();
                    if (last && lines.empty()) return;
                }
            });
        }
        ~Logger() {
            {
                std::lock_guard<std::mutex> g(m);
                stopping = true;
            }
            cv.notify_all();
            writer.join();
        }
        void say(std::string s) {
            {
                std::lock_guard<std::mutex> g(m);
                lines.push_back(std::move(s));
            }
            cv.notify_all();
        }
    };
    static Logger logger;
    o.log = [](const std::string& s) {
        static const bool timed = std::getenv("RHIZOME_TRACE") && std::getenv("RHIZOME_TRACE")[0] == '1';
        static const double began = rz::nowSeconds();
        char head[64];
        if (timed) std::snprintf(head, sizeof head, "MoonRay XPU2 Rhizome: %9.3f ", rz::nowSeconds() - began);
        else std::snprintf(head, sizeof head, "MoonRay XPU2 Rhizome: ");
        logger.say(head + s + "\n");
    };
    rz::RhizomeNode node(o, std::move(identity));
    nodeRef = &node;
    shaders.follow([&node](const std::string& s) { node.setShaders(s); });
    // Before the node goes: the compile's watcher no longer reports to it.
    struct Unfollow {
        ShaderWork& s;
        ~Unfollow() { s.follow(nullptr); }
    } unfollow{shaders};
    node.start();
    std::printf("MoonRay XPU2 Rhizome: %s %s listening at %s\n", a.master ? "master" : "node", rz::shortHex(node.hash()).c_str(),
                node.address().c_str());
    std::fflush(stdout);
    // Discovery: a node answers probes on every adapter; a master probes them and reaches each node of the rhizome
    // it finds (a direct cable's too), and those given by address.
    std::unique_ptr<rz::Responder> responder;
    std::unique_ptr<rz::Prober> prober;
    const uint16_t tcpPort = uint16_t(std::stoi(node.address().substr(node.address().rfind(':') + 1)));
    // Another machine reaches this one only through the firewall's rule: said once at the start when it is not there
    // (a test's machines on 127.0.0.1 need none).
    if (a.bind != "127.0.0.1" && !firewallAllowsThis())
        std::printf("MoonRay XPU2 Rhizome: the firewall has no rule for this program: other machines may not reach it "
                    "(moonray_xpu2 rhizome firewall adds one, once)\n");
    if (a.discovery) {
        try {
            if (a.master) {
                // Beyond broadcast (plan 18): the machines reached before, one by one, and the sweeps asked for.
                rz::ProbeOptions po;
                po.broadcast = a.broadcast;
                po.adapters = a.scanAdapters;
                for (const auto& s : a.scan) {
                    rz::Range r;
                    if (rz::parseRange(s, r)) po.ranges.push_back(r);
                }
                const std::vector<std::string> known = readKnown(folder);
                for (const auto& k : known) {
                    in_addr x{};
                    if (inet_pton(AF_INET, k.c_str(), &x) == 1) po.hosts.push_back(ntohl(x.s_addr));
                }
                if (!known.empty() || !po.ranges.empty() || po.adapters)
                    std::printf("MoonRay XPU2 Rhizome: probing %zu machines reached before%s%s\n", known.size(),
                                po.ranges.empty() ? "" : (", " + std::to_string(po.ranges.size()) + " ranges swept").c_str(),
                                po.adapters ? ", each adapter's subnet swept" : "");
                prober = std::make_unique<rz::Prober>(a.key, [&, folder](const rz::Hash& h, const std::string& at) {
                    if (h == node.hash()) return;
                    std::printf("MoonRay XPU2 Rhizome: found %s at %s\n", rz::shortHex(h).c_str(), at.c_str());
                    std::fflush(stdout);
                    remember(folder, at.substr(0, at.rfind(':')));
                    node.dial(at);
                }, rz::DiscoveryPort, 1.0, po);
            }
            else
                responder = std::make_unique<rz::Responder>(a.key, node.hash(), tcpPort);
        } catch (const std::exception& e) {
            std::printf("MoonRay XPU2 Rhizome: discovery is off here: %s\n", e.what());
        }
    }
    for (const auto& at : a.nodeAddresses) {
        // A machine given by address is remembered too (its host): probed at the next start without being given.
        if (a.master) remember(folder, at.substr(0, at.rfind(':')));
        node.dial(at);
    }
    if (!a.master) {
        // A node: each job's scene from the swarm's memory, made here, then ready for cells; then the next job.
        std::printf("MoonRay XPU2 Rhizome: node waiting for a master%s\n", a.masterAddress.empty() ? " to find it" : "");
        std::fflush(stdout);
        rz::Hash last{};
        // What the scene here is made of: the scene's manifest and the changes read into it since, in order.
        rz::Hash heldScene{};
        std::vector<rz::Hash> heldDeltas;
        for (unsigned job = 1;; ++job) {
            rz::Bytes m;
            {
                std::unique_lock<std::mutex> g(jobsMutex);
                jobsCv.wait(g, [&] { return !manifests.empty(); });
                m = std::move(manifests.back());  // frames it missed meanwhile: their changes are in this one's
                manifests.clear();
            }
            const rz::Hash which = rz::sha256(m);
            if (which == last) continue;  // the same job sent again (a master reaching it once more)
            last = which;
            try {
                const JobManifest jm = readManifest(m);
                const fs::path into = base / ("job" + std::to_string(job));
                std::vector<rz::Hash> deltas;
                for (const auto& d : jm.deltas) deltas.push_back(rz::sha256(d));
                const std::shared_ptr<Local> held = local();
                // The scene here, and its changes the first of this frame's: the rest of them read alone.
                const bool keep = held && heldScene == rz::sha256(jm.scene) && heldDeltas.size() <= deltas.size() &&
                                  std::equal(heldDeltas.begin(), heldDeltas.end(), deltas.begin());
                std::string name;
                if (keep) {
                    std::vector<std::string> files;
                    for (size_t i = heldDeltas.size(); i < jm.deltas.size(); ++i) {
                        const auto got = receiveScene(node, jm.deltas[i], into / ("delta" + std::to_string(i)));
                        files.insert(files.end(), got.begin(), got.end());
                    }
                    if (!files.empty()) held->apply(files);
                    heldDeltas = deltas;
                    merge.init(*held);  // the frame's copy, should it be a deputy (failover)
                } else {
                    RhizomeArgs j = a;
                    j.scenes = receiveScene(node, jm.scene, into);
                    std::vector<std::string> files;
                    for (size_t i = 0; i < jm.deltas.size(); ++i) {
                        const auto got = receiveScene(node, jm.deltas[i], into / ("delta" + std::to_string(i)));
                        files.insert(files.end(), got.begin(), got.end());
                    }
                    auto made = std::make_shared<Local>(j, base / ("staging" + std::to_string(job)), files);
                    {
                        std::lock_guard<std::mutex> g(currentMutex);
                        current = made;  // the last job's goes once no chunk of it is rendering
                    }
                    heldScene = rz::sha256(jm.scene);
                    heldDeltas = deltas;
                    merge.init(*made);  // its copy, should it be a deputy (failover)
                    name = fs::u8path(j.scenes.front()).filename().u8string();
                    o.job = name;
                }
                if (!jm.job.empty()) o.job = jm.job;
                jobFrame = int(jm.frame), jobFrames = jm.frames;
                const std::string what = jm.job.empty() ? o.job : jm.job + " frame " + std::to_string(jm.frame);
                node.setJob(what);
                node.markReady(which);
                std::printf("MoonRay XPU2 Rhizome: node ready for %s%s\n", what.c_str(), keep ? " (the scene kept, its changes read)" : "");
                if (const std::string t = poolTexturesLine(); !t.empty()) std::printf("MoonRay XPU2 Rhizome: %s\n", t.c_str());
            } catch (const std::exception& e) {
                std::printf("MoonRay XPU2 Rhizome: job %u could not be made here: %s\n", job, e.what());
            }
            std::fflush(stdout);
        }
    }
    // The master's jobs (plan 13): the one on its command line, then the queue's, one after another; each frame of each.
    std::vector<rz::Job> given;
    if (!a.scenes.empty()) {
        rz::Job j;
        j.scenes = a.scenes, j.deltas = a.deltas, j.frames = a.frames, j.out = a.out;
        j.name = a.name.empty() ? fs::u8path(a.scenes.front()).filename().u8string() : a.name;
        given.push_back(j);
    }
    const fs::path queueFolder = a.queue.empty() ? fs::path() : fs::u8path(a.queue);
    // The files placed in the pool, by their paths (a job's scene is placed once for all its frames).
    std::map<std::string, rz::Bytes> placed;
    uint64_t sceneBytes = 0;
    auto publishFiles = [&](const std::vector<std::string>& files) {
        std::string key;
        for (const auto& f : files) key += f + "\n";
        auto it = placed.find(key);
        if (it == placed.end()) it = placed.emplace(key, publishScene(node, files, sceneBytes)).first;
        return it->second;
    };
    std::vector<std::string> heldScenes;  // the files the master's scene was made of
    std::vector<rz::Bytes> heldDeltas;    // this job's changes read into it, in order
    uint32_t sequence = 0, staged = 0;
    bool firstFrame = true;
    // One frame of a job: the scene made (or its changes read), the nodes made ready, the frame rendered and written.
    // skip: a frame its checkpoint has done (its changes read, so the frames after it are right; nothing rendered).
    auto renderFrame = [&](const rz::Job& job, bool hasFrames, int frameNumber, size_t index, size_t total, JobCheckpoint& ck,
                           const fs::path& ckPath, bool skip) {
        if (skip && job.deltas.empty()) return;  // a frame made from its own files: nothing to carry to the next
        const bool many = total > 1;
        const std::vector<std::string> scenes = hasFrames ? job.scenesFor(frameNumber) : job.scenes;
        const std::vector<std::string> deltas = hasFrames ? job.deltasFor(frameNumber) : std::vector<std::string>{};
        std::shared_ptr<Local> l = local();
        if (!l || scenes != heldScenes) {
            RhizomeArgs j = a;
            j.scenes = scenes;
            auto made = std::make_shared<Local>(j, base / ("staging" + std::to_string(++staged)), deltas);
            {
                std::lock_guard<std::mutex> g(currentMutex);
                current = made;
            }
            l = made;
            heldScenes = scenes;
            heldDeltas.clear();
        } else if (!deltas.empty()) {
            l->apply(deltas);
        }
        if (!deltas.empty()) heldDeltas.push_back(publishFiles(deltas));
        if (skip) return;
        merge.init(*l);
        JobManifest jm;
        jm.job = job.name;
        jm.frame = uint32_t(frameNumber);
        jm.frames = hasFrames ? uint32_t(total) : 0u;
        jm.scene = publishFiles(scenes);
        jm.deltas = heldDeltas;
        const std::string what = hasFrames ? job.name + " frame " + std::to_string(frameNumber) + " (" + std::to_string(index + 1) + " of " +
                                                 std::to_string(total) + ")"
                                           : job.name;
        o.job = job.name;
        node.setJob(what);
        const rz::Bytes manifest = writeManifest(jm);
        const rz::Hash manifestHash = rz::sha256(manifest);
        node.publishManifest(manifest);
        // The nodes: for the run's first frame, as many as asked (--nodes) or given (--node) ready, else every one found
        // ready and none new for three seconds; for the frames after, every one there ready for this frame; --wait
        // seconds at most. A node not ready in time sits this frame out.
        const size_t needed = a.nodes ? a.nodes : a.nodeAddresses.size();
        const auto began = Clock::now();
        auto lastChange = Clock::now();
        size_t seen = 0;
        for (;;) {
            const double t = std::chrono::duration<double>(Clock::now() - began).count();
            const size_t joined = node.joined(), ready = node.ready();
            if (joined != seen) seen = joined, lastChange = Clock::now();
            const double quiet = std::chrono::duration<double>(Clock::now() - lastChange).count();
            if (firstFrame) {
                if (needed ? ready >= needed : (ready == joined && joined > 0 && quiet > 3.0)) break;
            } else if (ready >= joined) {
                break;
            }
            if (t > a.wait) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(firstFrame ? 100 : 20));
        }
        if (firstFrame) std::this_thread::sleep_for(std::chrono::milliseconds(500));  // the nodes meet each other
        firstFrame = false;
        const cpu::Plan& p = l->plan();
        rz::FrameShape shape;
        shape.width = p.width, shape.height = p.height, shape.cellSize = a.cell, shape.spp = p.spp, shape.chunkSamples = std::min(a.chunk, p.spp);
        shape.frame = ++sequence;  // the run's count: a late film of a frame before (even another job's) is not this one's
        // An adaptive scene: each cell stops once its error, pooled at the master from every node's films, is under the
        // scene's target (its most samples, spp, the rest); chunks of an even number of samples for the error's halves.
        if (p.adaptive) {
            shape.targetError = p.targetError, shape.minSamples = p.minSamples;
            if (shape.chunkSamples % 2) shape.chunkSamples = std::max(2u, shape.chunkSamples - 1);
            if (index == 0)
                std::printf("MoonRay XPU2 Rhizome: adaptive, %u to %u samples a pixel, a cell done under error %g\n", p.minSamples, p.spp,
                            double(p.targetError));
        }
        std::printf("MoonRay XPU2 Rhizome: %s: %zu nodes joined, %zu ready; the frame in %u cells of %u px, chunks of %u samples\n",
                    what.c_str(), node.joined(), node.ready(), shape.cells(), shape.cellSize, shape.chunkSamples);
        // The cells' importance from the indexer's probe: the important ones to the fast tier.
        double probeSeconds = 0;
        const std::vector<float> importance = l->importance(shape, 8, &probeSeconds);
        size_t above = 0;
        {
            double mean = 0;
            for (float v : importance) mean += v;
            mean /= std::max<size_t>(1, importance.size());
            for (float v : importance) above += v > mean;
        }
        std::printf("MoonRay XPU2 Rhizome: the indexer's probe in %.2f s: %zu of %u cells important\n", probeSeconds, above, shape.cells());
        const auto frameBegan = Clock::now();
        // The frame so far from the job's checkpoint, when it is this frame of the same scene.
        const rz::Bytes* resume = nullptr;
        if (ck.partial && ck.frame == frameNumber) {
            if (ck.manifest == manifestHash) resume = &ck.state;
            else std::printf("MoonRay XPU2 Rhizome: the scene changed since its checkpoint: the frame begun anew\n");
        }
        if (node.beginFrame(shape, importance, resume)) {
            const auto [back, all] = node.chunksBack();
            std::printf("MoonRay XPU2 Rhizome: the frame taken up from its checkpoint: %zu of %zu chunks done\n", back, all);
        }
        std::fflush(stdout);
        // Waited for, its checkpoint written every --checkpoint seconds meanwhile.
        for (;;) {
            if (node.waitFrame(a.checkpointSeconds > 0 ? a.checkpointSeconds : 24 * 3600.0)) break;
            if (a.checkpointSeconds <= 0) throw std::runtime_error("the frame did not complete");
            ck.partial = true, ck.frame = frameNumber, ck.manifest = manifestHash;
            ck.state = node.checkpoint();
            writeCheckpoint(ckPath, ck);
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - frameBegan).count();
        const std::vector<double> film = node.film();
        const std::vector<uint32_t> counts = node.counts();
        const std::string out = hasFrames ? rz::frameOutput(job.out, frameNumber, many) : job.out;
        {
            std::error_code e;
            const fs::path parent = fs::u8path(out).parent_path();
            if (!parent.empty()) fs::create_directories(parent, e);
        }
        writePfm(fs::u8path(out), shape.width, shape.height, film, counts);
        std::function<std::string(const std::string&)> rename;
        if (hasFrames) rename = [frameNumber, many](const std::string& n) { return frameOutputName(n, frameNumber, many); };
        l->writeOutputs(film, counts, merge.odd, merge.aovs, merge.depths, merge.crypto.get(), rename);
        const auto by = node.chunksByNode();
        std::string share;
        for (size_t i = 0; i < by.size(); ++i)
            share += (i ? ", " : "") + rz::shortHex(node.plan()->nodes[i].hash) + " " + std::to_string(by[i]) + (node.plan()->nodes[i].fast ? " (fast)" : " (slow)");
        std::printf("MoonRay XPU2 Rhizome: the frame in %.2f s, %llu films audited; chunks by node: %s\n", seconds,
                    (unsigned long long)node.audits(), share.c_str());
        if (const std::string audit = node.auditSummary(); !audit.empty()) std::printf("MoonRay XPU2 Rhizome: audits: %s\n", audit.c_str());
        if (shape.adaptive())
            std::printf("MoonRay XPU2 Rhizome: adaptive: %zu of %u cells converged early, %zu chunks left unrendered\n", node.convergedCells(),
                        shape.cells(), node.cancelledChunks());
        std::printf("MoonRay XPU2 Rhizome: wrote %s\n", out.c_str());
        std::fflush(stdout);
        ck.done.push_back(frameNumber);
        ck.partial = false, ck.state.clear();
        if (a.checkpointSeconds > 0) writeCheckpoint(ckPath, ck);
    };
    // A job: each of its frames. A job's scene is placed in the pool anew (its files may have been written again).
    auto renderJob = [&](const rz::Job& job) {
        job.check();
        placed.clear();
        heldScenes.clear();
        heldDeltas.clear();
        const std::vector<int> frames = job.frameList();
        const bool hasFrames = !frames.empty();
        const auto began = Clock::now();
        const size_t total = hasFrames ? frames.size() : 1;
        // Its checkpoint: the XPU2 build and the job, hashed, name it; one there (every machine stopped before) is
        // taken up, unless --fresh.
        rz::Bytes key(o.build.begin(), o.build.end());
        const std::string text = rz::writeJob(job);
        key.insert(key.end(), text.begin(), text.end());
        JobCheckpoint ck;
        ck.job = rz::sha256(key);
        ck.files = jobFiles(job);
        const fs::path ckPath = folder / "checkpoints" / (rz::hex(ck.job).substr(0, 16) + ".rzj");
        {
            JobCheckpoint was;
            if (!a.fresh && readCheckpoint(ckPath, was) && was.job == ck.job) {
                if (was.files != ck.files) {
                    std::printf("MoonRay XPU2 Rhizome: job %s: its files changed since its checkpoint: begun anew\n", job.name.c_str());
                } else if (was.finished) {
                    std::printf("MoonRay XPU2 Rhizome: job %s was finished by an earlier run: skipped (--fresh renders it again)\n",
                                job.name.c_str());
                    std::fflush(stdout);
                    return std::string("finished by an earlier run");
                } else {
                    ck = was;
                    std::printf("MoonRay XPU2 Rhizome: job %s taken up from its checkpoint: %zu of %zu frames done%s\n", job.name.c_str(),
                                ck.done.size(), total, ck.partial ? (", frame " + std::to_string(ck.frame) + " under way").c_str() : "");
                }
                std::fflush(stdout);
            }
        }
        for (size_t i = 0; i < total; ++i) {
            const int f = hasFrames ? frames[i] : 0;
            const bool skip = std::find(ck.done.begin(), ck.done.end(), f) != ck.done.end();
            renderFrame(job, hasFrames, f, i, total, ck, ckPath, skip);
        }
        // The job done: its checkpoint says so (the same job again is skipped while its files are as they were).
        ck.finished = true, ck.partial = false, ck.state.clear();
        if (a.checkpointSeconds > 0) writeCheckpoint(ckPath, ck);
        const double s = std::chrono::duration<double>(Clock::now() - began).count();
        std::printf("MoonRay XPU2 Rhizome: job %s: %zu frame%s in %.1f s\n", job.name.c_str(), total, total == 1 ? "" : "s", s);
        std::fflush(stdout);
        return std::to_string(total) + (total == 1 ? " frame in " : " frames in ") + std::to_string(int(s + 0.5)) + " s";
    };
    for (const auto& job : given) renderJob(job);
    if (!queueFolder.empty()) {
        std::printf("MoonRay XPU2 Rhizome: the queue %s%s\n", queueFolder.u8string().c_str(), a.queueWatch ? ", watched for new jobs" : "");
        // Jobs left running (the master stopped in them): back in the queue, in their order, their checkpoints kept.
        for (const fs::path& was : rz::queue::waiting(queueFolder / "running")) {
            try {
                rz::queue::move(was, queueFolder);
                std::printf("MoonRay XPU2 Rhizome: job %s was under way when the master stopped: back in the queue\n",
                            was.stem().u8string().c_str());
            } catch (const std::exception& e) {
                std::printf("MoonRay XPU2 Rhizome: %s\n", e.what());
            }
        }
        std::fflush(stdout);
        for (;;) {
            rz::Job job;
            fs::path running;
            try {
                running = rz::queue::take(queueFolder, job);
            } catch (const std::exception& e) {
                std::printf("MoonRay XPU2 Rhizome: a job in the queue does not read: %s\n", e.what());
                std::fflush(stdout);
                continue;
            }
            if (running.empty()) {
                if (!a.queueWatch) break;
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            std::printf("MoonRay XPU2 Rhizome: job %s from the queue\n", job.name.c_str());
            std::fflush(stdout);
            try {
                rz::queue::finish(running, queueFolder, true, renderJob(job));
            } catch (const std::exception& e) {
                std::printf("MoonRay XPU2 Rhizome: job %s failed: %s\n", job.name.c_str(), e.what());
                std::fflush(stdout);
                rz::queue::finish(running, queueFolder, false, e.what());
            }
        }
    }
    prober.reset();
    node.stop();
    if (finisher.joinable()) finisher.join();
    return 0;
}

}  // namespace rhizomesession
}  // namespace
