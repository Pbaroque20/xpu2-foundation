// Rhizome's checks (RHIZOME.md, 6): identities, messages, members, the plan, ledger and holder, the memory pool, the audit,
// and swarms of nodes over TCP on this machine with a stand-in renderer (a sample's value is a hash of its identity,
// as an XPU2 sample is a function of its identity): every pixel gets exactly its samples whatever is lost, stolen or
// refused, the important cells are the fast nodes', a node that fails has its work and its memory taken over, and one
// that cheats is caught. Prints a line a check and a JSON summary; exits non-zero on a failure.
#include "rhizome_node.h"
#include "rhizome_discovery.h"
#include <future>
#include <cstdio>
#include <filesystem>

using namespace xpu2::interactive::rhizome;

static int failures = 0, checks = 0;
static void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// The stand-in renderer: a sample's value from its pixel, index and channel.
static float sampleValue(uint32_t x, uint32_t y, uint32_t s, int c) {
    uint64_t h = (uint64_t(x) * 0x9e3779b97f4a7c15ull) ^ (uint64_t(y) * 0xc2b2ae3d27d4eb4full) ^ (uint64_t(s) * 0x165667b19e3779f9ull) ^ uint64_t(c + 1);
    h ^= h >> 33, h *= 0xff51afd7ed558ccdull, h ^= h >> 33, h *= 0xc4ceb9fe1a85ec53ull, h ^= h >> 33;
    return float(h >> 40) * 0x1p-24f;
}
static void pixelSum(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) {
    for (int c = 0; c < 4; ++c) sum[c] = 0;
    for (uint32_t s = first; s < first + count; ++s)
        for (int c = 0; c < 4; ++c) sum[c] += sampleValue(x, y, s, c);
}
// A node's renderer at a rate (samples a second; the stand-in sleeps to keep it), and a bias for a cheat.
static CellRenderer renderer(double rate, float bias = 0) {
    return [rate, bias](const uint32_t r[4], uint32_t first, uint32_t count) {
        Film f;
        for (int k = 0; k < 4; ++k) f.rect[k] = r[k];
        f.first = first, f.count = count;
        const size_t n = size_t(r[2] - r[0]) * (r[3] - r[1]);
        f.sums.resize(n * 4);
        f.counts.assign(n, count);
        size_t i = 0;
        for (uint32_t y = r[1]; y < r[3]; ++y)
            for (uint32_t x = r[0]; x < r[2]; ++x, ++i) {
                pixelSum(x, y, first, count, &f.sums[i * 4]);
                f.sums[i * 4] += bias * count;
            }
        std::this_thread::sleep_for(std::chrono::duration<double>(double(n) * count / rate));
        return f;
    };
}

static void identities() {
    auto a = Identity::make(), b = Identity::make();
    check(a->hash() != b->hash(), "two nodes have two hashes");
    check(keyMatches(a->publicKey(), a->hash()), "a node's hash is its public key's SHA-256");
    const char text[] = "a contribution";
    const Bytes sig = a->sign(text, sizeof text);
    check(verify(a->publicKey(), text, sizeof text, sig), "a signature verifies with its key");
    check(!verify(b->publicKey(), text, sizeof text, sig), "and not with another's");
    char other[] = "a contribution";
    other[0] = 'A';
    check(!verify(a->publicKey(), other, sizeof other, sig), "and not over other bytes");
    const auto folder = std::filesystem::temp_directory_path() / ("rhizome-test-" + shortHex(a->hash()));
    std::filesystem::remove_all(folder);
    const Hash first = Identity::load(folder)->hash();
    check(Identity::load(folder)->hash() == first, "a node keeps its hash across restarts (node.key)");
    std::filesystem::remove_all(folder);
    check(Identity::load(folder)->hash() != first, "a new folder, a new hash");
    std::filesystem::remove_all(folder);
}

static void messages() {
    auto a = Identity::make(), b = Identity::make();
    Writer w;
    w.u32(7), w.str("cell"), w.f64(1.5);
    const Bytes frame = encode(*a, Type::Contribution, 3, w.b);
    auto m = peek(frame);
    check(m && m->type == Type::Contribution && m->sequence == 3 && m->sender == a->hash(), "a message reads back");
    check(m && authentic(frame, *m, a->publicKey()), "and is authentic with its sender's key");
    check(m && !authentic(frame, *m, b->publicKey()), "not with another's key");
    Bytes tampered = frame;
    tampered[HeaderBytes + 1] ^= 1;
    auto t = peek(tampered);
    check(t && !authentic(tampered, *t, a->publicKey()), "a changed byte breaks the signature");
    Reader r(m->payload);
    check(r.u32() == 7 && r.str() == "cell" && r.f64() == 1.5 && r.done(), "its payload reads back");

    Members ms;
    ms.suspectAfter = 1, ms.goneAfter = 2;
    check(ms.introduce(a->hash(), a->publicKey(), {}, 0.0), "a member introduced");
    check(!ms.introduce(b->hash(), a->publicKey(), {}, 0.0), "a key that is not the hash's is refused");
    check(ms.accept(frame, *m, 0.5), "a member's message taken");
    check(!ms.accept(frame, *m, 0.6), "the same message again refused (its sequence)");
    ms.sweep(1.8);
    check(ms.find(a->hash())->state == State::Suspect, "silent past suspectAfter: suspected");
    check(ms.sweep(2.7).size() == 1 && ms.find(a->hash())->state == State::Gone, "past goneAfter: gone");
    ms.quarantine(a->hash());
    check(!ms.introduce(a->hash(), a->publicKey(), {}, 3.0), "a quarantined hash cannot come back");
}

static Hash fakeHash(int i) {
    Hash h{};
    h[0] = uint8_t(i + 1);
    return h;
}

static void plans() {
    FrameShape s;
    s.width = 1024, s.height = 512, s.cellSize = 64, s.spp = 16, s.chunkSamples = 4;
    std::vector<float> imp(s.cells(), 0);
    for (U32 c = 0; c < s.cells(); ++c) {
        const int cx = int(c % s.cellsX()), cy = int(c / s.cellsX());
        imp[c] = (cx >= 6 && cx < 10 && cy >= 2 && cy < 6) ? 10.0f : 1.0f;  // a feature in the middle
    }
    std::vector<Node> nodes = {{fakeHash(0), 1e7, 0.001}, {fakeHash(1), 9e6, 0.001}, {fakeHash(2), 2e6, 0.002}, {fakeHash(3), 1e6, 0.05}};
    const Plan p = Plan::make(s, nodes, imp);
    bool all = true, impFast = true;
    std::vector<double> pixels(4, 0);
    for (U32 c = 0; c < s.cells(); ++c) {
        all = all && p.owner[c] >= 0;
        if (p.important[c]) impFast = impFast && p.nodes[size_t(p.owner[c])].fast;
        if (p.owner[c] >= 0) pixels[size_t(p.owner[c])] += s.pixels(c);
    }
    check(all, "every cell has an owner");
    check(p.nodes[0].fast && p.nodes[1].fast && !p.nodes[2].fast && !p.nodes[3].fast, "tiers by rate: two fast, two slow");
    check(impFast, "the important cells are the fast tier's");
    check(pixels[0] > pixels[2] && pixels[2] > pixels[3], "shares follow the rates");
    // A node's cells together: few runs of neighbours (components) for each node.
    bool together = true;
    for (int n = 0; n < 4; ++n) {
        std::vector<U32> mine;
        for (U32 c = 0; c < s.cells(); ++c)
            if (p.owner[c] == n) mine.push_back(c);
        std::vector<int> comp(mine.size(), -1);
        int comps = 0;
        for (size_t i = 0; i < mine.size(); ++i) {
            if (comp[i] >= 0) continue;
            std::vector<size_t> stack = {i};
            comp[i] = comps;
            while (!stack.empty()) {
                const size_t k = stack.back();
                stack.pop_back();
                for (size_t j = 0; j < mine.size(); ++j)
                    if (comp[j] < 0 && s.adjacent(mine[k], mine[j])) comp[j] = comps, stack.push_back(j);
            }
            ++comps;
        }
        together = together && comps <= 4;
    }
    check(together, "each node's cells lie in a few runs of neighbours");
    {
        // Two nodes of one rate, one far: the far one takes more chunks a grant, so a round trip stays small beside them.
        std::vector<Node> two = {{fakeHash(0), 4e6, 0.001}, {fakeHash(1), 4e6, 0.08}};
        FrameShape deep = s;
        deep.spp = 1024;  // room for a grant of many chunks
        const Plan q = Plan::make(deep, two, {});
        check(q.grantChunks[1] > q.grantChunks[0], "the far node takes more chunks a grant (" + std::to_string(q.grantChunks[1]) + " to " +
                                                       std::to_string(q.grantChunks[0]) + ")");
    }

    // Holders: a slow node cannot take a fast node's important cells.
    Holder fast(p, 0);
    std::vector<Chunk> got = fast.grant(3);
    bool noImportant = true;
    for (const auto& c : got) noImportant = noImportant && !p.important[c.cell];
    check(noImportant, "a slow thief is not given a fast node's important cells");

    // The ledger: steals, a loss, chunks twice; each chunk is taken once.
    Ledger l(p);
    const size_t all0 = l.outstanding();
    check(all0 == size_t(s.cells()) * s.chunksPerCell(), "the ledger holds every chunk");
    Holder h1(p, 1), h2(p, 2);
    const auto stolen = h1.grant(2);
    l.granted(stolen, 2);
    h2.receive(stolen);
    while (auto c = h2.next()) l.arrived(*c);
    check(!l.arrived(Chunk{stolen.empty() ? 0 : stolen[0].cell, stolen.empty() ? 0 : stolen[0].index}), "a chunk's second film is refused");
    std::vector<bool> alive = {true, false, true, true};
    const auto heirs = l.lost(1, alive);
    size_t reassigned = 0;
    bool heirsAlive = true;
    for (const auto& [to, v] : heirs) {
        reassigned += v.size();
        heirsAlive = heirsAlive && alive[size_t(to)];
        for (const auto& c : v) heirsAlive = heirsAlive && (!p.important[c.cell] || p.nodes[size_t(to)].fast);
    }
    check(reassigned > 0 && heirsAlive, "a lost node's chunks go to working nodes, important ones to the fast tier");
    for (U32 c = 0; c < s.cells(); ++c)
        for (U32 k = 0; k < s.chunksPerCell(); ++k) l.arrived(Chunk{c, k});
    check(l.complete(), "the ledger completes");
}

static void memory() {
    Pool pool;
    for (int i = 0; i < 5; ++i) pool.addNode(fakeHash(i), (i == 4 ? 1 : 4) * (256ull << 20), i < 2 ? (128ull << 20) : 0);
    std::vector<Piece> pieces;
    for (int k = 0; k < 200; ++k) {
        Piece p;
        p.hash = sha256(&k, sizeof k);
        p.bytes = (1 + k % 4) << 20;
        p.prefers = k % 3 == 0 ? Tier::Vram : Tier::Ram;
        pieces.push_back(p);
        pool.add(p);
    }
    bool two = true, distinct = true, fits = true, vram = false;
    for (const auto& p : pieces) {
        const auto& hs = pool.holders(p.hash);
        two = two && hs.size() == 2;
        distinct = distinct && (hs.size() < 2 || hs[0].node != hs[1].node);
        for (const auto& c : hs) vram = vram || c.tier == Tier::Vram;
    }
    for (int i = 0; i < 5; ++i)
        for (Tier t : {Tier::Ram, Tier::Vram}) fits = fits && pool.used(fakeHash(i), t) <= pool.capacity(fakeHash(i), t);
    check(two && distinct, "every piece is kept on two nodes");
    check(fits, "no node holds more than it gives");
    check(vram, "pieces a card reads go to card memory where there is room");
    // Machines fail one after another: each failure's pieces are copied again from a survivor.
    bool moved = true;
    for (int dead : {0, 3}) {
        const auto moves = pool.fail(fakeHash(dead));
        for (const auto& mv : moves) moved = moved && mv.from != fakeHash(dead) && mv.to.node != fakeHash(dead) && mv.from != Hash{};
        for (const auto& p : pieces)
            for (const auto& c : pool.holders(p.hash)) moved = moved && c.node != fakeHash(dead);
    }
    bool kept = true;
    for (const auto& p : pieces) kept = kept && !pool.lostAll(p.hash) && pool.holders(p.hash).size() == 2;
    check(moved, "a failed node's copies are made again from a node that holds them");
    check(kept, "after two failures every piece still has two copies");
    {
        // Short of room: a piece keeps what copies fit, and gets its second when a node joins.
        Pool small;
        small.addNode(fakeHash(0), 8ull << 20, 0);
        small.addNode(fakeHash(1), 1ull << 20, 0);
        Piece big{sha256("big", 3), 4ull << 20, Tier::Ram};
        small.add(big);
        const bool shortOne = small.holders(big.hash).size() == 1;
        small.addNode(fakeHash(2), 8ull << 20, 0);
        const auto moves = small.joined();
        check(shortOne && moves.size() == 1 && moves[0].from == fakeHash(0) && small.holders(big.hash).size() == 2,
              "short of room a piece waits for a node with room, then is copied there");
    }
}

static void audits() {
    Auditor a;
    a.rate = 1.0 / 64;
    const uint32_t r[4] = {0, 0, 64, 64};
    const Film honest = renderer(1e12)(r, 4, 4);
    check(a.check(fakeHash(0), honest, pixelSum).passed, "an honest film passes");
    // Device noise: each pixel a few tenths of a percent off either way, as a card's film is beside a processor's.
    Film noisy = honest;
    for (size_t i = 0; i < noisy.counts.size(); ++i) {
        const float k = 1.0f + 0.004f * ((i * 2654435761u >> 7) % 2 ? 1.0f : -1.0f) * float((i * 40503u) % 100) / 100.0f;
        for (int c = 0; c < 4; ++c) noisy.sums[i * 4 + c] *= k;
    }
    Auditor every;
    every.rate = 1.0;
    check(every.check(fakeHash(3), noisy, pixelSum).passed, "a film with device noise (up to 0.4% a pixel, either way) passes");
    const Film biased = renderer(1e12, 0.01f)(r, 4, 4);
    check(!a.check(fakeHash(1), biased, pixelSum).passed, "a film one percent off fails");
    Film brighter = honest;
    for (auto& s : brighter.sums) s *= 1.003f;
    Auditor b2;
    b2.rate = 1.0 / 16;
    const Verdict vb = b2.check(fakeHash(4), brighter, pixelSum);
    check(!vb.passed, "a film 0.3% brighter everywhere fails (bias " + std::to_string(vb.bias * 100) + "%)");
    Film some = honest;
    for (size_t i = 0; i < some.counts.size(); i += 10) some.sums[i * 4] += 1.0f;  // a tenth of the pixels wrong
    Auditor b;
    b.rate = 1.0;  // every pixel
    check(!b.check(fakeHash(2), some, pixelSum).passed, "a film with a tenth of its pixels wrong fails");
    Film short_ = honest;
    short_.counts[7] -= 1;
    check(!every.check(fakeHash(5), short_, pixelSum).passed, "a pixel short of its samples fails");
}

struct Swarm {
    std::unique_ptr<RhizomeNode> master;
    std::vector<std::unique_ptr<RhizomeNode>> workers;
};
static NodeOptions options(const std::string& key, bool master, double rate, float bias = 0) {
    NodeOptions o;
    o.rhizomeKey = key;
    o.master = master;
    o.caps.rate = rate;
    o.caps.cpuThreads = 8;
    o.ramBytes = 256ull << 20, o.vramBytes = 64ull << 20;
    o.render = renderer(rate, bias);
    o.renderPixel = pixelSum;
    if (std::getenv("RHIZOME_TEST_LOG"))
        o.log = [](const std::string& s) {
            static const double began = nowSeconds();
            std::printf("    %8.3f %s\n", nowSeconds() - began, s.c_str());
        };
    return o;
}
static Swarm swarm(const std::vector<std::pair<double, float>>& workers, double masterRate, const std::string& key = "studio rhizome key") {
    Swarm s;
    s.master = std::make_unique<RhizomeNode>(options(key, true, masterRate));
    s.master->start();
    for (const auto& [rate, bias] : workers) {
        NodeOptions o = options(key, false, rate, bias);
        o.masterAddress = s.master->address();
        s.workers.push_back(std::make_unique<RhizomeNode>(o));
        s.workers.back()->start();
    }
    for (int k = 0; k < 200 && s.master->joined() < workers.size(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));  // the workers meet each other
    return s;
}
static FrameShape shape() {
    FrameShape s;
    s.width = 512, s.height = 256, s.cellSize = 64, s.spp = 16, s.chunkSamples = 4;
    return s;
}
static std::vector<float> importance(const FrameShape& s) {
    std::vector<float> imp(s.cells(), 1.0f);
    for (U32 c = 0; c < s.cells(); ++c) {
        const U32 cx = c % s.cellsX(), cy = c / s.cellsX();
        if (cx >= 3 && cx < 5 && cy >= 1 && cy < 3) imp[c] = 8.0f;
    }
    return imp;
}
// Whether the master's picture is every pixel's spp samples, each once, as the stand-in renders them.
static bool exact(RhizomeNode& m, const FrameShape& s, double* worst = nullptr) {
    const auto film = m.film();
    const auto counts = m.counts();
    double w = 0;
    bool ok = true;
    for (U32 y = 0; y < s.height; ++y)
        for (U32 x = 0; x < s.width; ++x) {
            const size_t i = size_t(y) * s.width + x;
            if (counts[i] != s.spp) ok = false;
            float ref[4];
            pixelSum(x, y, 0, s.spp, ref);
            for (int c = 0; c < 4; ++c) w = std::max(w, std::fabs(film[i * 4 + c] - ref[c]) / std::max(1e-6, double(ref[c])));
        }
    if (worst) *worst = w;
    return ok && w < 1e-4;
}

static void swarms() {
    const FrameShape s = shape();
    {
        // Two fast workers, two slow, the master rendering too.
        Swarm sw = swarm({{4e6, 0}, {4e6, 0}, {6e5, 0}, {4e5, 0}}, 4e6);
        check(sw.master->joined() == 4, "four workers join the master");
        const Hash a = sw.workers[0]->hash(), b = sw.workers[1]->hash();
        check(a != b && a != sw.master->hash(), "each node has its own hash");
        // The memory pool across the swarm.
        std::vector<Piece> pieces;
        for (int k = 0; k < 24; ++k) {
            Bytes data(1 << 16, uint8_t(k));
            Piece p{sha256(data), data.size(), k % 2 ? Tier::Vram : Tier::Ram};
            pieces.push_back(p);
            sw.master->addPiece(p, data);
        }
        sw.master->beginFrame(s, importance(s));
        const bool done = sw.master->waitFrame(60);
        double worst = 0;
        check(done, "the frame completes");
        check(done && exact(*sw.master, s, &worst), "every pixel has its 16 samples once, as rendered (worst " + std::to_string(worst) + ")");
        const auto [fastImp, slowImp] = sw.master->importantChunks();
        check(fastImp > 0 && slowImp == 0, "the important cells were rendered by the fast nodes (" + std::to_string(fastImp) + " chunks, slow " +
                                               std::to_string(slowImp) + ")");
        const auto by = sw.master->chunksByNode();
        size_t contributing = 0;
        for (auto n : by) contributing += n > 0;
        check(contributing == by.size(), "every node contributed");
        check(sw.master->audits() > 0, "the master audited the workers' films (" + std::to_string(sw.master->audits()) + ")");
        // Every piece reached the nodes the pool placed it on.
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        bool placed = true;
        for (const auto& p : pieces)
            for (const auto& c : sw.master->holdersOf(p.hash)) {
                RhizomeNode* n = c.node == sw.master->hash() ? sw.master.get() : nullptr;
                for (auto& w : sw.workers)
                    if (w->hash() == c.node) n = w.get();
                placed = placed && n && n->holds(p.hash);
            }
        check(placed, "each piece is held where the pool placed it");

        // A worker fails: its memory is taken over.
        const Hash dead = sw.workers[2]->hash();
        sw.workers[2]->stop();
        // Gone after two seconds of silence, then its pieces copied again: looked at until they are, ten seconds at most.
        bool covered = false;
        for (int k = 0; k < 100 && !covered; ++k) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            covered = true;
            for (const auto& p : pieces) {
                const auto hs = sw.master->holdersOf(p.hash);
                covered = covered && hs.size() == 2;
                for (const auto& c : hs) {
                    covered = covered && c.node != dead;
                    RhizomeNode* n = c.node == sw.master->hash() ? sw.master.get() : nullptr;
                    for (auto& w : sw.workers)
                        if (w->hash() == c.node && w->hash() != dead) n = w.get();
                    covered = covered && n && n->holds(p.hash);
                }
            }
        }
        check(covered, "a failed node's pieces are copied to the others: two copies each again");
    }
    {
        // A worker fails mid-frame: its chunks go to the others and the picture is whole.
        Swarm sw = swarm({{2e6, 0}, {2e6, 0}, {2e6, 0}}, 2e6);
        sw.master->beginFrame(s, importance(s));
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        sw.workers[1]->stop();
        const bool done = sw.master->waitFrame(60);
        double worst = 0;
        check(done && exact(*sw.master, s, &worst), "a node lost mid-frame: the frame completes, every sample once");
    }
    {
        // A worker that cheats: caught by the audit, quarantined, its work done again.
        Swarm sw = swarm({{2e6, 0}, {2e6, 0.05f}, {2e6, 0}}, 2e6);
        const Hash cheat = sw.workers[1]->hash();
        sw.master->beginFrame(s, importance(s));
        const bool done = sw.master->waitFrame(60);
        check(sw.master->quarantined(cheat), "a node sending a wrong film is quarantined");
        check(done && exact(*sw.master, s), "and the picture is the honest one");
    }
    {
        // A node without the rhizome key is not let in.
        Swarm sw = swarm({{2e6, 0}}, 2e6);
        NodeOptions o = options("another key", false, 2e6);
        o.masterAddress = sw.master->address();
        RhizomeNode stranger(o);
        stranger.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        check(sw.master->joined() == 1, "a node without the rhizome key does not join");
        stranger.stop();
    }
}

// The scene through the swarm as a session moves it: the master's pieces and manifest, each worker taking the manifest
// as it joins and fetching every piece while the others do, then saying it is ready; a frame then only on ready nodes.
static void sceneThroughSwarm(bool joinFirst) {
    const std::string key = "studio rhizome key";
    NodeOptions mo = options(key, true, 2e6);
    mo.readyAtJoin = false;
    RhizomeNode master(mo);
    master.start();
    std::vector<Hash> pieces;
    auto publish = [&] {
        for (int k = 0; k < 32; ++k) {
            Bytes data(4u << 20, uint8_t(k * 7 + 1));
            data[0] = uint8_t(k);
            Piece p{sha256(data), data.size(), Tier::Ram};
            master.addPiece(p, data);
            pieces.push_back(p.hash);
        }
        Writer mw;
        mw.u32(uint32_t(pieces.size()));
        for (const auto& h : pieces) mw.hash(h);
        master.publishManifest(mw.b);
    };
    // The master's pieces before anyone joins, or once the workers have (as a session reading its scene while they come).
    if (!joinFirst) publish();
    struct Worker {
        std::unique_ptr<RhizomeNode> node;
        std::promise<Bytes> manifest;
        std::atomic<bool> got{false};
        bool fetched = false;
    };
    std::vector<std::unique_ptr<Worker>> workers;
    for (int i = 0; i < 2; ++i) {
        auto w = std::make_unique<Worker>();
        NodeOptions o = options(key, false, 2e6);
        o.readyAtJoin = false;
        o.masterAddress = master.address();
        Worker* wp = w.get();
        o.onManifest = [wp](const Bytes& m) {
            if (!wp->got.exchange(true)) wp->manifest.set_value(m);
        };
        w->node = std::make_unique<RhizomeNode>(o);
        w->node->start();
        workers.push_back(std::move(w));
    }
    if (joinFirst) {
        for (int k = 0; k < 200 && master.joined() < 2; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        publish();
    }
    const std::string when = joinFirst ? " (joined before the scene was published)" : "";
    std::vector<std::thread> threads;
    for (auto& w : workers)
        threads.emplace_back([wp = w.get()] {
            auto f = wp->manifest.get_future();
            const double began = nowSeconds();
            if (f.wait_for(std::chrono::seconds(20)) != std::future_status::ready) {
                std::printf("    worker %s: no manifest in 20 s\n", shortHex(wp->node->hash()).c_str());
                return;
            }
            const Bytes manifest = f.get();  // kept: the reader points into it
            Reader r(manifest);
            std::vector<Hash> want(r.u32());
            for (auto& h : want) h = r.hash();
            wp->fetched = wp->node->fetch(want, 60);
            if (!wp->fetched || std::getenv("RHIZOME_TEST_LOG"))
                std::printf("    worker %s: %zu pieces %s in %.2f s\n", shortHex(wp->node->hash()).c_str(), want.size(),
                            wp->fetched ? "fetched" : "NOT fetched", nowSeconds() - began);
            if (wp->fetched) wp->node->markReady();
        });
    for (auto& t : threads) t.join();
    bool got = true, all = true, whole = true;
    for (auto& w : workers) {
        got = got && w->got;
        all = all && w->fetched;
        for (const auto& h : pieces) whole = whole && sha256(w->node->piece(h)) == h;
    }
    check(got, "each worker takes the manifest" + when);
    check(all, "each worker fetches the scene's pieces from the swarm" + when);
    check(whole, "every piece arrives whole on each worker" + when);
    for (int k = 0; k < 500 && master.ready() < 2; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    check(master.ready() == 2, "the workers say they are ready" + when);
    const FrameShape s = shape();
    master.beginFrame(s, {});
    const bool done = master.waitFrame(60);
    check(done && exact(master, s), "a frame on the ready workers completes, every sample once" + when);
    for (auto& w : workers) w->node->stop();
    master.stop();
}

// Signed approvals in the swarm (rhizome_trust.h): a master that requires them holds a node until an approver
// endorses it, a node keeps its endorsement and pins its master, trust-first endorses at once, a revocation
// refuses a node everywhere, a build that differs is refused, and a master reaches a waiting node itself.
static bool until(const std::function<bool()>& f, double seconds = 5) {
    for (int k = 0; k < int(seconds * 50); ++k) {
        if (f()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return f();
}
static void approvals() {
    const std::string key = "studio rhizome key";
    const auto root = std::filesystem::temp_directory_path() / "rhizome-approvals-test";
    std::filesystem::remove_all(root);
    auto masterId = Identity::load(root / "master");
    auto master = std::make_shared<Trust>(root / "master", *masterId);
    master->require(true);
    master->trustSelf();
    NodeOptions mo = options(key, true, 2e6);
    mo.trust = master;
    mo.build = "build-a";
    RhizomeNode m(mo, std::move(masterId));
    m.start();

    auto nodeId = Identity::load(root / "node");
    auto nodeTrust = std::make_shared<Trust>(root / "node", *nodeId);
    NodeOptions no = options(key, false, 2e6);
    no.trust = nodeTrust;
    no.build = "build-a";
    no.masterAddress = m.address();
    const Hash nodeHash = nodeId->hash();
    RhizomeNode n(no, std::move(nodeId));
    n.start();
    check(until([&] { auto w = m.waitingForApproval(); return w.size() == 1 && w[0] == nodeHash; }), "a node not yet approved waits for approval");
    check(!m.isMember(nodeHash), "and is not a member meanwhile");
    // The approve command: an endorsement written into the master's folder by its key.
    {
        auto id = Identity::load(root / "master");
        Trust t(root / "master", *id);
        t.endorse(nodeHash);
    }
    check(until([&] { return m.isMember(nodeHash); }), "approved, it joins within a second or two");
    check(until([&] { return Trust(root / "node", *Identity::load(root / "node")).shown().size() == 1; }), "the node keeps its endorsement");
    check(until([&] { return Trust(root / "node", *Identity::load(root / "node")).trusted(m.hash()); }), "and pins its master as its approver");

    // A node of another build is refused.
    {
        NodeOptions bo = options(key, false, 2e6);
        bo.build = "build-b";
        bo.masterAddress = m.address();
        RhizomeNode other(bo);
        other.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        check(!m.isMember(other.hash()) && m.waitingForApproval().size() == 0, "a node of another build is refused");
        other.stop();
    }
    // Trust first: a master that endorses each machine it meets on this run.
    {
        auto fid = Identity::load(root / "first");
        auto ft = std::make_shared<Trust>(root / "first", *fid);
        ft->require(true);
        ft->trustSelf();
        NodeOptions fo = options(key, true, 2e6);
        fo.trust = ft;
        fo.trustFirst = true;
        RhizomeNode fm(fo, std::move(fid));
        fm.start();
        NodeOptions wo = options(key, false, 2e6);
        wo.masterAddress = fm.address();
        RhizomeNode w(wo);
        w.start();
        check(until([&] { return fm.isMember(w.hash()); }), "trust first: a machine is endorsed and joins at once");
        // A master reaches a node itself (as discovery has it do): a node that waits, with no master given.
        NodeOptions ho = options(key, false, 2e6);
        RhizomeNode headless(ho);
        headless.start();
        fm.dial(headless.address());
        check(until([&] { return fm.isMember(headless.hash()) && headless.isMember(fm.hash()); }), "a master reaches a waiting node and both are members");
        std::printf("    (stopping trust-first's machines)\n");
        std::fflush(stdout);
        headless.stop();
        std::printf("    (headless stopped)\n");
        std::fflush(stdout);
        w.stop();
        std::printf("    (worker stopped)\n");
        std::fflush(stdout);
        fm.stop();
        std::printf("    (stopped)\n");
        std::fflush(stdout);
    }
    // A revocation by the master's key: the node is refused from then on.
    {
        auto id = Identity::load(root / "master");
        Trust t(root / "master", *id);
        t.revoke(nodeHash);
    }
    check(until([&] { return !m.isMember(nodeHash); }), "a revoked node is refused");
    std::printf("    (stopping the node)\n");
    std::fflush(stdout);
    n.stop();
    std::printf("    (stopping the master)\n");
    std::fflush(stdout);
    m.stop();
    std::filesystem::remove_all(root);
}

// Discovery (rhizome_discovery.h): a master's probes on every adapter and loopback, a node's answer straight back with
// its hash and TCP port and a keyed check; a member of another rhizome is not found.
static void discovery() {
    Winsock w;
    const uint16_t port = 47911;  // a test port of its own
    auto a = Identity::make(), b = Identity::make();
    const auto adapters = broadcastAddresses();
    check(!adapters.empty(), "the broadcast address of each adapter is known (" + std::to_string(adapters.size()) + ")");
    Responder ra("studio rhizome key", a->hash(), 47123, port), rb("another studio's key", b->hash(), 47124, port);
    std::mutex m;
    std::map<Hash, std::string> found;
    Prober p("studio rhizome key", [&](const Hash& h, const std::string& at) {
        std::lock_guard<std::mutex> g(m);
        found[h] = at;
    }, port, 0.2);
    const bool got = until([&] {
        std::lock_guard<std::mutex> g(m);
        return found.count(a->hash()) != 0;
    });
    std::lock_guard<std::mutex> g(m);
    check(got, "a master finds a node of its rhizome by discovery");
    check(got && found[a->hash()].size() > 6 && found[a->hash()].substr(found[a->hash()].size() - 6) == ":47123", "at its address and TCP port (" +
          (got ? found[a->hash()] : std::string("none")) + ")");
    check(!found.count(b->hash()), "a node of another rhizome is not found");
}

int main() {
    // RHIZOME_TEST_ONLY=<section>: that section alone (identities, messages, plans, memory, audits, swarms, scene, approvals).
    const char* only = std::getenv("RHIZOME_TEST_ONLY");
    auto run = [&](const char* name, const std::function<void()>& f) {
        if (only && std::string(only) != name) return;
        f();
        std::fflush(stdout);
    };
    run("identities", identities);
    run("messages", messages);
    run("plans", plans);
    run("memory", memory);
    run("audits", audits);
    run("swarms", swarms);
    run("scene", [] {
        sceneThroughSwarm(false);
        sceneThroughSwarm(true);
    });
    run("approvals", approvals);
    run("discovery", discovery);
    std::printf("{\"checks\": %d, \"failures\": %d}\n", checks, failures);
    return failures ? 1 : 0;
}
