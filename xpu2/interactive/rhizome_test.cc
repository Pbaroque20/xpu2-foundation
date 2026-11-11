// Rhizome's checks (RHIZOME.md, 6): identities, messages, members, the plan, ledger and holder, the memory pool, the audit,
// and swarms of nodes over TCP on this machine with a stand-in renderer (a sample's value is a hash of its identity,
// as an XPU2 sample is a function of its identity): every pixel gets exactly its samples whatever is lost, stolen or
// refused, the important cells are the fast nodes', a node that fails has its work and its memory taken over, and one
// that cheats is caught. Prints a line a check and a JSON summary; exits non-zero on a failure.
#include "rhizome_node.h"
#include "rhizome_discovery.h"
#include "rhizome_importance.h"
#include "rhizome_jobs.h"
#include <future>
#include <tuple>
#include <cstdio>
#include <filesystem>

using namespace xpu2::interactive::rhizome;

static int failures = 0, checks = 0;
static void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// The stand-in renderer: a sample's value from its pixel, index and channel. noisyRight (the adaptive checks): the
// picture's left half nearly flat (it converges at once), its right half as noisy as before (it never does).
static bool noisyRight = false;
static float sampleValue(uint32_t x, uint32_t y, uint32_t s, int c) {
    uint64_t h = (uint64_t(x) * 0x9e3779b97f4a7c15ull) ^ (uint64_t(y) * 0xc2b2ae3d27d4eb4full) ^ (uint64_t(s) * 0x165667b19e3779f9ull) ^ uint64_t(c + 1);
    h ^= h >> 33, h *= 0xff51afd7ed558ccdull, h ^= h >> 33, h *= 0xc4ceb9fe1a85ec53ull, h ^= h >> 33;
    const float u = float(h >> 40) * 0x1p-24f;
    if (noisyRight && x < 256) return 0.5f + 0.002f * (u - 0.5f);
    return u;
}
static void pixelSum(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) {
    for (int c = 0; c < 4; ++c) sum[c] = 0;
    for (uint32_t s = first; s < first + count; ++s)
        for (int c = 0; c < 4; ++c) sum[c] += sampleValue(x, y, s, c);
}
// The frames checks: the scene's value as the frame changes it, added to every sample (0 elsewhere).
static std::atomic<float> sceneFrame{0};
// A node's renderer at a rate (samples a second; the stand-in sleeps to keep it), and a bias for a cheat.
static CellRenderer renderer(double rate, float bias = 0) {
    return [rate, bias](const uint32_t r[4], uint32_t first, uint32_t count) {
        const float frame = sceneFrame.load();
        Film f;
        for (int k = 0; k < 4; ++k) f.rect[k] = r[k];
        f.first = first, f.count = count;
        const size_t n = size_t(r[2] - r[0]) * (r[3] - r[1]);
        f.sums.resize(n * 4);
        f.odd.assign(n * 4, 0.0f);
        f.counts.assign(n, count);
        size_t i = 0;
        for (uint32_t y = r[1]; y < r[3]; ++y)
            for (uint32_t x = r[0]; x < r[2]; ++x, ++i) {
                pixelSum(x, y, first, count, &f.sums[i * 4]);
                f.sums[i * 4] += bias * count;
                f.sums[i * 4 + 1] += frame * count;
                for (uint32_t s = first | 1u; s < first + count; s += 2)
                    for (int c = 0; c < 4; ++c) f.odd[i * 4 + c] += sampleValue(x, y, s, c);
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

static void encryption() {
    auto a = Identity::make(), b = Identity::make();
    Ephemeral ea, eb;
    const auto sa = ea.agree(eb.publicKey()), sb = eb.agree(ea.publicKey());
    check(sa && sb && *sa == *sb, "two one-time keys agree on one secret");
    Ephemeral ec;
    check(sa && *ec.agree(eb.publicKey()) != *sa, "a third key does not");
    check(!ea.agree(a->publicKey()), "a signing key is not taken for a one-time key");
    Bytes cut = eb.publicKey();
    cut.pop_back();
    check(!ea.agree(cut), "nor a short one");
    const SessionKeys da = sessionKeys(*sa, "studio", true, ea.publicKey(), eb.publicKey(), a->hash(), b->hash());
    const SessionKeys db = sessionKeys(*sb, "studio", false, ea.publicKey(), eb.publicKey(), a->hash(), b->hash());
    check(da.out == db.in && da.in == db.out && da.out != da.in, "each direction its own key, both sides alike");
    const SessionKeys other = sessionKeys(*sa, "another swarm", true, ea.publicKey(), eb.publicKey(), a->hash(), b->hash());
    check(other.out != da.out, "another rhizome key, other keys");
    Seal send(da.out), take(db.in);
    const Bytes text = encode(*a, Type::Contribution, 1, Bytes(1000, 7));
    const Bytes one = send.seal(text), two = send.seal(text);
    check(one != two, "the same frame twice is sealed twice differently");
    bool hidden = true;
    for (size_t i = SealedHeaderBytes; i + 16 <= one.size() && hidden; ++i) hidden = std::memcmp(one.data() + i, text.data() + 60, 16) != 0;
    check(hidden, "a sealed frame does not show its bytes");
    auto body = [](const Bytes& r) { return Bytes(r.begin() + SealedHeaderBytes, r.end()); };
    check(Seal::length(one.data(), 1u << 20) && *Seal::length(one.data(), 1u << 20) == one.size() - SealedHeaderBytes, "its header gives its length");
    const auto got = take.open(one.data(), body(one));
    check(got && *got == text, "and it opens on the other side");
    Seal replay(db.in);
    check(replay.open(one.data(), body(one)) && !replay.open(one.data(), body(one)), "a frame played again does not open");
    Seal skip(db.in);
    check(!skip.open(two.data(), body(two)), "nor one out of order");
    Bytes bent = two;
    bent[SealedHeaderBytes + 100] ^= 1;
    check(!take.open(bent.data(), body(bent)), "nor a changed one");
    Seal wrong(da.out);
    check(!wrong.open(two.data(), body(two)), "nor with the other direction's key");
    check(!Seal::length(text.data(), 1u << 20), "a plain frame is not a sealed one");
}

// The importance of a probe lattice of 64 x 32 (every 8 px of a 512 x 256 frame, cells of 64 px: 8 x 4 cells): sky
// above, a ground plane below, a box on it (cells 2-3 across), hair in cell 6 of the second row, a glowing fog in cell 0
// of the third row.
static void importances() {
    FrameShape s;
    s.width = 512, s.height = 256, s.cellSize = 64;
    const uint32_t w = 64, h = 32;
    std::vector<ProbeHit> lattice(w * h);
    for (uint32_t j = 0; j < h; ++j)
        for (uint32_t i = 0; i < w; ++i) {
            ProbeHit& p = lattice[j * w + i];
            if (j >= 12) p.object = 0, p.depth = 10.0f + float(j);  // the ground (from y 100)
            if (j >= 12 && j < 24 && i >= 18 && i < 30) p.object = 1, p.depth = 5.0f;  // the box
            if (j >= 8 && j < 16 && i >= 48 && i < 56) p.object = 2, p.depth = 6.0f, p.families = probe::Hair;
            if (j >= 16 && j < 24 && i < 8) p.volumes = 1, p.glow = true;
        }
    const auto imp = importanceFromProbe(lattice, w, h, 8, s);
    auto cell = [&](uint32_t cx, uint32_t cy) { return imp[cy * s.cellsX() + cx]; };
    check(imp.size() == s.cells(), "a value a cell");
    check(cell(0, 0) == 0.0f && cell(5, 0) == 0.0f, "the sky alone matters least");
    check(cell(5, 3) > cell(5, 0), "the ground more than the sky");
    check(cell(2, 2) > cell(5, 3), "the box's silhouette more than the plain ground (" + std::to_string(cell(2, 2)) + " to " + std::to_string(cell(5, 3)) + ")");
    check(cell(6, 1) > cell(5, 3), "hair more than the plain ground");
    check(cell(0, 2) > cell(1, 3), "a glowing fog more than the ground beside it");
    double mean = 0;
    for (float v : imp) mean += v;
    mean /= imp.size();
    check(cell(2, 2) > mean && cell(6, 1) > mean && cell(0, 2) > mean && cell(5, 0) < mean,
          "silhouettes, hair and glow above the mean (important), the sky below");
    check(importanceFromProbe({}, w, h, 8, s) == std::vector<float>(s.cells(), 0.0f), "no probe: flat");
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

    // Plan 17: the next frame's cells back to the nodes that rendered them, though the rates moved.
    std::vector<Hash> before(s.cells());
    for (U32 c = 0; c < s.cells(); ++c) before[c] = p.nodes[size_t(p.owner[c])].hash;
    std::vector<Node> moved = {{fakeHash(0), 1.1e7, 0.001}, {fakeHash(1), 8e6, 0.001}, {fakeHash(2), 2.4e6, 0.002}, {fakeHash(3), 0.8e6, 0.05}};
    const Plan plain = Plan::make(s, moved, imp), kept = Plan::make(s, moved, imp, 0.5, 0.25, &before);
    auto same = [&](const Plan& q) {
        U32 n = 0;
        for (U32 c = 0; c < s.cells(); ++c) n += q.nodes[size_t(q.owner[c])].hash == before[c] ? 1 : 0;
        return n;
    };
    check(same(kept) > same(plain) && same(kept) >= s.cells() * 8 / 10 && kept.kept == same(kept),
          "the rates moved: " + std::to_string(same(kept)) + " of " + std::to_string(s.cells()) + " cells with the node of the frame before (" +
              std::to_string(same(plain)) + " without)");
    bool nearAll = true, nearImpFast = true;
    std::vector<double> share(4, 0), wanted(4, 0);
    double rates = 0;
    for (const auto& n : kept.nodes) rates += n.rate;
    for (U32 c = 0; c < s.cells(); ++c) {
        nearAll = nearAll && kept.owner[c] >= 0;
        if (kept.owner[c] < 0) continue;
        nearImpFast = nearImpFast && (!kept.important[c] || kept.nodes[size_t(kept.owner[c])].fast);
        share[size_t(kept.owner[c])] += s.pixels(c);
    }
    double worstShare = 0;
    for (size_t i = 0; i < 4; ++i)
        worstShare = std::max(worstShare, std::fabs(share[i] / (double(s.width) * s.height) - kept.nodes[i].rate / rates));
    check(nearAll && nearImpFast && worstShare < 0.05, "with affinity every cell is given, important ones to the fast tier, shares by rate (worst " +
                                                          std::to_string(worstShare * 100) + "% off)");
    // A node gone: its cells to the others, the rest kept.
    std::vector<Node> three(moved.begin(), moved.begin() + 3);
    const Plan less = Plan::make(s, three, imp, 0.5, 0.25, &before);
    bool lessAll = true;
    U32 keptOthers = 0, othersBefore = 0;
    for (U32 c = 0; c < s.cells(); ++c) {
        lessAll = lessAll && less.owner[c] >= 0 && less.owner[c] < 3;
        if (before[c] != fakeHash(3)) othersBefore++, keptOthers += less.nodes[size_t(less.owner[c])].hash == before[c] ? 1 : 0;
    }
    const U32 lessPlain = same(Plan::make(s, three, imp));
    check(lessAll && keptOthers >= othersBefore * 8 / 10 && keptOthers >= lessPlain,
          "a node gone: its cells given out, " + std::to_string(keptOthers) + " of the others' " + std::to_string(othersBefore) + " kept (" +
              std::to_string(lessPlain) + " without)");
    // A node joining: it takes its share, the others keep most of theirs.
    std::vector<Node> five = moved;
    five.push_back(Node{fakeHash(4), 9e6, 0.001});
    const Plan more = Plan::make(s, five, imp, 0.5, 0.25, &before);
    const U32 moreKept = same(more), morePlain = same(Plan::make(s, five, imp));
    double newShare = 0;
    for (U32 c = 0; c < s.cells(); ++c) newShare += more.owner[c] == 4 ? s.pixels(c) : 0;
    check(moreKept > morePlain && newShare > 0, "a node joining: " + std::to_string(moreKept) + " cells kept (" + std::to_string(morePlain) +
                                                    " without), the new node given " + std::to_string(int(newShare * 100 / (double(s.width) * s.height))) + "%");
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
    // A card's film beside a processor's: of a few samples, a path through a volume that turned another way leaves a
    // pixel far off now and then. One outlier among a film's few picks is not a wrong film; a node with outliers far
    // past the honest share over its films is.
    const uint32_t small[4] = {0, 0, 2, 2};
    auto withOutlier = [&](uint32_t first, bool off) {
        Film f = renderer(1e12)(small, first, 4);
        if (off)
            for (int c = 0; c < 3; ++c) f.sums[c] *= 0.85f;  // its colour 15% darker (a path that split)
        return f;
    };
    Auditor card;
    card.rate = 1.0;
    check(card.check(fakeHash(6), withOutlier(0, true), pixelSum).passed, "one pixel far off among a film's 4 picks passes");
    bool cardPassed = true;
    for (uint32_t k = 1; k < 60; ++k) cardPassed = cardPassed && card.check(fakeHash(6), withOutlier(4 * k, k % 10 == 0), pixelSum).passed;
    check(cardPassed, "an outlier in one film of ten (2.5% of pixels) passes over 60 films (" + card.summary() + ")");
    Auditor liar;
    liar.rate = 1.0;
    bool liarCaught = false;
    uint32_t caughtAt = 0;
    for (uint32_t k = 0; k < 60 && !liarCaught; ++k)
        if (!liar.check(fakeHash(7), withOutlier(4 * k, true), pixelSum).passed) liarCaught = true, caughtAt = k + 1;
    check(liarCaught, "an outlier in every film (25% of pixels) is caught, at film " + std::to_string(caughtAt));
    // A film 3% darker everywhere: every pixel within the outlier limit, so the bias catches it.
    Film dim = honest;
    for (size_t i = 0; i < dim.counts.size(); ++i)
        for (int c = 0; c < 3; ++c) dim.sums[i * 4 + c] *= 0.97f;
    Auditor d;
    d.rate = 1.0 / 64;
    check(!d.check(fakeHash(8), dim, pixelSum).passed, "a film 3% darker everywhere fails by its bias");
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
        // Every connection: one frame in the open each way (Hello, Welcome), the rest sealed. Four workers to the
        // master and six pairs of workers: twenty frames in the open.
        uint64_t plain = sw.master->plainFramesIn(), sealed = sw.master->sealedFramesIn();
        bool everySealed = sw.master->sealedFramesIn() > 0;
        for (auto& w : sw.workers) plain += w->plainFramesIn(), sealed += w->sealedFramesIn(), everySealed = everySealed && w->sealedFramesIn() > 0;
        check(plain == 20 && everySealed, "the traffic is sealed after each handshake (" + std::to_string(plain) + " frames in the open, " +
                                              std::to_string(sealed) + " sealed)");
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
        // Told it is refused, it never takes the master for gone (a deputy cut off took over from a live master).
        for (int k = 0; k < 100 && !sw.workers[1]->refused(); ++k) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        check(sw.workers[1]->refused(), "the quarantined node is told so");
        std::this_thread::sleep_for(std::chrono::milliseconds(2500));  // past goneAfter
        check(!sw.workers[1]->isMaster(), "and does not take over");
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
// A workstation (plan 12): one node in use from the start steps back; its cells go to the others and the picture is
// whole. Back at work, it steals again.
static void workstations() {
    const FrameShape s = shape();
    std::atomic<bool> inUse{true};
    Swarm sw;
    sw.master = std::make_unique<RhizomeNode>(options("studio rhizome key", true, 4e6));
    sw.master->start();
    for (int k = 0; k < 3; ++k) {
        NodeOptions o = options("studio rhizome key", false, k == 2 ? 2e6 : 4e6);
        o.masterAddress = sw.master->address();
        if (k == 2) o.mayWork = [&inUse] { return !inUse.load(); };
        sw.workers.push_back(std::make_unique<RhizomeNode>(o));
        sw.workers.back()->start();
    }
    for (int k = 0; k < 200 && sw.master->joined() < 3; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    sw.master->beginFrame(s, importance(s));
    const bool done = sw.master->waitFrame(60);
    double worst = 0;
    check(done && exact(*sw.master, s, &worst), "a workstation in use steps back and the others render its cells, each sample once");
    const auto by = sw.master->chunksByNode();
    const int who = sw.master->plan()->nodeIndex(sw.workers[2]->hash());
    check(who < 0 || by[size_t(who)] == 0, "it rendered nothing while in use");
    // Back at work: a second frame, and it takes part.
    inUse = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    sw.master->beginFrame(s, importance(s));
    const bool again = sw.master->waitFrame(60);
    const auto by2 = sw.master->chunksByNode();
    const int who2 = sw.master->plan()->nodeIndex(sw.workers[2]->hash());
    check(again && exact(*sw.master, s) && who2 >= 0 && by2[size_t(who2)] > 0, "back at work, it renders its share");
}

// Plan 15: each machine's GPU programs reach the others in its heartbeats, and its status shows them.
static void shaderStates() {
    Swarm sw;
    NodeOptions mo = options("studio rhizome key", true, 4e6);
    const std::string status = (std::filesystem::temp_directory_path() / ("rhizome-test-status-" + std::to_string(GetCurrentProcessId()) + ".json")).string();
    mo.statusPath = status;
    sw.master = std::make_unique<RhizomeNode>(mo);
    sw.master->setShaders("compiled");
    sw.master->start();
    for (int k = 0; k < 2; ++k) {
        NodeOptions o = options("studio rhizome key", false, 4e6);
        o.masterAddress = sw.master->address();
        sw.workers.push_back(std::make_unique<RhizomeNode>(o));
        sw.workers.back()->setShaders(k == 0 ? "missing: Test Card" : "no graphics card");
        sw.workers.back()->start();
    }
    for (int k = 0; k < 200 && sw.master->joined() < 2; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    check(sw.master->shadersOf(sw.workers[0]->hash()) == "missing: Test Card" && sw.master->shadersOf(sw.workers[1]->hash()) == "no graphics card",
          "the master hears each node's GPU programs");
    check(sw.workers[1]->shadersOf(sw.master->hash()) == "compiled", "a node hears the master's");
    // A compile under way, then done: the change heard.
    sw.workers[0]->setShaders("compiling: 3 of 9 tasks, 40 s");
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    check(sw.master->shadersOf(sw.workers[0]->hash()) == "compiling: 3 of 9 tasks, 40 s", "a compile's progress heard");
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));  // a status written since
    std::string j;
    {
        std::ifstream in(status, std::ios::binary);
        j.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const std::string txt = status.substr(0, status.size() - 4) + "txt";
    std::string text;
    {
        std::ifstream in(txt, std::ios::binary);
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    check(j.find("\"gpuPrograms\": \"compiling: 3 of 9 tasks, 40 s\"") != std::string::npos && j.find("\"gpuPrograms\": \"no graphics card\"") != std::string::npos,
          "status.json shows each machine's GPU programs");
    check(text.find("GPU programs: compiled") != std::string::npos, "status.txt shows them");
    // Its format (RHIZOME_API.md): version 1, written at a Unix time, this machine's row marked self with a plain role.
    const size_t self = j.find("\"self\": true");
    check(j.find("\"rhizome\": 1,") != std::string::npos && j.find("\"written\": ") != std::string::npos && self != std::string::npos &&
              j.find("\"role\": \"master\"", self) != std::string::npos && j.find("\"self\": true", self + 1) == std::string::npos &&
              j.find("this machine") == std::string::npos,
          "status.json is format 1: its version, the time written, one row marked self");
    std::error_code e;
    std::filesystem::remove(status, e);
    std::filesystem::remove(txt, e);
}

// Adaptive sampling across the swarm: the left half of the frame (flat) converges after its minimum samples and its
// cells' chunks are cancelled on every node; the right half (noisy) takes every sample.
static void adaptive() {
    noisyRight = true;
    {
        Swarm sw = swarm({{4e6, 0}, {4e6, 0}, {1e6, 0}}, 4e6);
        FrameShape s = shape();
        s.spp = 64, s.chunkSamples = 4, s.targetError = 0.01f, s.minSamples = 8;
        sw.master->beginFrame(s, importance(s));
        const bool done = sw.master->waitFrame(60);
        check(done, "an adaptive frame completes");
        const auto film = sw.master->film();
        const auto counts = sw.master->counts();
        bool leftShort = true, leftRight = true, rightFull = true, rightExact = true;
        size_t samples = 0;
        for (U32 y = 0; y < s.height; ++y)
            for (U32 x = 0; x < s.width; ++x) {
                const size_t i = size_t(y) * s.width + x;
                samples += counts[i];
                if (x < 256) {
                    leftShort = leftShort && counts[i] >= s.minSamples && counts[i] < s.spp;
                    leftRight = leftRight && counts[i] && std::fabs(film[i * 4] / counts[i] - 0.5) < 0.002;
                } else {
                    rightFull = rightFull && counts[i] == s.spp;
                    float ref[4];
                    pixelSum(x, y, 0, s.spp, ref);
                    rightExact = rightExact && std::fabs(film[i * 4] - ref[0]) < 1e-3 * std::max(1.0f, ref[0]);
                }
            }
        check(leftShort, "the flat half stops between its minimum and its most samples");
        check(leftRight, "and its picture is right");
        check(rightFull && rightExact, "the noisy half takes every sample, each once");
        const size_t total = size_t(s.width) * s.height * s.spp;
        check(sw.master->convergedCells() == 16 && sw.master->cancelledChunks() > 0,
              "the flat half's 16 cells converged at the master (" + std::to_string(sw.master->convergedCells()) + "), " +
                  std::to_string(sw.master->cancelledChunks()) + " chunks cancelled");
        check(samples < total * 7 / 10, "the swarm rendered " + std::to_string(samples * 100 / total) + "% of a uniform frame's samples");
    }
    noisyRight = false;
}

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
    {
        std::lock_guard<std::mutex> g(m);
        check(got, "a master finds a node of its rhizome by discovery");
        check(got && found[a->hash()].size() > 6 && found[a->hash()].substr(found[a->hash()].size() - 6) == ":47123", "at its address and TCP port (" +
              (got ? found[a->hash()] : std::string("none")) + ")");
        check(!found.count(b->hash()), "a node of another rhizome is not found");
    }

    // Plan 18: beyond broadcast. Ranges read; a machine probed by its address alone, and one found by a sweep (no
    // broadcasts: loopback's other addresses stand for a VPN's).
    Range r;
    const bool ranges = parseRange("10.8.0.0/24", r) && r.count == 254 && r.first == 0x0A080001u && parseRange("10.8.0.7", r) && r.count == 1 &&
                        r.first == 0x0A080007u && parseRange("10.8.4.9/22", r) && r.first == 0x0A080401u && r.count == 1022 &&
                        !parseRange("10.0.0.0/8", r) && !parseRange("10.8.0/24", r) && !parseRange("host/24", r);
    check(ranges, "address ranges read (a /24's 254 hosts, one address, at most /16, nonsense refused)");
    std::map<Hash, std::string> direct, swept;
    ProbeOptions one;
    one.broadcast = false;
    one.hosts = {0x7F000005u};  // 127.0.0.5
    Prober byAddress("studio rhizome key", [&](const Hash& h, const std::string& at) {
        std::lock_guard<std::mutex> g2(m);
        direct[h] = at;
    }, port, 0.2, one);
    ProbeOptions sweep;
    sweep.broadcast = false;
    parseRange("127.0.3.0/24", r);
    sweep.ranges = {r};
    sweep.perSecond = 400;
    Prober bySweep("studio rhizome key", [&](const Hash& h, const std::string& at) {
        std::lock_guard<std::mutex> g2(m);
        swept[h] = at;
    }, port, 0.2, sweep);
    const bool gotDirect = until([&] {
        std::lock_guard<std::mutex> g2(m);
        return direct.count(a->hash()) != 0;
    });
    const bool gotSwept = until([&] {
        std::lock_guard<std::mutex> g2(m);
        return swept.count(a->hash()) != 0;
    });
    std::lock_guard<std::mutex> g3(m);
    // (Windows answers from 127.0.0.1 whichever loopback address was probed: the address found is that one.)
    check(gotDirect && !direct.count(b->hash()) && byAddress.unicastProbes() > 0,
          "a machine reached by its address alone, broadcast off (" + (gotDirect ? direct[a->hash()] : std::string("none")) + ")");
    check(gotSwept && bySweep.unicastProbes() > 0,
          "a machine found by a sweep of a range (" + (gotSwept ? swept[a->hash()] : std::string("none")) + ", " +
              std::to_string(bySweep.unicastProbes()) + " addresses probed)");
}

// Master failover: the master names its successors (the fastest first) and sends the deputy each film it takes in;
// when it fails mid-frame the deputy takes over with the picture so far, the others follow it, what no one held is
// given out again, and the frame completes at the new master as one machine renders it.
// early: the master fails at a tenth of the frame, the nodes holding much of their work still (their reports count).
static void failover(bool early) {
    const std::string key = "studio rhizome key";
    const FrameShape s = shape();
    const double k = early ? 0.25 : 1.0;  // slower nodes: still holding work when the master fails
    Swarm sw = swarm({{4e5 * k, 0}, {2e5 * k, 0}, {1e5 * k, 0}}, 3e5 * k, key);
    const std::string when = early ? " (early, nodes holding work)" : "";
    sw.master->beginFrame(s, importance(s));
    const Hash fastest = sw.workers[0]->hash();
    auto order = [&](RhizomeNode& n) { return n.succession(); };
    check(until([&] { return order(*sw.workers[1]).size() == 3 && order(*sw.workers[1])[0] == fastest; }),
          "the master names its successors, the fastest first, and every node holds the list");
    // A quarter of the frame back, then the master fails.
    auto back = [&] {
        uint64_t n = 0;
        for (auto v : sw.master->chunksByNode()) n += v;
        return n;
    };
    const uint64_t total = uint64_t(s.cells()) * s.chunksPerCell();
    check(until([&] { return back() >= (early ? total / 10 : total / 4); }, 60), std::string(early ? "a tenth" : "a quarter") +
                                                                                  " of the frame comes back before the master fails");
    check(until([&] { return sw.workers[0]->replicaChunks() > 0; }), "the deputy holds the films the master took in (" +
                                                                    std::to_string(sw.workers[0]->replicaChunks()) + ")");
    sw.master->stop();
    check(until([&] { return sw.workers[0]->isMaster(); }, 10), "the deputy takes over as master" + when);
    check(until([&] { return sw.workers[1]->masterOf() == fastest && sw.workers[2]->masterOf() == fastest; }, 10),
          "the other nodes take it as their master" + when);
    const bool done = sw.workers[0]->waitFrame(early ? 180 : 90);
    double worst = 0;
    check(done && exact(*sw.workers[0], s, &worst), "the frame completes at the new master, every sample once" + when + " (worst " +
                                                         std::to_string(worst) + ")");
    for (auto& w : sw.workers) w->stop();
}

// Jobs (plan 13): frame ranges, a frame's files by their slots, a picture's name a frame, job files and the queue.
static void jobs() {
    check(parseFrames("1-5") == std::vector<int>{1, 2, 3, 4, 5}, "frames 1-5");
    check(parseFrames("1-9:4") == std::vector<int>{1, 5, 9}, "frames 1-9:4, every fourth");
    check(parseFrames("3, 7,10-12") == std::vector<int>{3, 7, 10, 11, 12}, "a list of frames and ranges");
    check(parseFrames("-2-1") == std::vector<int>{-2, -1, 0, 1} && parseFrames("").empty(), "negative frames; none");
    bool threw = false;
    try {
        parseFrames("9-1");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a range that goes down is refused");
    check(framePath("C:/shot/shot.####.rdla", 12) == "C:/shot/shot.0012.rdla", "#### is the frame, padded");
    check(framePath("shot.#.rdla", 120) == "shot.120.rdla" && framePath("shot.%04d.rdla", 7) == "shot.0007.rdla" &&
              framePath("shot.%d.rdla", 7) == "shot.7.rdla",
          "# and printf slots");
    check(framePath("C:/take#2/shot.rdla", 5) == "C:/take#2/shot.rdla", "a # in a folder is not a slot");
    check(frameOutput("out/beauty.pfm", 3, true) == "out/beauty.0003.pfm" && frameOutput("out/beauty.pfm", 3, false) == "out/beauty.pfm" &&
              frameOutput("out/b.####.exr", 3, false) == "out/b.0003.exr",
          "a picture a frame: its slot, else .NNNN for more than one frame");
    Job j;
    j.name = "hydra spin";
    j.scenes = {"C:/s/hydra.rdla", "C:/s/hydra.rdlb"};
    j.deltas = {"C:/s/hydra.####.delta.rdla"};
    j.frames = "1-24:2";
    j.out = "C:/out/hydra.####.pfm";
    const Job back = readJob(writeJob(j));
    check(back.name == j.name && back.scenes == j.scenes && back.deltas == j.deltas && back.frames == j.frames && back.out == j.out,
          "a job file reads back as written");
    check(back.scenesFor(5) == j.scenes && back.deltasFor(5) == std::vector<std::string>{"C:/s/hydra.0005.delta.rdla"} &&
              back.frameList().size() == 12,
          "a frame's scene and delta files");
    Job bad = j;
    bad.deltas = {"C:/s/no-slot.rdla"};
    threw = false;
    try {
        bad.check();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a delta without a slot is refused");
    // The queue: jobs taken in the order they came, finished into done\ and failed\.
    namespace fs = std::filesystem;
    const fs::path q = fs::temp_directory_path() / ("rhizome-queue-" + std::to_string(GetCurrentProcessId()));
    std::error_code e;
    fs::remove_all(q, e);
    Job one = j, two = j;
    one.name = "one", two.name = "two";
    queue::submit(q, one);
    queue::submit(q, two);
    check(queue::waiting(q).size() == 2, "two jobs queued");
    Job taken;
    const fs::path r1 = queue::take(q, taken);
    check(!r1.empty() && taken.name == "one" && fs::exists(q / "running" / r1.filename()) && queue::waiting(q).size() == 1,
          "the first job taken, moved to running");
    queue::finish(r1, q, true, "12 frames in 40 s");
    Job taken2;
    const fs::path r2 = queue::take(q, taken2);
    queue::finish(r2, q, false, "the scene did not load");
    Job none;
    check(taken2.name == "two" && queue::take(q, none).empty(), "then the second; then none");
    std::ifstream done(q / "done" / r1.filename());
    const std::string doneText((std::istreambuf_iterator<char>(done)), std::istreambuf_iterator<char>());
    check(doneText.find("# done: 12 frames in 40 s") != std::string::npos && fs::exists(q / "failed" / r2.filename()),
          "finished jobs moved to done and failed, with what came of them");
    done.close();
    fs::remove_all(q, e);
}

// Frames one after another on one swarm (plan 13): each frame's picture its own, a film of the frame before that lands
// late left out; a node ready for one manifest is not ready for the next until it says so.
// Checkpoints (plan 14): a frame stopped part way and every machine gone; a new swarm (other machines) takes it up from
// the master's checkpoint and renders only what was not back: the picture every sample once.
static void checkpoints() {
    const FrameShape s = shape();
    Bytes ck;
    size_t backThen = 0, all = 0;
    {
        Swarm sw = swarm({{4e5, 0}, {4e5, 0}}, 4e5);
        sw.master->beginFrame(s, importance(s));
        for (int k = 0; k < 400; ++k) {
            std::tie(backThen, all) = sw.master->chunksBack();
            if (backThen * 3 >= all) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        ck = sw.master->checkpoint();
        std::tie(backThen, all) = sw.master->chunksBack();  // the checkpoint's (none more lands: taken under the lock)
        check(!ck.empty() && backThen > 0 && backThen < all,
              "a checkpoint part way: " + std::to_string(backThen) + " of " + std::to_string(all) + " chunks back");
        sw.master->stop();
        for (auto& w : sw.workers) w->stop();
    }
    {
        Swarm sw = swarm({{4e6, 0}, {4e6, 0}, {2e6, 0}}, 4e6);
        const bool resumed = sw.master->beginFrame(s, importance(s), &ck);
        const bool done = sw.master->waitFrame(60);
        uint64_t rendered = 0;
        for (auto n : sw.master->chunksByNode()) rendered += n;
        double worst = 0;
        check(resumed && done && exact(*sw.master, s, &worst), "new machines take it up: every pixel its samples, each once");
        const size_t fromCheckpoint = sw.master->plan()->done.size();
        check(fromCheckpoint >= backThen && rendered + fromCheckpoint == all,
              "only the chunks not back were rendered (" + std::to_string(rendered) + ", " + std::to_string(fromCheckpoint) +
                  " from the checkpoint, of " + std::to_string(all) + ")");
    }
    {
        // A checkpoint of another frame's shape is not used.
        Swarm sw = swarm({{4e6, 0}}, 4e6);
        FrameShape other = s;
        other.spp = 32;
        const bool resumed = sw.master->beginFrame(other, importance(other), &ck);
        check(!resumed && sw.master->waitFrame(60), "a checkpoint of another shape is not used; the frame renders anew");
    }
    {
        // A checkpoint with every chunk back: the frame completes at once.
        Swarm sw = swarm({{4e6, 0}}, 4e6);
        sw.master->beginFrame(s, importance(s));
        sw.master->waitFrame(60);
        const Bytes whole = sw.master->checkpoint();
        sw.master->beginFrame(s, importance(s), &whole);
        check(sw.master->waitFrame(1) && exact(*sw.master, s), "a checkpoint of a whole frame: complete at once, as it was");
    }
}

static void frames() {
    std::atomic<int> answering{1};
    struct Worker {
        RhizomeNode* node = nullptr;
    };
    std::vector<std::shared_ptr<Worker>> ws;
    Swarm sw;
    NodeOptions mo = options("studio rhizome key", true, 4e6);
    mo.readyAtJoin = false;
    sw.master = std::make_unique<RhizomeNode>(mo);
    sw.master->start();
    for (int k = 0; k < 3; ++k) {
        auto w = std::make_shared<Worker>();
        // One slow node: its chunks are still rendering as a frame ends, their films landing in the next.
        NodeOptions o = options("studio rhizome key", false, k == 2 ? 2e5 : 4e6);
        o.masterAddress = sw.master->address();
        o.readyAtJoin = false;
        o.onManifest = [w, &answering](const Bytes& m) {
            if (answering && w->node) w->node->markReady(sha256(m));
        };
        sw.workers.push_back(std::make_unique<RhizomeNode>(o));
        w->node = sw.workers.back().get();
        ws.push_back(w);
        sw.workers.back()->start();
    }
    for (int k = 0; k < 200 && sw.master->joined() < 3; ++k) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    sw.master->publishManifest(Bytes{'f', '1'});
    check(until([&] { return sw.master->ready() == 3; }), "three nodes ready for the first frame");
    answering = 0;
    sw.master->publishManifest(Bytes{'f', '2'});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    check(sw.master->ready() == 0, "none ready for the next frame's manifest until they say so");
    answering = 1;
    sw.master->publishManifest(Bytes{'f', '3'});
    check(until([&] { return sw.master->ready() == 3; }), "then all three");
    FrameShape s = shape();
    bool each = true;
    for (U32 f = 1; f <= 3; ++f) {
        s.frame = f;
        sceneFrame = float(f);
        sw.master->beginFrame(s, importance(s));
        const bool done = sw.master->waitFrame(60);
        const auto film = sw.master->film();
        const auto counts = sw.master->counts();
        bool ok = done;
        for (size_t i = 0; ok && i < counts.size(); ++i) {
            const U32 x = U32(i % s.width), y = U32(i / s.width);
            float ref[4];
            pixelSum(x, y, 0, s.spp, ref);
            ok = counts[i] == s.spp && std::fabs(film[i * 4 + 1] - (ref[1] + float(f) * s.spp)) < 1e-3f * (ref[1] + float(f) * s.spp);
        }
        each = each && ok;
        check(ok, "frame " + std::to_string(f) + ": its own picture, every sample once, none of the frame before's");
    }
    sceneFrame = 0;
    check(each, "three frames in a row on one swarm");
}

int main() {
    // RHIZOME_TEST_ONLY=<section>: that section alone (identities, encryption, messages, plans, importance, memory, audits, swarms, adaptive, workstations, shaders, jobs, frames, checkpoints, scene, approvals,
    // discovery, failover).
    const char* only = std::getenv("RHIZOME_TEST_ONLY");
    auto run = [&](const char* name, const std::function<void()>& f) {
        if (only && std::string(only) != name) return;
        f();
        std::fflush(stdout);
    };
    run("identities", identities);
    run("encryption", encryption);
    run("messages", messages);
    run("plans", plans);
    run("importance", importances);
    run("memory", memory);
    run("audits", audits);
    run("swarms", swarms);
    run("adaptive", adaptive);
    run("workstations", workstations);
    run("shaders", shaderStates);
    run("jobs", jobs);
    run("frames", frames);
    run("checkpoints", checkpoints);
    run("scene", [] {
        sceneThroughSwarm(false);
        sceneThroughSwarm(true);
    });
    run("approvals", approvals);
    run("discovery", discovery);
    run("failover-early", [] { failover(true); });
    run("failover", [] {
        failover(false);
        failover(true);
    });
    std::printf("{\"checks\": %d, \"failures\": %d}\n", checks, failures);
    return failures ? 1 : 0;
}
