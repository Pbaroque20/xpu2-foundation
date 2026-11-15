// Rhizome's scheduling (RHIZOME.md, 3-4): the master's plan of a frame (tiers of nodes by rate, the important cells to the
// fast tier and the rest to the slow tier, each node's cells kept together along a Morton curve, chunk sizes by
// distance), its ledger (which chunks came back, from whom; what a lost node held, given to the nearest), and a
// node's holder (the ranges it holds, the order it works them, what it gives a node that steals).
//
// A chunk is a cell's sample range [index * ChunkSamples, +ChunkSamples): a cell of spp samples has spp / ChunkSamples
// of them. Only the holder of a chunk hands it on, and the ledger takes each chunk's film once, so each sample is in
// the picture exactly once whatever is lost or stolen on the way (a chunk may be rendered twice after a loss; the
// second film is thrown away).
#pragma once
#include "rhizome_identity.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <vector>

namespace xpu2::interactive::rhizome {

using U32 = uint32_t;

struct Chunk {
    U32 cell = 0, index = 0;
    bool operator<(const Chunk& o) const { return cell != o.cell ? cell < o.cell : index < o.index; }
    bool operator==(const Chunk& o) const { return cell == o.cell && index == o.index; }
};

// A node as the scheduler sees it.
struct Node {
    Hash hash{};
    double rate = 0;   // samples a second (0: unknown, taken as the others' median)
    double rtt = 0;    // seconds to the master
    bool fast = false; // its tier, set by the plan
};

struct FrameShape {
    U32 width = 0, height = 0, cellSize = 256, spp = 64, chunkSamples = 4;
    // Adaptive sampling across the swarm (RHIZOME.md, plan 10): a cell's chunks past minSamples are cancelled once its
    // pooled error is under targetError (0: uniform, every chunk rendered). The halves' error needs even chunks.
    float targetError = 0;
    U32 minSamples = 0;
    // The frame's number in its job (plan 13): a film rendered for another frame is not this one's.
    U32 frame = 0;
    bool adaptive() const { return targetError > 0 && chunkSamples % 2 == 0; }
    U32 cellsX() const { return (width + cellSize - 1) / cellSize; }
    U32 cellsY() const { return (height + cellSize - 1) / cellSize; }
    U32 cells() const { return cellsX() * cellsY(); }
    U32 chunksPerCell() const { return (spp + chunkSamples - 1) / chunkSamples; }
    // A chunk's samples: [first, first + count).
    U32 first(U32 index) const { return index * chunkSamples; }
    U32 count(U32 index) const { return std::min(chunkSamples, spp - first(index)); }
    // A cell's pixels (min x, min y, max x, max y; half open).
    void rect(U32 cell, U32 r[4]) const {
        const U32 cx = cell % cellsX(), cy = cell / cellsX();
        r[0] = cx * cellSize, r[1] = cy * cellSize;
        r[2] = std::min(width, r[0] + cellSize), r[3] = std::min(height, r[1] + cellSize);
    }
    U32 pixels(U32 cell) const {
        U32 r[4];
        rect(cell, r);
        return (r[2] - r[0]) * (r[3] - r[1]);
    }
    bool adjacent(U32 a, U32 b) const {
        const int ax = int(a % cellsX()), ay = int(a / cellsX()), bx = int(b % cellsX()), by = int(b / cellsX());
        return std::abs(ax - bx) + std::abs(ay - by) == 1;
    }
};

inline U32 morton(U32 x, U32 y) {
    auto spread = [](U32 v) {
        v &= 0xffff;
        v = (v | (v << 8)) & 0x00ff00ff;
        v = (v | (v << 4)) & 0x0f0f0f0f;
        v = (v | (v << 2)) & 0x33333333;
        v = (v | (v << 1)) & 0x55555555;
        return v;
    };
    return spread(x) | (spread(y) << 1);
}

// The master's plan of a frame: each cell's owner and each node's chunk size.
struct Plan {
    FrameShape shape;
    std::vector<int> owner;         // a cell's node, by its place in nodes
    std::vector<unsigned char> important;
    std::vector<Node> nodes;
    std::vector<U32> grantChunks;   // a node's chunks a grant, at least one (far nodes take more at a time)
    // Chunks already in the picture (a frame taken up again from its checkpoint, plan 14), as cell * chunksPerCell +
    // index: no one is given them, the ledger has them back.
    std::vector<U32> done;
    U32 kept = 0;  // cells given back to the node that rendered them in the frame before (make's affinity; not sent)
    bool isDone(U32 cell, U32 index) const {
        const U32 id = cell * shape.chunksPerCell() + index;
        return std::binary_search(done.begin(), done.end(), id);
    }

    // Tiers by rate: a node is fast when its rate is at least fastShare of the fastest's. Then the important cells
    // (importance above the mean of the cells') go to the fast tier and the rest to the slow tier, each node's run of
    // cells along the Morton curve sized by its rate, so a node's cells are together (its caches warm) and the
    // expected times even.
    // affinity (plan 17): each cell's node in the frame before (the one that rendered most of it; a zero hash: none).
    // A cell goes back to it first, while that node is in the plan, of the tier the cell asks for and short of its
    // share: the texture tiles the cell reads are there already (fetched from the swarm's memory as it rendered), its
    // caches warm. The cells left are dealt as above.
    static Plan make(const FrameShape& shape, std::vector<Node> nodes, const std::vector<float>& importance, double fastShare = 0.5,
                     double targetSeconds = 0.25, const std::vector<Hash>* affinity = nullptr) {
        Plan p;
        p.shape = shape;
        const U32 cells = shape.cells();
        p.owner.assign(cells, -1);
        p.important.assign(cells, 0);
        if (nodes.empty() || cells == 0) {
            p.nodes = std::move(nodes);
            return p;
        }
        // Unknown rates: the known ones' median (one when none is known).
        std::vector<double> known;
        for (const auto& n : nodes)
            if (n.rate > 0) known.push_back(n.rate);
        std::sort(known.begin(), known.end());
        const double median = known.empty() ? 1.0 : known[known.size() / 2];
        double fastest = 0;
        for (auto& n : nodes) {
            if (n.rate <= 0) n.rate = median;
            fastest = std::max(fastest, n.rate);
        }
        for (auto& n : nodes) n.fast = n.rate >= fastShare * fastest;
        p.nodes = std::move(nodes);

        // Important cells: above the mean importance (none when the importance is flat or not known).
        double mean = 0;
        for (U32 c = 0; c < cells; ++c) mean += c < importance.size() ? importance[c] : 0.0f;
        mean /= cells;
        for (U32 c = 0; c < cells; ++c) p.important[c] = c < importance.size() && importance[c] > mean ? 1 : 0;

        // The cells in Morton order, important and not.
        std::vector<U32> order(cells);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](U32 a, U32 b) {
            return morton(a % shape.cellsX(), a / shape.cellsX()) < morton(b % shape.cellsX(), b / shape.cellsX());
        });
        // A node's share of the frame's pixels by its rate.
        double totalRate = 0, totalPixels = 0;
        for (const auto& n : p.nodes) totalRate += n.rate;
        for (U32 c = 0; c < cells; ++c) totalPixels += shape.pixels(c);
        std::vector<double> target(p.nodes.size()), given(p.nodes.size(), 0);
        for (size_t i = 0; i < p.nodes.size(); ++i) target[i] = totalPixels * p.nodes[i].rate / totalRate;

        // Each cell back to its node of the frame before (affinity), along the curve, while the node is short of its share.
        if (affinity && affinity->size() == cells) {
            for (U32 c : order) {
                const Hash& h = (*affinity)[c];
                if (h == Hash{}) continue;
                const int i = p.nodeIndex(h);
                if (i < 0 || (p.important[c] && !p.nodes[size_t(i)].fast) || given[size_t(i)] + shape.pixels(c) > target[size_t(i)]) continue;
                p.owner[c] = i;
                given[size_t(i)] += shape.pixels(c);
                ++p.kept;
            }
        }
        std::vector<U32> imp, rest;
        for (U32 c : order)
            if (p.owner[c] < 0) (p.important[c] ? imp : rest).push_back(c);

        std::vector<size_t> fast, slow;
        for (size_t i = 0; i < p.nodes.size(); ++i) (p.nodes[i].fast ? fast : slow).push_back(i);
        // Runs along a list of cells: each node of a tier in turn takes the next cells until its share is met.
        auto deal = [&](const std::vector<U32>& list, size_t& at, const std::vector<size_t>& tier, bool fill) {
            for (size_t k = 0; k < tier.size() && at < list.size(); ++k) {
                const size_t i = tier[k];
                const bool last = k + 1 == tier.size();
                while (at < list.size() && (given[i] < target[i] || (last && fill))) {
                    p.owner[list[at]] = int(i);
                    given[i] += shape.pixels(list[at]);
                    ++at;
                }
            }
        };
        size_t ai = 0, ar = 0;
        // The important cells go to the fast tier, all of them (filling past its share where it must).
        deal(imp, ai, fast, true);
        // The rest to the slow tier, then whatever is left to the fast tier's remaining shares, then to anyone.
        deal(rest, ar, slow, false);
        deal(rest, ar, fast, false);
        if (ar < rest.size()) {
            const auto& everyone = slow.empty() ? fast : slow;
            deal(rest, ar, everyone, true);
        }

        // Chunks a grant: enough that a round trip is a small part of rendering them (far nodes take more at once).
        p.grantChunks.resize(p.nodes.size());
        const double chunkPixels = double(shape.cellSize) * shape.cellSize * shape.chunkSamples;
        for (size_t i = 0; i < p.nodes.size(); ++i) {
            const double chunkSeconds = chunkPixels / std::max(1.0, p.nodes[i].rate);
            const double want = std::max(targetSeconds, 8.0 * p.nodes[i].rtt) / std::max(1e-6, chunkSeconds);
            p.grantChunks[i] = U32(std::clamp(want, 1.0, double(shape.chunksPerCell())));
        }
        return p;
    }
    int nodeIndex(const Hash& h) const {
        for (size_t i = 0; i < nodes.size(); ++i)
            if (nodes[i].hash == h) return int(i);
        return -1;
    }
};

// The master's ledger of a frame: each chunk's holder as far as the master knows, and whether its film came back.
class Ledger {
public:
    explicit Ledger(const Plan& plan) : plan_(plan) {
        const U32 per = plan.shape.chunksPerCell();
        for (U32 c = 0; c < plan.shape.cells(); ++c)
            for (U32 k = 0; k < per; ++k) holder_[Chunk{c, k}] = plan.owner[c];
        outstanding_ = holder_.size();
        for (U32 id : plan.done) {
            const Chunk c{id / per, id % per};
            if (holder_.count(c) && returned_.insert(c).second) --outstanding_;
        }
    }
    // A grant reported: the chunks are the thief's now.
    void granted(const std::vector<Chunk>& chunks, int to) {
        for (const auto& c : chunks)
            if (!returned_.count(c)) holder_[c] = to;
    }
    // A chunk's film arrived (and passed the audit): true when it is the first, to be added to the picture.
    bool arrived(const Chunk& c) {
        if (!holder_.count(c) || returned_.count(c)) return false;
        returned_.insert(c);
        --outstanding_;
        return true;
    }
    bool returned(const Chunk& c) const { return returned_.count(c) != 0; }
    // A cell converged (adaptive): its chunks not back are not needed; the ones cancelled.
    std::vector<Chunk> cancel(U32 cell) {
        std::vector<Chunk> out;
        for (U32 k = 0; k < plan_.shape.chunksPerCell(); ++k) {
            const Chunk c{cell, k};
            if (!holder_.count(c) || returned_.count(c)) continue;
            returned_.insert(c);
            --outstanding_;
            out.push_back(c);
        }
        cancelled_ += out.size();
        return out;
    }
    size_t cancelled() const { return cancelled_; }
    size_t outstanding() const { return outstanding_; }
    bool complete() const { return outstanding_ == 0; }
    // A node lost (gone, or quarantined): the chunks it held that have not come back, each cell's given to the node
    // best placed to take it (an owner of a neighbouring cell, of the tier the cell's importance asks for, nearest
    // to the master; any working node else). alive[i]: whether node i can take work. Returns the new holders.
    std::map<int, std::vector<Chunk>> lost(int node, const std::vector<bool>& alive) {
        std::map<U32, std::vector<Chunk>> byCell;
        for (const auto& [c, h] : holder_)
            if (h == node && !returned_.count(c)) byCell[c.cell].push_back(c);
        std::map<int, std::vector<Chunk>> out;
        for (auto& [cell, chunks] : byCell) {
            const int to = heir(cell, node, alive);
            if (to < 0) continue;
            for (const auto& c : chunks) holder_[c] = to;
            auto& v = out[to];
            v.insert(v.end(), chunks.begin(), chunks.end());
        }
        return out;
    }
    int holder(const Chunk& c) const {
        auto it = holder_.find(c);
        return it == holder_.end() ? -1 : it->second;
    }

private:
    int heir(U32 cell, int dead, const std::vector<bool>& alive) const {
        const bool wantFast = plan_.important[cell] != 0;
        int best = -1;
        double bestScore = 1e300;
        for (size_t i = 0; i < plan_.nodes.size(); ++i) {
            if (int(i) == dead || i >= alive.size() || !alive[i]) continue;
            const Node& n = plan_.nodes[i];
            bool neighbour = false;
            for (U32 c = 0; c < plan_.shape.cells() && !neighbour; ++c)
                neighbour = plan_.owner[c] == int(i) && plan_.shape.adjacent(c, cell);
            // Lower is better: the wrong tier, not a neighbour, then distance, then a slower rate.
            const double score = (n.fast != wantFast ? 1e6 : 0) + (neighbour ? 0 : 1e3) + n.rtt * 10 + 1.0 / std::max(1e-9, n.rate);
            if (score < bestScore) bestScore = score, best = int(i);
        }
        return best;
    }
    const Plan& plan_;
    std::map<Chunk, int> holder_;
    std::set<Chunk> returned_;
    size_t outstanding_ = 0, cancelled_ = 0;
};

// A node's chunks: those its plan gave it, those granted to it; the order it works them and what it gives away.
// Owned by the node's work thread.
class Holder {
public:
    Holder(const Plan& plan, int self) : plan_(plan), self_(self) {
        const U32 per = plan.shape.chunksPerCell();
        for (U32 c = 0; c < plan.shape.cells(); ++c)
            if (plan.owner[c] == self)
                for (U32 k = 0; k < per; ++k)
                    if (!plan.isDone(c, k)) held_[c].insert(k);
    }
    // Chunks given to this node (a grant, or a lost node's).
    void receive(const std::vector<Chunk>& chunks) {
        for (const auto& c : chunks) held_[c.cell].insert(c.index);
    }
    // The next chunk to render: in the cell it is in while that has work, else the cell with the least done (its
    // lowest chunk index), important cells first on a tie, so the node's picture converges evenly.
    std::optional<Chunk> next() {
        if (current_ && held_.count(*current_) && !held_[*current_].empty() && sinceSwitch_ < plan_.grantChunks[self_]) {
            ++sinceSwitch_;
            return take(*current_);
        }
        std::optional<U32> best;
        for (const auto& [cell, ks] : held_) {
            if (ks.empty()) continue;
            if (!best) {
                best = cell;
                continue;
            }
            const U32 a = *ks.begin(), b = *held_[*best].begin();
            if (a < b || (a == b && plan_.important[cell] > plan_.important[*best])) best = cell;
        }
        if (!best) return std::nullopt;
        current_ = best;
        sinceSwitch_ = 1;
        return take(*best);
    }
    // Every chunk this node holds, given up (a workstation stepping back).
    std::vector<Chunk> takeAll() {
        std::vector<Chunk> out;
        for (const auto& [cell, ks] : held_)
            for (U32 k : ks) out.push_back(Chunk{cell, k});
        held_.clear();
        current_.reset();
        return out;
    }
    // A cell converged: its chunks dropped.
    void drop(U32 cell) {
        held_.erase(cell);
        if (current_ && *current_ == cell) current_.reset();
    }
    // A node asks for work. The grant: from the cell with the most left that the thief may take (a slow node may not
    // take an important cell from a fast holder), half its remaining chunks, at least one, at most the thief's
    // grant size, taken from the end of its sample ranges so the holder's own progressive order is undisturbed.
    std::vector<Chunk> grant(int thief) {
        const Node& t = plan_.nodes[size_t(thief)];
        const bool selfFast = plan_.nodes[size_t(self_)].fast;
        std::optional<U32> pick;
        size_t most = 0;
        for (const auto& [cell, ks] : held_) {
            if (ks.size() < 2 && !(ks.size() == 1 && cell != current_)) continue;  // keep the one chunk in hand
            if (plan_.important[cell] && selfFast && !t.fast) continue;
            if (ks.size() > most) most = ks.size(), pick = cell;
        }
        std::vector<Chunk> out;
        if (!pick) return out;
        auto& ks = held_[*pick];
        const size_t give = std::min<size_t>(std::max<size_t>(1, ks.size() / 2), plan_.grantChunks[size_t(thief)]);
        for (size_t k = 0; k < give && !ks.empty(); ++k) {
            auto last = std::prev(ks.end());
            out.push_back(Chunk{*pick, *last});
            ks.erase(last);
        }
        return out;
    }
    size_t remaining() const {
        size_t n = 0;
        for (const auto& [c, ks] : held_) n += ks.size();
        return n;
    }
    // The chunks it holds (a node's report to a master that took over).
    std::vector<Chunk> held() const {
        std::vector<Chunk> out;
        for (const auto& [c, ks] : held_)
            for (U32 k : ks) out.push_back(Chunk{c, k});
        return out;
    }
    // The owners to ask for work, best first: owners of cells next to this node's (by the plan), then the others;
    // nearer and of the cell's tier first. A slow node asks fast ones last.
    std::vector<int> victims() const {
        std::vector<std::pair<double, int>> v;
        for (size_t i = 0; i < plan_.nodes.size(); ++i) {
            if (int(i) == self_) continue;
            bool neighbour = false;
            for (U32 a = 0; a < plan_.shape.cells() && !neighbour; ++a)
                if (plan_.owner[a] == self_)
                    for (U32 b = 0; b < plan_.shape.cells() && !neighbour; ++b)
                        neighbour = plan_.owner[b] == int(i) && plan_.shape.adjacent(a, b);
            const bool wrongTier = !plan_.nodes[size_t(self_)].fast && plan_.nodes[i].fast;
            v.push_back({(neighbour ? 0 : 1e3) + (wrongTier ? 1e2 : 0) + plan_.nodes[i].rtt, int(i)});
        }
        std::sort(v.begin(), v.end());
        std::vector<int> out;
        for (const auto& [s, i] : v) out.push_back(i);
        return out;
    }

private:
    Chunk take(U32 cell) {
        auto& ks = held_[cell];
        const U32 k = *ks.begin();
        ks.erase(ks.begin());
        return Chunk{cell, k};
    }
    const Plan& plan_;
    int self_;
    std::map<U32, std::set<U32>> held_;
    std::optional<U32> current_;
    U32 sinceSwitch_ = 0;
};

}  // namespace xpu2::interactive::rhizome
