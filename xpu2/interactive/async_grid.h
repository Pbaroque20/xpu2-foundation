// The work grid and its cellular rules (spec §7-8), host side.
//
// The image is cut into square cells (64 px by default). Each cell holds its
// unstarted work as a queue of chunks, ordered by sample range and then tile, so
// a cell converges evenly. Cells start owned by the device the scene indexer
// finds suited to them (device_suit.h), else in a CPU/GPU checkerboard. Every
// claim is one atomic on a cursor in host memory: a chunk is claimed exactly
// once (CA-06), and nothing is ever shared across PCIe. The GPU lane's broker
// (async_render.h) claims here on the GPU's behalf, for its cellular workers,
// and hands the chunks to the device through its mailbox ring, so stealing in
// either direction needs no cross-device atomics (MBX-03).
//
// A uniform frame refines the whole picture as MoonRay's passes do: its sample
// ranges grow 1, 1, 2, then the chunk size, and no chunk is handed to a device
// past the least-sampled range of the cells it works without a right of
// passage (its frontier: its own reserved cells and the open ones, so the
// slower device's cells never hold the faster one back); an important cell may
// run a range ahead. Important parts first: once the
// picture so far is scored, the cells with the most in them (above the mean)
// stay their owner's, and the rest are anyone's.
//
// Workers of both devices (CPU threads, and the GPU's cellular workers, each a
// share of a launch) move by the cellular rules: finish the cell they are in,
// then their device's own reserved cells, then the open ones (the most
// important first), then by a right of passage the other device's; neighbours
// before the hottest cell anywhere. Ownership persists across frames and flips
// towards the device that rendered most of a cell (CA-10, CA-12).
#pragma once
#include "async_integrator.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace xpu2::interactive::async {

enum class Device : U32 { Cpu = 0, Gpu = 1 };

// A cell's queue of chunks. It only grows: the coordinator appends a pass's chunks
// while workers claim, so chunks live in fixed blocks that never move, and a claim
// never passes the published count (single writer, many claimants).
class ChunkQueue {
public:
    static constexpr U32 BlockSize = 4096, MaxBlocks = 2048;
    ChunkQueue() : blocks_(new std::atomic<Chunk*>[MaxBlocks]) {
        for (U32 i = 0; i < MaxBlocks; ++i) blocks_[i].store(nullptr, std::memory_order_relaxed);
    }
    ~ChunkQueue() {
        for (U32 i = 0; i < MaxBlocks; ++i) delete[] blocks_[i].load(std::memory_order_relaxed);
    }
    ChunkQueue(const ChunkQueue&) = delete;

    // Coordinator only.
    void append(const Chunk* chunks, U32 n) {
        U32 at = published_.load(std::memory_order_relaxed);
        if (U64(at) + n > U64(BlockSize) * MaxBlocks) throw std::length_error("Cell chunk queue full");
        for (U32 i = 0; i < n; ++i, ++at) {
            auto& block = blocks_[at / BlockSize];
            Chunk* b = block.load(std::memory_order_relaxed);
            if (!b) {
                b = new Chunk[BlockSize];
                block.store(b, std::memory_order_relaxed);
            }
            b[at % BlockSize] = chunks[i];
        }
        published_.store(at, std::memory_order_release);
    }
    U32 published() const { return published_.load(std::memory_order_acquire); }
    U32 taken() const { return next_.load(std::memory_order_acquire); }
    U32 remaining() const {
        const U32 end = published(), taken = next_.load(std::memory_order_acquire);
        return taken >= end ? 0 : end - taken;
    }
    // Up to n chunks, none at or past position limit; first: where they start in the queue.
    U32 claim(U32 n, Chunk* out, U32 limit = ~0u, U32* first = nullptr) {
        const U32 end = std::min(published(), limit);
        U32 start = next_.load(std::memory_order_relaxed);
        do {
            if (start >= end) return 0;
        } while (!next_.compare_exchange_weak(start, std::min(end, start + n), std::memory_order_acq_rel,
                                              std::memory_order_relaxed));
        const U32 got = std::min(end, start + n) - start;
        if (first) *first = start;
        for (U32 i = 0; i < got; ++i) out[i] = blocks_[(start + i) / BlockSize].load(std::memory_order_relaxed)[(start + i) % BlockSize];
        return got;
    }

private:
    std::unique_ptr<std::atomic<Chunk*>[]> blocks_;
    alignas(64) std::atomic<U32> published_{0};
    alignas(64) std::atomic<U32> next_{0};
};

struct GridOptions {
    U32 cellSize = 64;        // pixels; a multiple of 8 (GRD-01)
    U32 samplesPerChunk = 4;  // sample indices per chunk
    int mode = 2;             // 0 CPU only, 1 GPU only, 2 both
    // A uniform frame's progressive order: ranges of 1, 1, 2, then samplesPerChunk samples, and the frontier (false:
    // ranges of samplesPerChunk, handed out as they come).
    bool progressive = true;
    // A share of a uniform frame (Rhizome, RHIZOME.md: a node's chunk): its sample indices [sampleFirst, sampleFirst +
    // sampleCount) of the frame's spp, each pixel's samples the ones a whole frame gives it (0: all of them).
    U32 sampleFirst = 0, sampleCount = 0;
};

class Grid {
public:
    struct Cell {
        U32 x = 0, y = 0;
        Device owner = Device::Cpu;
        ChunkQueue queue;
        std::atomic<U64> samples[2] = {};  // rendered per device, for ownership flips
        U32 tiles = 0;                     // its tiles with pixels in the frame: the chunks of one sample range
        float gpu = 0.5f;                  // the GPU's probability of being its efficient owner (device_suit.h)
        std::atomic<U32> score{0};         // important parts first: what is in it (setScores), a float's bits
        std::atomic<bool> important{false};
    };
    // A sample range of a uniform frame.
    struct Range {
        U32 first, count;
    };

    // A uniform frame: every chunk of every sample range is published at once. With
    // empty = true the cells start empty, for passes appended later (adaptive).
    // viewport: the pixels rendered (min x, min y, max x, max y, half open; null: all), whose tiles alone take chunks.
    Grid(U32 width, U32 height, U32 spp, GridOptions options, const std::vector<Device>* owners = nullptr,
         bool empty = false, const int* viewport = nullptr)
        : width_(width), height_(height), spp_(spp), options_(options) {
        if (viewport)
            for (int i = 0; i < 4; ++i) viewport_[i] = viewport[i];
        if (!width || !height || !spp || !options.cellSize || options.cellSize % TileSize ||
            !options.samplesPerChunk || options.mode < 0 || options.mode > 2)
            throw std::invalid_argument("Invalid work grid");
        cellsX_ = (width + options.cellSize - 1) / options.cellSize;
        cellsY_ = (height + options.cellSize - 1) / options.cellSize;
        const U32 tilesX = (width + TileSize - 1) / TileSize, tilesY = (height + TileSize - 1) / TileSize;
        const U32 tilesPerCell = options.cellSize / TileSize;
        cells_ = std::vector<Cell>(cellsX_ * cellsY_);
        if (owners && owners->size() != cells_.size()) owners = nullptr;
        // A uniform frame's sample ranges: growing ones first in the progressive order.
        if (!empty) {
            const U32 from = options.sampleCount ? options.sampleFirst : 0u,
                      to = options.sampleCount ? std::min(spp, options.sampleFirst + options.sampleCount) : spp;
            for (U32 first = from; first < to;) {
                const U32 done = first - from;
                const U32 size = !options.progressive ? options.samplesPerChunk : done < 2 ? 1u : done < 4 ? 2u : options.samplesPerChunk;
                ranges_.push_back({first, std::min(size, to - first)});
                first += ranges_.back().count;
            }
        }
        for (U32 cy = 0; cy < cellsY_; ++cy)
            for (U32 cx = 0; cx < cellsX_; ++cx) {
                Cell& cell = cells_[cy * cellsX_ + cx];
                const U32 index = cy * cellsX_ + cx;
                cell.x = cx;
                cell.y = cy;
                cell.owner = options.mode == 0   ? Device::Cpu
                             : options.mode == 1 ? Device::Gpu
                             : owners            ? (*owners)[index]
                                                 : ((cx + cy) % 2 == 0 ? Device::Gpu : Device::Cpu);  // GRD-02
                if (empty) continue;
                std::vector<Chunk> chunks;
                for (const Range& range : ranges_)
                    for (U32 ty = cy * tilesPerCell; ty < std::min(tilesY, (cy + 1) * tilesPerCell); ++ty)
                        for (U32 tx = cx * tilesPerCell; tx < std::min(tilesX, (cx + 1) * tilesPerCell); ++tx)
                            if (tilePixels(ty * tilesX + tx)) chunks.push_back({ty * tilesX + tx, range.first, range.count, index});
                cell.tiles = ranges_.empty() ? 0 : U32(chunks.size() / ranges_.size());
                cell.queue.append(chunks.data(), U32(chunks.size()));
                total_ += chunks.size();
            }
        refreshFrontiers();
    }
    Grid(const Grid&) = delete;

    U32 width() const { return width_; }
    U32 height() const { return height_; }
    U32 spp() const { return spp_; }
    U32 cellCount() const { return U32(cells_.size()); }
    U64 totalChunks() const { return total_; }
    const Cell& cell(U32 index) const { return cells_[index]; }
    Device owner(U32 index) const { return cells_[index].owner; }

    U32 remaining(U32 index) const { return closed() ? 0u : cells_[index].queue.remaining(); }
    // Coordinator only: publishes more chunks of one cell.
    void append(U32 index, const Chunk* chunks, U32 n) {
        cells_[index].queue.append(chunks, n);
        total_ += n;
    }
    // A sample the GPU took and the CPU lane rendered (a deferral): the CPU's in its cell, for the ownership flips.
    void deferredToCpu(U32 tile) {
        Cell& c = cells_[cellOf(tile)];
        U64 gpu = c.samples[1].load(std::memory_order_relaxed);
        while (gpu && !c.samples[1].compare_exchange_weak(gpu, gpu - 1, std::memory_order_relaxed)) {}
        c.samples[0].fetch_add(1, std::memory_order_relaxed);
    }
    // The cell holding a tile.
    U32 cellOf(U32 tile) const {
        const U32 tilesX = (width_ + TileSize - 1) / TileSize, perCell = options_.cellSize / TileSize;
        return ((tile / tilesX) / perCell) * cellsX_ + (tile % tilesX) / perCell;
    }
    // A tile's pixels inside the image.
    U32 tilePixels(U32 tile) const {
        const U32 tilesX = (width_ + TileSize - 1) / TileSize;
        const int x = int((tile % tilesX) * TileSize), y = int((tile / tilesX) * TileSize);
        const int x0 = std::max(x, viewport_[0]), y0 = std::max(y, viewport_[1]);
        const int x1 = std::min({x + int(TileSize), int(width_), viewport_[2]}), y1 = std::min({y + int(TileSize), int(height_), viewport_[3]});
        return x1 > x0 && y1 > y0 ? U32(x1 - x0) * U32(y1 - y0) : 0u;
    }
    bool empty() const {
        for (U32 i = 0; i < cells_.size(); ++i)
            if (remaining(i)) return false;
        return true;
    }

    // Claims up to n chunks of one cell for a device, none past what its frontier allows it. Returns how many were
    // claimed (none once the grid is closed).
    U32 claim(U32 index, U32 n, Chunk* out, Device device = Device::Cpu) {
        if (closed()) return 0;
        const U32 got = cells_[index].queue.claim(n, out, limit(index, device));
        if (got && !ranges_.empty()) refreshFrontiers();
        return got;
    }
    // The chunks of a cell a worker of a device may claim now.
    U32 claimable(U32 index, Device device = Device::Cpu) const {
        if (closed()) return 0;
        const ChunkQueue& q = cells_[index].queue;
        const U32 end = std::min(q.published(), limit(index, device)), taken = q.taken();
        return taken >= end ? 0 : end - taken;
    }
    // The progressive order's frontier of a device: the least-sampled range among the cells it works without a right
    // of passage.
    U32 frontier(Device device) const { return frontier_[U32(device)].load(std::memory_order_acquire); }
    const std::vector<Range>& ranges() const { return ranges_; }

    // Important parts first: each cell's score (below zero: not known yet). The cells above the mean of those
    // known are important: their owner's, and allowed a range ahead of the frontier; the rest are open to anyone.
    void setScores(const std::vector<float>& scores) {
        if (scores.size() != cells_.size()) return;
        double sum = 0;
        U32 known = 0;
        for (float s : scores)
            if (s >= 0) {
                sum += s;
                ++known;
            }
        if (!known) return;
        const float mean = float(sum / known);
        for (size_t i = 0; i < cells_.size(); ++i) {
            cells_[i].score.store(bits(std::max(scores[i], 0.0f)), std::memory_order_relaxed);
            cells_[i].important.store(scores[i] > mean, std::memory_order_relaxed);
        }
        scored_.store(true, std::memory_order_release);
        refreshFrontiers();
    }
    bool scored() const { return scored_.load(std::memory_order_acquire); }
    bool important(U32 index) const { return cells_[index].important.load(std::memory_order_relaxed); }
    // A cell its owner works first: every cell until the picture is scored, then the important ones.
    bool reserved(U32 index) const { return !scored() || important(index); }
    // The scene indexer's probabilities (device_suit.h), each cell's.
    void setSuitability(const std::vector<float>& p) {
        if (p.size() == cells_.size())
            for (size_t i = 0; i < p.size(); ++i) cells_[i].gpu = p[i];
    }

    // What the coordinator measures, for the rights of passage: the GPU's chunk items a second, one CPU worker's
    // samples a second, the CPU's workers, a GPU launch's target seconds; and each claim the GPU's broker makes.
    void setRates(double gpuItems, double cpuWorker, unsigned cpuWorkers, double window) {
        gpuItems_.store(bits(gpuItems), std::memory_order_relaxed);
        cpuWorker_.store(bits(cpuWorker), std::memory_order_relaxed);
        cpuWorkers_.store(cpuWorkers, std::memory_order_relaxed);
        window_.store(bits(window), std::memory_order_relaxed);
    }
    void gpuClaimed() { gpuClaim_.store(nowNs(), std::memory_order_relaxed); }

    // A right of passage: whether a device may work in a cell the other one has reserved. Before anything is
    // measured, yes. The CPU may when the GPU has not claimed for a while (a latency gap: its launches are slow to
    // come back), or when one CPU chunk ends well before the GPU would get through the work it has reserved, so the
    // chunk never holds the frame up. The GPU may when the CPU's reserved work outlasts a few of its launches, in a
    // cell where it would not defer most samples to the CPU anyway.
    bool passage(Device device, U32 index) const {
        const double gpu = number(gpuItems_.load(std::memory_order_relaxed)), cpu = number(cpuWorker_.load(std::memory_order_relaxed));
        if (gpu <= 0 || cpu <= 0) return true;
        const double chunk = double(options_.samplesPerChunk) * TilePixels, window = number(window_.load(std::memory_order_relaxed));
        if (device == Device::Cpu) {
            const long long last = gpuClaim_.load(std::memory_order_relaxed);
            if (last && double(nowNs() - last) * 1e-9 > std::max(4 * window, 0.05)) return true;
            return chunk / cpu < 0.5 * double(reservedChunks(Device::Gpu)) * chunk / gpu;
        }
        if (cells_[index].gpu < 0.2f) return false;
        const unsigned workers = std::max(1u, cpuWorkers_.load(std::memory_order_relaxed));
        return double(reservedChunks(Device::Cpu)) * chunk / (cpu * workers) > 4 * window;
    }
    // A drain: no chunk is handed out any more, and the grid counts as empty; what was claimed finishes.
    void close() { closed_.store(true, std::memory_order_release); }
    bool closed() const { return closed_.load(std::memory_order_acquire); }

    void rendered(U32 index, Device device, U64 samples) {
        cells_[index].samples[U32(device)].fetch_add(samples, std::memory_order_relaxed);
    }

    // The decision loop of a worker of either device (CA rules, spec §8): the cell to work in next, or ~0u when it
    // may take nothing now. Finish the cell it is in; then its device's reserved cells (by pressure, its unclaimed
    // work), then the open ones (the most important first), then the other device's reserved ones by a right of
    // passage; in each, a neighbour before the best anywhere.
    U32 choose(Device device, U32 current, U32 salt) const {
        auto tier = [&](U32 i) { return !reserved(i) ? 1 : owner(i) == device ? 0 : 2; };
        auto claimable = [&](U32 i) { return this->claimable(i, device); };
        // The CPU's right of passage is the same in every cell: worked out once a call.
        int cpuPassage = -1;
        auto passes = [&](U32 i) {
            if (device == Device::Gpu) return passage(device, i);
            if (cpuPassage < 0) cpuPassage = passage(device, i) ? 1 : 0;
            return cpuPassage == 1;
        };
        if (current < cells_.size() && claimable(current) && (tier(current) != 2 || passes(current))) return current;  // Finish
        for (int t = 0; t < 3; ++t) {
            U32 best = ~0u;
            float bestKey = 0;
            auto consider = [&](U32 index) {
                if (tier(index) != t) return;
                const U32 work = claimable(index);
                if (!work || (t == 2 && !passes(index))) return;
                const float key = t == 1 ? number(cells_[index].score.load(std::memory_order_relaxed)) + 1e-9f * float(work) : float(work);
                // Ties break by a hash of (cell, salt), so workers spread out (CA-07).
                if (best == ~0u || key > bestKey || (key == bestKey && hash(index, salt) > hash(best, salt))) {
                    best = index;
                    bestKey = key;
                }
            };
            if (current < cells_.size()) {
                forNeighbours(current, consider);
                if (best != ~0u) return best;
            }
            for (U32 i = 0; i < cells_.size(); ++i) consider(i);
            if (best != ~0u) return best;
        }
        return ~0u;
    }

    // Starting cell for the w-th of n workers on a device: spread over its cells.
    U32 seed(Device device, U32 worker, U32 workers) const {
        std::vector<U32> own;
        for (U32 i = 0; i < cells_.size(); ++i)
            if (owner(i) == device) own.push_back(i);
        if (own.empty()) return choose(device, ~0u, worker);
        return own[(size_t(worker) * own.size()) / std::max(1u, workers)];
    }

private:
    // The queue positions of a cell a device's claim may reach: the ranges up to its frontier, one more for an
    // important cell.
    U32 limit(U32 index, Device device) const {
        if (ranges_.empty()) return ~0u;
        const U64 lead = scored() && important(index) ? 1 : 0;
        const U64 upTo = std::min<U64>(U64(frontier(device)) + lead + 1, ranges_.size());
        return U32(std::min<U64>(upTo * cells_[index].tiles, ~0u));
    }
    // Each device's frontier, from where every cell's queue stands: the least range still to hand out among its own
    // reserved cells and the open ones (where none is left there, among all).
    void refreshFrontiers() {
        if (ranges_.empty()) return;
        U32 least[3] = {~0u, ~0u, ~0u};  // the CPU's, the GPU's, every cell's
        for (U32 i = 0; i < cells_.size(); ++i) {
            const Cell& c = cells_[i];
            if (!c.tiles || !c.queue.remaining()) continue;
            const U32 at = c.queue.taken() / c.tiles;
            const bool open = !reserved(i);
            for (U32 dev = 0; dev < 2; ++dev)
                if (open || U32(c.owner) == dev) least[dev] = std::min(least[dev], at);
            least[2] = std::min(least[2], at);
        }
        for (U32 dev = 0; dev < 2; ++dev)
            frontier_[dev].store(std::min<U32>(least[dev] != ~0u ? least[dev] : least[2], U32(ranges_.size())), std::memory_order_release);
    }
    // The unclaimed chunks of the cells a device has reserved.
    U64 reservedChunks(Device device) const {
        U64 n = 0;
        for (U32 i = 0; i < cells_.size(); ++i)
            if (owner(i) == device && reserved(i)) n += remaining(i);
        return n;
    }
    // Floats and doubles kept as their bits in the integer atomics (scene_rdl2's AtomicFloat.h has std::atomic's own).
    static U32 bits(float v) {
        U32 b;
        std::memcpy(&b, &v, sizeof b);
        return b;
    }
    static U64 bits(double v) {
        U64 b;
        std::memcpy(&b, &v, sizeof b);
        return b;
    }
    static float number(U32 b) {
        float v;
        std::memcpy(&v, &b, sizeof v);
        return v;
    }
    static double number(U64 b) {
        double v;
        std::memcpy(&v, &b, sizeof v);
        return v;
    }
    static long long nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    template <class F>
    void forNeighbours(U32 index, F f) const {
        const int x = int(index % cellsX_), y = int(index / cellsX_);
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                if ((dx || dy) && nx >= 0 && ny >= 0 && nx < int(cellsX_) && ny < int(cellsY_))
                    f(U32(ny) * cellsX_ + U32(nx));
            }
    }
    static U32 hash(U32 a, U32 b) {
        U32 h = a * 0x9e3779b1u ^ (b + 0x7f4a7c15u + (a << 6) + (a >> 2));
        h ^= h >> 16;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        return h;
    }

    U32 width_, height_, spp_, cellsX_ = 0, cellsY_ = 0;
    int viewport_[4] = {0, 0, 0x7fffffff, 0x7fffffff};
    GridOptions options_;
    std::vector<Cell> cells_;
    std::atomic<U64> total_{0};
    std::atomic<bool> closed_{false};
    std::vector<Range> ranges_;  // a uniform frame's sample ranges (none: adaptive)
    std::atomic<U32> frontier_[2] = {};
    std::atomic<bool> scored_{false};
    std::atomic<U64> gpuItems_{0}, cpuWorker_{0}, window_{bits(0.01)};
    std::atomic<unsigned> cpuWorkers_{1};
    std::atomic<long long> gpuClaim_{0};
};

// Cell ownership that persists from frame to frame (GRD-03) and flips towards
// the device that rendered most of a cell (CA-10), with the supervisor's
// limits: at most 8 flips a frame, a 16-frame cooldown, and a 64-frame freeze
// after thrashing (CA-12).
class Ownership {
public:
    // The owners for a new frame of a grid with `cells` cells; empty for the
    // checkerboard.
    const std::vector<Device>* owners(U32 cells) const {
        return owners_.size() == cells ? &owners_ : nullptr;
    }

    // Records a finished frame and decides the flips for the next one.
    void finished(const Grid& grid) {
        const U32 n = grid.cellCount();
        if (owners_.size() != n) {
            owners_.resize(n);
            history_.assign(size_t(n) * Window, 0.f);
            lastFlip_.assign(n, -1000000);
            for (U32 i = 0; i < n; ++i) owners_[i] = grid.owner(i);
        }
        for (U32 i = 0; i < n; ++i) {
            const double cpu = double(grid.cell(i).samples[0].load()), gpu = double(grid.cell(i).samples[1].load());
            history_[size_t(i) * Window + frame_ % Window] = cpu + gpu > 0 ? float(gpu / (cpu + gpu)) : 0.5f;
        }
        ++frame_;
        if (frame_ < frozenUntil_ || frame_ < Window) return;
        struct Candidate { U32 cell; float share; };
        std::vector<Candidate> flips;
        for (U32 i = 0; i < n; ++i) {
            if (frame_ - lastFlip_[i] < 16) continue;  // cooldown
            float gpuShare = 0;
            for (U32 k = 0; k < Window; ++k) gpuShare += history_[size_t(i) * Window + k];
            gpuShare /= Window;
            const float other = owners_[i] == Device::Cpu ? gpuShare : 1 - gpuShare;
            if (other > 0.6f) flips.push_back({i, other});
        }
        std::sort(flips.begin(), flips.end(), [](auto a, auto b) { return a.share > b.share; });
        // At most 8 flips a frame (CA-10), and never more than 10% of the cells, so
        // the first thrash rule of CA-12 holds by construction; the second (a cell
        // flipping back within 32 frames) freezes ownership for 64 frames.
        const size_t limit = std::min<size_t>(8, std::max<size_t>(1, n / 10));
        for (size_t k = 0; k < flips.size() && k < limit; ++k) {
            const U32 i = flips[k].cell;
            if (frame_ - lastFlip_[i] < 32) {
                frozenUntil_ = frame_ + 64;
                return;
            }
            owners_[i] = owners_[i] == Device::Cpu ? Device::Gpu : Device::Cpu;
            lastFlip_[i] = frame_;
        }
    }

private:
    static constexpr U32 Window = 8;
    std::vector<Device> owners_;
    std::vector<float> history_;
    std::vector<long long> lastFlip_;
    long long frame_ = 0, frozenUntil_ = 0;
};

}  // namespace xpu2::interactive::async
