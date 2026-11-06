// Hive's audit (HIVE.md, 5): the master renders again a few pixels of each contribution and compares their sums. A
// pixel's sum over a sample range is the same on any machine (every sample is defined by its identity), so a node
// that sends a wrong film is caught at the rate pixels are checked. The pixels are chosen after the contribution has
// arrived, by a keyed hash under a seed only the master knows, so a node cannot tell which of its pixels will be
// checked. A node's first contribution has a whole row checked as well.
#pragma once
#include "hive_identity.h"
#include <cmath>
#include <functional>
#include <set>
#include <vector>

namespace xpu2::interactive::hive {

// A film for a cell's sample range: its rectangle (half open), the samples [first, first + count), and per pixel
// four floats of sums (RGB and alpha, the film's order: rows from the top) and a count.
struct Film {
    uint32_t rect[4] = {0, 0, 0, 0};
    uint32_t first = 0, count = 0;
    std::vector<float> sums;
    std::vector<uint32_t> counts;
    uint32_t width() const { return rect[2] - rect[0]; }
    uint32_t height() const { return rect[3] - rect[1]; }
};

// What renders a pixel's samples again: its sums for samples [first, first + count).
using PixelRenderer = std::function<void(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4])>;

struct Verdict {
    bool passed = true;
    uint32_t checked = 0, failed = 0;
    double worst = 0;  // the largest relative difference seen
};

class Auditor {
public:
    double rate = 1.0 / 1000;  // pixels checked a pixel sent
    double tolerance = 2e-3;   // relative, for floating point across processors and cards (sums in another order)
    double floor = 1e-4;       // absolute, per sample, near black

    Auditor() { randomBytes(seed_.data(), seed_.size()); }
    explicit Auditor(const Hash& seed) : seed_(seed) {}

    // The pixels to check of a film from a node: at least one, about rate of its pixels, plus a whole row of a
    // node's first film.
    std::vector<std::pair<uint32_t, uint32_t>> picks(const Hash& node, const Film& f) {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        const uint32_t w = f.width(), h = f.height();
        if (!w || !h) return out;
        const uint64_t pixels = uint64_t(w) * h;
        const uint32_t n = uint32_t(std::max<double>(1.0, std::ceil(double(pixels) * rate)));
        std::set<uint64_t> seen;
        for (uint32_t k = 0; k < n * 4 && out.size() < n; ++k) {
            unsigned char m[32 + 4 * 4];
            std::memcpy(m, node.data(), 32);
            const uint32_t parts[4] = {f.rect[0] | (f.rect[1] << 16), f.first, f.count, k};
            std::memcpy(m + 32, parts, sizeof parts);
            const Hash d = sha256(m, sizeof m, seed_.data(), seed_.size());
            uint64_t v;
            std::memcpy(&v, d.data(), 8);
            const uint64_t at = v % pixels;
            if (!seen.insert(at).second) continue;
            out.push_back({f.rect[0] + uint32_t(at % w), f.rect[1] + uint32_t(at / w)});
        }
        if (!audited_.count(node)) {
            audited_.insert(node);
            const uint32_t row = f.rect[1] + h / 2;
            for (uint32_t x = f.rect[0]; x < f.rect[2]; ++x)
                if (!seen.count(uint64_t(row - f.rect[1]) * w + (x - f.rect[0]))) out.push_back({x, row});
        }
        return out;
    }
    // Checks a film from a node by rendering its picks again.
    Verdict check(const Hash& node, const Film& f, const PixelRenderer& render) {
        Verdict v;
        if (f.sums.size() != size_t(f.width()) * f.height() * 4 || f.counts.size() != size_t(f.width()) * f.height()) {
            v.passed = false;
            return v;
        }
        for (const auto& [x, y] : picks(node, f)) {
            const size_t i = size_t(y - f.rect[1]) * f.width() + (x - f.rect[0]);
            float ref[4] = {0, 0, 0, 0};
            render(x, y, f.first, f.count, ref);
            ++v.checked;
            bool bad = f.counts[i] != f.count;
            for (int c = 0; c < 4; ++c) {
                const double a = f.sums[i * 4 + c], b = ref[c];
                const double d = std::fabs(a - b), scale = std::max(std::fabs(b), floor * f.count);
                v.worst = std::max(v.worst, d / scale);
                if (!(d <= tolerance * scale)) bad = true;  // NaN fails
            }
            if (bad) ++v.failed;
        }
        v.passed = v.failed == 0;
        return v;
    }

private:
    Hash seed_{};
    std::set<Hash> audited_;
};

}  // namespace xpu2::interactive::hive
