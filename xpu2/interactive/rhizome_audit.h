// Rhizome's audit (RHIZOME.md, 5): the master renders again a few pixels of each contribution and compares their sums. A
// pixel's sum over a sample range is the same on any machine (every sample is defined by its identity), so a node
// that sends a wrong film is caught at the rate pixels are checked. The pixels are chosen after the contribution has
// arrived, by a keyed hash under a seed only the master knows, so a node cannot tell which of its pixels will be
// checked. A node's first contribution has a whole row checked as well.
#pragma once
#include "rhizome_identity.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace xpu2::interactive::rhizome {

// A film for a cell's sample range: its rectangle (half open), the samples [first, first + count), and per pixel
// four floats of sums (RGB and alpha, the film's order: rows from the top) and a count.
struct Film {
    uint32_t frame = 0;  // the frame it was rendered for (FrameShape::frame)
    uint32_t rect[4] = {0, 0, 0, 0};
    uint32_t first = 0, count = 0;
    std::vector<float> sums;
    std::vector<uint32_t> counts;
    // The renderer's other outputs, per pixel: the odd-index samples' sums (four floats; the beauty's aux outputs),
    // and the AOV accumulators before resolving (its schema's channels) with their closest depths, merged at the
    // master by each output's filter. Empty when the scene has none.
    std::vector<float> odd;
    std::vector<double> aovs;
    std::vector<float> aovDepths;
    // Cryptomatte's fragments of the film's pixels as sums, before finalizing (the caller's format; empty without).
    Bytes crypto;
    // Its render waited for pieces from the swarm's memory (a texture's tiles, plan 16): its time says nothing of the
    // machine's rate, and is not sent (not on the wire).
    bool waited = false;
    uint32_t width() const { return rect[2] - rect[0]; }
    uint32_t height() const { return rect[3] - rect[1]; }
};

// One pixel's adaptive error from its sums, odd-index sums and count, as XPU2 works it out on either device
// (async_integrator.h pixelError, MoonRay's estimatePixelErrorInternal): the luma of the even and odd halves'
// difference over the square root of the pixel's luma, plus the alpha difference; -1 until both halves have a sample.
inline float pixelError(const double total[4], const double odd[4], uint32_t samples) {
    const double n = samples, nOdd = samples / 2, nEven = n - nOdd;
    if (nOdd <= 0 || nEven <= 0) return -1.0f;
    double diff[4], avg[3];
    for (int c = 0; c < 4; ++c) {
        if (total[c] != total[c] || odd[c] != odd[c]) return 0.0f;
        diff[c] = std::fabs((total[c] * nOdd - odd[c] * n) / (nOdd * nEven));
        if (c < 3) avg[c] = total[c] / n;
    }
    const double lumDiff = 0.299 * diff[0] + 0.587 * diff[1] + 0.114 * diff[2];
    const double lumAvg = 0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2];
    if (lumAvg <= 0) return float(diff[3]);
    return float(lumDiff / std::sqrt(lumAvg) + diff[3]);
}

// What renders a pixel's samples again: its sums for samples [first, first + count).
using PixelRenderer = std::function<void(uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4])>;

struct Verdict {
    bool passed = true;
    uint32_t checked = 0, failed = 0;
    double worst = 0;  // the largest relative difference seen
    double bias = 0;   // the mean of the signed relative differences
    std::string detail;  // the first pixels that differ: where, the node's sums and count, the master's
};

class Auditor {
public:
    double rate = 1.0 / 1000;  // pixels checked a pixel sent
    double pixelLimit = 0.05;   // a pixel's relative difference past which it is an outlier (honest devices: under 1%)
    double outlierShare = 0.05; // the share of pixels checked that may be outliers (paths that split between devices)
    double biasLimit = 0.002;   // the mean signed difference allowed, at least (four standard errors where wider)
    double floor = 1e-4;        // absolute, per sample and channel, near black

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
    // What the audit has seen of a node so far (every film's picks): pixels checked, outliers, and the signed
    // differences of the pixels within pixelLimit (their sum and sum of squares).
    struct Tally {
        uint64_t checked = 0, outliers = 0, inliers = 0;
        double sum = 0, squares = 0, worst = 0;
        double bias() const { return inliers ? sum / double(inliers) : 0.0; }
        double sd() const {
            if (inliers < 2) return 0.0;
            const double m = bias();
            return std::sqrt(std::max(0.0, (squares - double(inliers) * m * m) / double(inliers - 1)));
        }
    };
    Tally tally(const Hash& node) const {
        const auto it = tallies_.find(node);
        return it == tallies_.end() ? Tally{} : it->second;
    }
    // The outliers an honest node may have among n pixels checked: outlierShare of them, four standard deviations
    // more, and one (a node's first films: one path that split is not a wrong film).
    double outliersAllowed(uint64_t n) const {
        const double p = outlierShare, k = double(n);
        return p * k + 4.0 * std::sqrt(p * (1.0 - p) * k) + 1.0;
    }
    // A line for the log: each node's audit so far.
    std::string summary() const {
        std::string s;
        for (const auto& [h, t] : tallies_) {
            char line[200];
            std::snprintf(line, sizeof line, "%s%s %llu pixels, %llu outliers (%.2f%%), bias %+.4f%%, worst %.2f%%", s.empty() ? "" : "; ",
                          hex(h.data(), 6).c_str(), (unsigned long long)t.checked, (unsigned long long)t.outliers,
                          t.checked ? 100.0 * double(t.outliers) / double(t.checked) : 0.0, t.bias() * 100.0, t.worst * 100.0);
            s += line;
        }
        return s;
    }

    // Checks a film from a node by rendering its picks again. A processor and a graphics card give a sample a little
    // differently (a path through a volume or a glass can turn another way on a last bit), so a pixel's sum differs
    // by a few tenths of a percent either way between devices, and now and then (a path that split, of the few
    // samples a film has) by far more; what a wrong film changes is checked instead, on all of the node's pixels
    // checked so far (a film's few picks alone cannot tell a split path from a wrong film):
    //   counts: each pixel has exactly the film's samples, none of them NaN or infinite (at once);
    //   outliers (a pixel off by more than pixelLimit): no more than outliersAllowed of the node's pixels checked;
    //   no bias: the mean of the signed differences of the pixels within pixelLimit sits within biasLimit, or four
    //   standard errors where they are few or wide (device noise averages out, a film made brighter or darker does not;
    //   one changed by more than pixelLimit is all outliers).
    Verdict check(const Hash& node, const Film& f, const PixelRenderer& render) {
        Verdict v;
        if (f.sums.size() != size_t(f.width()) * f.height() * 4 || f.counts.size() != size_t(f.width()) * f.height()) {
            v.passed = false;
            v.detail = " its film is not its rectangle's size;";
            return v;
        }
        Tally t = tally(node);
        bool countsWrong = false;
        auto note = [&](uint32_t x, uint32_t y, size_t i, const float* ref) {
            if (++v.failed > 4) return;
            char s[200];
            std::snprintf(s, sizeof s, " (%u,%u) node %.5g %.5g %.5g %.5g n%u, master %.5g %.5g %.5g %.5g;", x, y, f.sums[i * 4], f.sums[i * 4 + 1],
                          f.sums[i * 4 + 2], f.sums[i * 4 + 3], f.counts[i], ref[0], ref[1], ref[2], ref[3]);
            v.detail += s;
        };
        for (const auto& [x, y] : picks(node, f)) {
            const size_t i = size_t(y - f.rect[1]) * f.width() + (x - f.rect[0]);
            float ref[4] = {0, 0, 0, 0};
            render(x, y, f.first, f.count, ref);
            ++v.checked;
            if (f.counts[i] != f.count) {
                countsWrong = true;
                note(x, y, i, ref);
                continue;
            }
            double a = 0, b = 0;
            bool finite = true;
            for (int c = 0; c < 4; ++c) {
                a += f.sums[i * 4 + c], b += ref[c];
                finite = finite && std::isfinite(f.sums[i * 4 + c]);
            }
            const double scale = std::max(std::fabs(b), floor * f.count * 4), rel = (a - b) / scale;
            if (!finite || !std::isfinite(rel)) {
                countsWrong = true;  // a NaN or infinite sum the master's render does not have
                note(x, y, i, ref);
                continue;
            }
            ++t.checked;
            v.worst = std::max(v.worst, std::fabs(rel));
            t.worst = std::max(t.worst, std::fabs(rel));
            if (std::fabs(rel) > pixelLimit) {
                ++t.outliers;
                note(x, y, i, ref);
            } else {
                ++t.inliers;
                t.sum += rel, t.squares += rel * rel;
            }
        }
        tallies_[node] = t;
        v.bias = t.bias();
        const double allowed = std::max(biasLimit, 4.0 * t.sd() / std::sqrt(double(std::max<uint64_t>(1, t.inliers))));
        const bool biased = t.inliers > 0 && std::fabs(t.bias()) > allowed;
        const bool tooMany = double(t.outliers) > outliersAllowed(t.checked);
        char s[200];
        if (biased) {
            std::snprintf(s, sizeof s, " a bias of %.3f%% (at most %.3f%%);", t.bias() * 100.0, allowed * 100.0);
            v.detail += s;
        }
        if (tooMany || (v.failed && !countsWrong)) {
            std::snprintf(s, sizeof s, " %llu outliers of %llu pixels checked so far (at most %.1f);", (unsigned long long)t.outliers,
                          (unsigned long long)t.checked, outliersAllowed(t.checked));
            v.detail += s;
        }
        v.passed = !countsWrong && !biased && !tooMany;
        return v;
    }

private:
    Hash seed_{};
    std::set<Hash> audited_;
    std::map<Hash, Tally> tallies_;
};

}  // namespace xpu2::interactive::rhizome
