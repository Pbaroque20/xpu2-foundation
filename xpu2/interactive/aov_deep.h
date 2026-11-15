// Deep output as MoonRay keeps it (pbr/core/DeepBuffer.cc, PathIntegrator.cc's deep camera ray): each sample's camera
// ray, where it meets a surface, adds its pixel's hard surface segment of the same deep IDs, a normal within
// deep_curvature_tolerance of the segment's first and a depth within half of deep_z_tolerance of its first; else it
// starts a segment. A segment keeps its nearest and farthest distances, an 8x8 subpixel mask of the samples' positions
// (duplicated to 4x4, 2x2 or 1x1 below 64, 16 and 4 samples a pixel), and the sums of their alpha, beauty and count.
// The CPU lane adds to the host's DeepFilm; the GPU to lists of its own a pixel (async_gpu.cu mergeDeep, the words
// below), which join the host's at the frame's end segment by segment by the same rule. Hard surfaces only: a scene
// whose camera rays reach a volume is left to MoonRay (its deep volume segments are not ported).
#pragma once
#include "aov_state.h"
#if !defined(__CUDACC_RTC__)
#include <algorithm>
#include <cmath>
#include <vector>
#endif

namespace xpu2::interactive::aov {

constexpr U32 DeepMaxIds = 3;

// A sample's camera ray at its first surface (deepParams' mHitGeom, the hard surface's data): its distance, the ray's
// render-space z (t to z), the geometric normal in render space, the deep IDs, and the sample's position in its pixel.
struct DeepHit {
    U32 hit = 0;
    float t = 0.0f, rayZ = 0.0f;
    V3 normal{0, 0, 0};
    float ids[DeepMaxIds] = {0, 0, 0};
    float sx = 0.0f, sy = 0.0f;
};

// The subpixel mask a sample sets: its 8x8 subpixel's bit, or the block about it at a coarser subpixel resolution
// (DeepBuffer::addSample's duplication; res 8, 4, 2 or 1).
X2_HD inline void deepMask(float sx, float sy, U32 res, U32& lo, U32& hi) {
    U32 x = U32(sx * 8.0f), y = U32(sy * 8.0f);
    x = x > 7u ? 7u : x;
    y = y > 7u ? 7u : y;
    const U32 span = res >= 8 ? 1u : res >= 4 ? 2u : res >= 2 ? 4u : 8u;
    const U32 x0 = x & ~(span - 1u), y0 = y & ~(span - 1u);
    lo = hi = 0;
    for (U32 yy = y0; yy < y0 + span; ++yy)
        for (U32 xx = x0; xx < x0 + span; ++xx) {
            const U32 bit = yy * 8u + xx;
            if (bit < 32u) lo |= 1u << bit;
            else hi |= 1u << (bit - 32u);
        }
}
// The subpixel resolution for a frame's samples a pixel (RenderFramePasses.cc).
X2_HD inline U32 deepSubpixelRes(U32 spp) { return spp >= 64 ? 8u : spp >= 16 ? 4u : spp >= 4 ? 2u : 1u; }

// Whether a sample (or a segment's first sample) joins a segment: the same IDs, the normals within the curvature
// tolerance's cosine, the distance within half the z tolerance of the segment's first.
X2_HD inline bool deepJoins(const float* ids, const float* segIds, U32 idCount, V3 n, V3 segN, float t, float segFirst, float segRayZ,
                            float cosTolerance, float zTolerance) {
    for (U32 i = 0; i < idCount; ++i)
        if (ids[i] != segIds[i]) return false;
    return n.x * segN.x + n.y * segN.y + n.z * segN.z > cosTolerance && fabsf(segRayZ * (segFirst - t)) < zTolerance * 0.5f;
}

// A segment on the card (async_gpu.cu): words in this order.
enum DeepWord : U32 {
    DeepNext = 0, DeepId0 = 1, DeepFirst = DeepId0 + DeepMaxIds, DeepRayZ, DeepNx, DeepNy, DeepNz, DeepFront, DeepBack, DeepMaskLo,
    DeepMaskHi, DeepAlpha, DeepWeight, DeepR, DeepG, DeepB, DeepWords
};

#ifndef __CUDACC__
// The samples a pixel the subpixel resolution goes by (FrameState's mMaxSamplesPerPixel): an adaptive frame's most, a
// uniform one's, or the pixel sample map's largest.
inline U32 deepFrameSpp(bool adaptive, unsigned maxSamples, unsigned spp, const std::vector<unsigned>& pixelSamples) {
    U32 s = adaptive ? U32(maxSamples) : U32(spp);
    for (unsigned k : pixelSamples) s = std::max<U32>(s, U32(k));
    return s;
}
// The host's deep film: each pixel's hard surface segments (DeepBuffer::HardSurfaceSegment), beauty and alpha.
class DeepFilm {
public:
    struct Segment {
        float ids[DeepMaxIds] = {0, 0, 0};
        float first = 0, rayZ = 0, front = 0, back = 0;
        V3 normal{0, 0, 0};
        U32 maskLo = 0, maskHi = 0;
        float alpha = 0, weight = 0;
        float rgb[3] = {0, 0, 0};
    };
    void init(U32 width, U32 height, U32 idCount, float curvatureDegrees, float zTolerance, U32 spp) {
        width_ = width, height_ = height;
        idCount_ = std::min(idCount, DeepMaxIds);
        cos_ = float(std::cos(double(curvatureDegrees) * 3.14159265358979323846 / 180.0));
        z_ = zTolerance;
        res_ = deepSubpixelRes(spp);
        pixels_.assign(size_t(width) * height, {});
    }
    bool empty() const { return pixels_.empty(); }
    U32 width() const { return width_; }
    U32 height() const { return height_; }
    U32 idCount() const { return idCount_; }
    float cosTolerance() const { return cos_; }
    float zTolerance() const { return z_; }
    U32 subpixelRes() const { return res_; }
    const std::vector<Segment>& segments(size_t pixel) const { return pixels_[pixel]; }
    // One sample (addSample8x8, its duplicates at once). The caller holds the pixel's lock (HostFilm's tile lock).
    void add(size_t pixel, const DeepHit& h, const double* rgb, double alpha) {
        if (!h.hit || pixel >= pixels_.size()) return;
        U32 lo, hi;
        deepMask(h.sx, h.sy, res_, lo, hi);
        V3 n = h.normal;
        const float l = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (l > 0) n = V3{n.x / l, n.y / l, n.z / l};
        for (Segment& s : pixels_[pixel])
            if (deepJoins(h.ids, s.ids, idCount_, n, s.normal, h.t, s.first, s.rayZ, cos_, z_)) {
                s.front = std::min(s.front, h.t);
                s.back = std::max(s.back, h.t);
                s.maskLo |= lo, s.maskHi |= hi;
                s.alpha += float(alpha);
                for (int c = 0; c < 3; ++c) s.rgb[c] += float(rgb[c]);
                s.weight += 1.0f;
                return;
            }
        Segment s;
        for (U32 i = 0; i < DeepMaxIds; ++i) s.ids[i] = h.ids[i];
        s.first = s.front = s.back = h.t;
        s.rayZ = h.rayZ;
        s.normal = n;
        s.maskLo = lo, s.maskHi = hi;
        s.alpha = float(alpha);
        for (int c = 0; c < 3; ++c) s.rgb[c] = float(rgb[c]);
        s.weight = 1.0f;
        pixels_[pixel].push_back(s);
    }
    // A segment made elsewhere (the card's, a share's): into the one it joins by its first sample, else as it is.
    void merge(size_t pixel, const Segment& g) {
        if (pixel >= pixels_.size() || !(g.weight > 0)) return;
        for (Segment& s : pixels_[pixel])
            if (deepJoins(g.ids, s.ids, idCount_, g.normal, s.normal, g.first, s.first, s.rayZ, cos_, z_)) {
                s.front = std::min(s.front, g.front);
                s.back = std::max(s.back, g.back);
                s.maskLo |= g.maskLo, s.maskHi |= g.maskHi;
                s.alpha += g.alpha;
                for (int c = 0; c < 3; ++c) s.rgb[c] += g.rgb[c];
                s.weight += g.weight;
                return;
            }
        pixels_[pixel].push_back(g);
    }

private:
    U32 width_ = 0, height_ = 0, idCount_ = 0, res_ = 8;
    float cos_ = 0.7071f, z_ = 2.0f;
    std::vector<std::vector<Segment>> pixels_;
};
#endif

}  // namespace xpu2::interactive::aov
