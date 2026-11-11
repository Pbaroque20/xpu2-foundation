// Host side of MoonRay's light image distributions (light_image.h): reading a light's texture with all its mip
// levels, colour correcting it, and tabulating the distributions as ImageDistribution::init does, quirks included:
// each level's weights are read from level 0 at that level's texel centres, its size is the base size shifted
// (whatever size the file gives an odd level), and a texture of fewer than three channels fills only those.
//
// Adapted from MoonRay pbr/core/Distribution.cc and ImageColorCorrect.ispc, Apache-2.0, copyright 2023-2024
// DreamWorks Animation LLC; see MOONRAY_LICENSE.txt.
#pragma once
#include "light_image.h"
#include "shade_colorcorrect.h"
#include "remote_files.h"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace xpu2::interactive::lights {

// Distribution2D's mappings.
enum Mapping : U32 { MapPlanar = 0, MapCircular = 1, MapHemispherical = 2, MapSpherical = 3 };

// A light texture's settings, as the rdl2 Light holds them.
struct ImageSettings {
    std::string file;
    Mapping mapping = MapPlanar;
    float gamma[3] = {1, 1, 1}, contrast[3] = {1, 1, 1}, saturation[3] = {1, 1, 1}, gain[3] = {1, 1, 1},
          offset[3] = {0, 0, 0}, temperature[3] = {0, 0, 0};
    float rotation = 0, translation[2] = {0, 0}, coverage[2] = {1, 1}, repsU = 1, repsV = 1;
    bool mirrorU = false, mirrorV = false;
    float border[3] = {1, 1, 1};
    std::string key() const {
        std::string k = file + "|" + std::to_string(int(mapping));
        for (const float* a : {gamma, contrast, saturation, gain, offset, temperature, translation, coverage, border})
            for (int i = 0; i < (a == translation || a == coverage ? 2 : 3); ++i) k += "|" + std::to_string(a[i]);
        return k + "|" + std::to_string(rotation) + "|" + std::to_string(repsU) + "|" + std::to_string(repsV) + "|" +
               std::to_string(int(mirrorU)) + std::to_string(int(mirrorV));
    }
};

// A built distribution: its arrays on the host and the view over them.
struct ImageData {
    std::string key;
    std::vector<float> pixels, cdf;
    std::vector<U32> guide;
    std::vector<Dist1D> dists;
    std::vector<ImageLevel> levels;
    Image view{};
    void point() {
        view.pixels = pixels.data();
        view.cdf = cdf.data();
        view.guide = guide.data();
        view.dists = dists.data();
        view.levels = levels.data();
    }
};

namespace build {
// GuideDistribution1D::tabulateCdf over cdf[offset, offset + size) holding the weights.
inline float tabulate(std::vector<float>& cdfs, std::vector<U32>& guides, Dist1D& d) {
    float* cdf = cdfs.data() + d.offset;
    U32* guide = guides.data() + d.offset;
    const U32 n = d.size;
    d.invSize = 1.0f / float(n);
    float sum = 0.0f;
    for (U32 i = 0; i < n; i++) sum += cdf[i];
    if (sum != 0.0f) {
        float partial = 0.0f;
        for (U32 i = 0; i + 1 < n; i++) {
            partial += cdf[i];
            if (partial > sum) partial = sum;
            cdf[i] = partial;
        }
    } else {
        for (U32 i = 0; i + 1 < n; i++) cdf[i] = float(i + 1) * d.invSize;
        sum = 1.0f;
    }
    cdf[n - 1] = sum;
    d.totalWeight = sum;
    d.invTotalWeight = 1.0f / sum;
    const float invSizeGuide = 1.0f / float(n);
    U32 i = 0;
    for (U32 j = 0; j < n; ++j) {
        const float tau = float(j) * d.totalWeight * invSizeGuide;
        while (i < n - 1 && cdf[i] < tau) ++i;
        guide[j] = i;
    }
    d.thresholdLow = 0.5f * cdf[0] * d.invTotalWeight;
    d.thresholdHigh = n > 1 ? 0.5f * (cdf[n - 2] + cdf[n - 1]) * d.invTotalWeight : d.thresholdLow;
    d.linearCoeffLow = d.thresholdLow != 0.0f ? 0.5f * d.invSize / d.thresholdLow : 0.0f;
    d.linearCoeffHigh = d.thresholdHigh != 1.0f ? 0.5f * d.invSize / (1.0f - d.thresholdHigh) : 0.0f;
    return sum * d.invSize;
}
inline float luminance(V3 c) { return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z; }  // scene_rdl2 luminance
}  // namespace build

// Reads and tabulates a light's texture (ImageDistribution::init). Throws when the file cannot be read (MoonRay then
// lights with the fatal colour) and returns null when it holds no usable level (the light is off).
inline std::shared_ptr<ImageData> buildImage(const ImageSettings& s) {
    // From its source when it has one (remote_files.h: a Rhizome node's texture from the swarm's memory), else the disk.
    std::unique_ptr<xpu2::remote::Proxy> proxy;
    auto in = xpu2::remote::open(s.file, proxy);
    if (!in) throw std::runtime_error("Image file \"" + s.file + "\" does not exist.");
    OIIO::ImageSpec spec = in->spec();
    auto out = std::make_shared<ImageData>();
    out->key = s.key();
    int width = 0, height = 0, widthAbove = 0, heightAbove = 0;
    std::vector<float> input;
    std::vector<std::pair<U32, U32>> actual;  // each level's size as read
    for (int level = 0;; ++level) {
        if (!in->seek_subimage(0, level)) break;
        spec = in->spec();
        const int w = spec.width, h = spec.height;
        if (w < 1 || h < 1) break;
        if (w == 1 && h == 1) break;  // MoonRay: 1x1 mip levels are not supported
        if (level == 0) {
            width = w;
            height = h;
        } else {
            const bool wOk = w == widthAbove >> 1 || ((widthAbove & 1) && w == (widthAbove >> 1) + 1);
            const bool hOk = h == heightAbove >> 1 || ((heightAbove & 1) && h == (heightAbove >> 1) + 1);
            if (!wOk || !hOk) break;
        }
        widthAbove = w;
        heightAbove = h;
        const int nc = spec.nchannels;
        input.assign(size_t(w) * h * nc, 0.0f);
        if (!in->read_image(0, level, 0, -1, OIIO::TypeDesc::FLOAT, input.data()))
            throw std::runtime_error("Cannot read image file: \"" + s.file + "\" (" + in->geterror() + ")");
        // Flipped (MoonRay's frame buffers run from the bottom), three channels.
        const size_t base = out->pixels.size();
        out->pixels.resize(base + size_t(w) * h * 3, 0.0f);
        const int copy = nc < 3 ? nc : 3;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < copy; ++c)
                    out->pixels[base + (size_t(y) * w + x) * 3 + c] = input[(size_t(h - 1 - y) * w + x) * nc + c];
        out->levels.push_back({U32(width >> level), U32(height >> level), U32(base), 0});
        actual.push_back({U32(w), U32(h)});
    }
    in->close();
    if (out->levels.empty()) throw std::runtime_error("Cannot seek into image file: \"" + s.file + "\"");
    Image& im = out->view;
    im.levelCount = U32(out->levels.size());
    im.width = U32(width);
    im.height = U32(height);
    // The texture transform: translate(-t) * scale(reps / coverage) * rotate(angle), as row vectors.
    im.transformed = (s.rotation != 0.0f || s.translation[0] != 0.0f || s.translation[1] != 0.0f || s.coverage[0] != 1.0f ||
                      s.coverage[1] != 1.0f || s.repsU != 1.0f || s.repsV != 1.0f) ? 1u : 0u;
    if (im.transformed) {
        const float tx = -s.translation[0], ty = -s.translation[1];
        const float sx = s.repsU / s.coverage[0], sy = s.repsV / s.coverage[1];
        const float theta = s.rotation * (3.14159265358979323846f / 180.0f);
        const float c = std::cos(theta), sn = std::sin(theta);
        const float csx = c * sx, ssx = sn * sx, csy = c * sy, ssy = sn * sy;
        const float m[9] = {csx, ssx, 0.0f, -ssy, csy, 0.0f, csx * tx - ssy * ty, ssx * tx + csy * ty, 1.0f};
        for (int i = 0; i < 9; ++i) im.m[i] = m[i];
    } else {
        const float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        for (int i = 0; i < 9; ++i) im.m[i] = m[i];
    }
    im.repsU = s.repsU;
    im.repsV = s.repsV;
    im.mirrorU = s.mirrorU ? 1u : 0u;
    im.mirrorV = s.mirrorV ? 1u : 0u;
    for (int i = 0; i < 3; ++i) im.border[i] = s.border[i];

    // Colour correction, every level, when any setting is not its default (PBR_colorCorrectFormatArray).
    auto ne = [](const float* a, float v) { return a[0] != v || a[1] != v || a[2] != v; };
    const bool gammaOn = ne(s.gamma, 1), saturationOn = ne(s.saturation, 1), contrastOn = ne(s.contrast, 1),
               gainOffsetOn = ne(s.gain, 1) || ne(s.offset, 0), temperatureOn = ne(s.temperature, 0);
    if (gammaOn || saturationOn || contrastOn || gainOffsetOn || temperatureOn) {
        const float eps = 0.000001f;
        const float gx = 1.0f / std::max(eps, s.gamma[0]), gy = 1.0f / std::max(eps, s.gamma[1]), gz = 1.0f / std::max(eps, s.gamma[2]);
        for (size_t i = 0; i < out->pixels.size(); i += 3) {
            float r = out->pixels[i], g = out->pixels[i + 1], b = out->pixels[i + 2];
            if (saturationOn) shade::applySaturation(s.saturation[0], s.saturation[1], s.saturation[2], r, g, b);
            if (contrastOn) shade::applyNukeContrast(s.contrast[0], s.contrast[1], s.contrast[2], r, g, b);
            if (gammaOn) shade::applyGamma(gx, gy, gz, r, g, b);
            if (gainOffsetOn) shade::applyGainAndOffset(s.gain[0], s.gain[1], s.gain[2], s.offset[0], s.offset[1], s.offset[2], r, g, b);
            if (temperatureOn) shade::applyTMI(s.temperature[0], s.temperature[1], s.temperature[2], r, g, b);
            out->pixels[i] = std::max(r, 0.0f);
            out->pixels[i + 1] = std::max(g, 0.0f);
            out->pixels[i + 2] = std::max(b, 0.0f);
        }
    }
    out->point();

    // The distributions: each level's weights, then its mapping, then its tables.
    for (U32 level = 0; level < im.levelCount; ++level) {
        ImageLevel& lv = out->levels[level];
        const U32 w = lv.width, h = lv.height;
        lv.dists = U32(out->dists.size());
        const size_t base = out->cdf.size();
        out->cdf.resize(base + size_t(w) * h + h, 0.0f);
        out->guide.resize(base + size_t(w) * h + h, 0u);
        for (U32 y = 0; y < h; ++y) {
            Dist1D d{};
            d.offset = U32(base + size_t(y) * w);
            d.size = w;
            out->dists.push_back(d);
        }
        Dist1D marg{};
        marg.offset = U32(base + size_t(w) * h);
        marg.size = h;
        out->dists.push_back(marg);
        out->point();
        float* weights = out->cdf.data() + base;
        if (im.transformed) {
            const float* m = im.m;
            const float uCount = std::ceil(2.0f * std::max(std::fabs(m[0]), std::fabs(m[1])));
            const float vCount = std::ceil(2.0f * std::max(std::fabs(m[3]), std::fabs(m[4])));
            const float uRcp = 1.0f / uCount, vRcp = 1.0f / vCount;
            const float uOffset = 0.5f * (uCount - 1.0f), vOffset = 0.5f * (vCount - 1.0f);
            for (U32 y = 0; y < h; ++y) {
                const float v = (float(y) + 0.5f) / float(h);
                for (U32 x = 0; x < w; ++x) {
                    const float u = (float(x) + 0.5f) / float(w);
                    V3 color{0, 0, 0};
                    for (float vs = 0.0f; vs < vCount; vs += 1.0f) {
                        const float va = v + (vs - vOffset) * vRcp;
                        for (float us = 0.0f; us < uCount; us += 1.0f) {
                            const V3 c = imageEval(out->view, u + (us - uOffset) * uRcp, va, float(level), FilterNearest);
                            color = {color.x + c.x, color.y + c.y, color.z + c.z};
                        }
                    }
                    const float k = uRcp * vRcp;
                    weights[size_t(y) * w + x] = std::max(build::luminance({color.x * k, color.y * k, color.z * k}), SEpsilon);
                }
            }
        } else {
            for (U32 y = 0; y < h; ++y) {
                const float v = (float(y) + 0.5f) / float(h);
                for (U32 x = 0; x < w; ++x) {
                    const float u = (float(x) + 0.5f) / float(w);
                    weights[size_t(y) * w + x] = std::max(build::luminance(imageEval(out->view, u, v, float(level), FilterNearest)), SEpsilon);
                }
            }
        }
        // Distribution2D::tabulateCdf's mapping.
        if (s.mapping == MapSpherical || s.mapping == MapHemispherical) {
            const float scl = (3.14159265358979323846f - 0.0f) / ((float(h) - 0.5f) - (-0.5f));
            const float ofs = 0.0f - (-0.5f * scl);
            const U32 half = h / 2;
            for (U32 y = 0; y < h; ++y) {
                float sinTheta = std::sin(float(y) * scl + ofs);
                sinTheta = s.mapping == MapSpherical ? std::max(sinTheta, SEpsilon) : (y >= half ? sinTheta : SEpsilon);
                for (U32 x = 0; x < w; ++x) weights[size_t(y) * w + x] *= sinTheta;
            }
        } else if (s.mapping == MapCircular) {
            const float sclU = 2.0f / ((float(w) - 0.5f) - (-0.5f)), ofsU = -1.0f - (-0.5f * sclU);
            const float sclV = 2.0f / ((float(h) - 0.5f) - (-0.5f)), ofsV = -1.0f - (-0.5f * sclV);
            const float halfU = 2.0f / float(w), halfV = 2.0f / float(h);
            for (U32 y = 0; y < h; ++y) {
                const float tCorner = std::fabs(float(y) * sclV + ofsV) - halfV;
                const float t = std::max(tCorner, 0.0f), t2 = t * t;
                for (U32 x = 0; x < w; ++x) {
                    const float sCorner = std::fabs(float(x) * sclU + ofsU) - halfU;
                    const float sv = std::max(sCorner, 0.0f);
                    if (sv * sv + t2 > 1.0f) weights[size_t(y) * w + x] = SEpsilon;
                }
            }
        }
        for (U32 y = 0; y < h; ++y) {
            const float integral = build::tabulate(out->cdf, out->guide, out->dists[lv.dists + y]);
            out->cdf[out->dists[lv.dists + h].offset + y] = integral;
        }
        build::tabulate(out->cdf, out->guide, out->dists[lv.dists + h]);
        out->point();
    }
    (void)actual;
    return out;
}

}  // namespace xpu2::interactive::lights
