// Deep files as MoonRay writes them (pbr/core/DeepBuffer.cc write, writeHardSurfaceSegments and its NoMask form;
// rndr/RenderOutputDriverImplWrite.cc deepWrite), through MoonRay's own OpenDCX library (libdeepfile): each file
// whose outputs are all deep, its channels Z, ZBack, the subpixel mask and flags (OpenDCX 2.0 alone), A, the deep IDs
// as deep.<name>, and each output's channels; a pixel's hard surface segments, with masks as they are, or without
// (OpenEXR 2.0) front to back with each one's alpha from its coverage of what the ones before it left. Host only.
#pragma once
#include "aov_deep.h"
#include "aov_plan.h"
#include <OpenEXR/ImfHeader.h>
#include <algorithm>
#include <string>
#include <vector>
#include "deepfile/DcxChannelContext.h"
#include "deepfile/DcxChannelSet.h"
#include "deepfile/DcxDeepImageTile.h"

namespace xpu2::interactive::aov {

// Whether a file of the plan is a deep file (all its outputs deep; a mix is MoonRay's error, and XPU2's).
inline bool deepFile(const File& f, bool& mixed) {
    bool deep = false, flat = false;
    for (const Image& i : f.images)
        for (const Output& o : i.outputs) (o.deep ? deep : flat) = true;
    mixed = deep && flat;
    return deep && !flat;
}

// DeepBuffer::write for one file: film rows as MoonRay's film keeps them, written from the data window's first line.
// samples: each pixel's samples (the NoMask form's coverage). Returns "" or what went wrong.
inline std::string writeDeepFile(const std::string& filename, const File& file, const Plan& plan, const DeepFilm& film,
                                 const std::vector<U32>& samples) {
    namespace dcx = OPENDCX_INTERNAL_NAMESPACE;
    try {
        dcx::ChannelSet chanSet;
        chanSet.insert(dcx::Chan_ZFront);
        chanSet.insert(dcx::Chan_ZBack);
        const bool masks = plan.deepFormat == 1;
        if (masks) {
            chanSet.insert(dcx::Chan_SpBits1);
            chanSet.insert(dcx::Chan_SpBits2);
            chanSet.insert(dcx::Chan_DeepFlags);
        }
        chanSet.insert(dcx::Chan_A);
        dcx::ChannelContext chanCtx;
        std::vector<dcx::ChannelIdx> idChannels;
        const U32 ids = film.idCount();
        for (U32 i = 0; i < ids && i < plan.deepIds.size(); ++i) {
            chanCtx.addChannelAlias(plan.deepIds[i], "deep", dcx::Chan_Invalid, 0, "", Imf::FLOAT, 0);
            idChannels.push_back(chanCtx.getChannel(std::string("deep.") + plan.deepIds[i]));
            chanSet.insert(idChannels.back());
        }
        // Each output's channels: the beauty's three, or the alpha's one (what XPU2 keeps a segment).
        struct Out {
            int result;
            std::vector<dcx::ChannelIdx> channels;
        };
        std::vector<Out> outs;
        for (const Image& image : file.images)
            for (const Output& o : image.outputs) {
                Out out{o.result, {}};
                for (const std::string& name : o.channels) {
                    const size_t dot = name.find('.');
                    const std::string channel = name.substr(dot == std::string::npos ? 0 : dot + 1);
                    const std::string layer = name.substr(0, dot);
                    chanCtx.addChannelAlias(channel, layer, dcx::Chan_Invalid, 0, "", Imf::HALF, 0);
                    out.channels.push_back(chanCtx.getChannel(name));
                    chanSet.insert(out.channels.back());
                }
                outs.push_back(std::move(out));
            }
        const IMATH_NAMESPACE::Box2i displayWindow(IMATH_NAMESPACE::V2i(plan.aperture[0], plan.aperture[1]),
                                                   IMATH_NAMESPACE::V2i(plan.aperture[2] - 1, plan.aperture[3] - 1));
        const IMATH_NAMESPACE::Box2i dataWindow(IMATH_NAMESPACE::V2i(plan.region[0], plan.region[1]),
                                                IMATH_NAMESPACE::V2i(plan.region[2] - 1, plan.region[3] - 1));
        dcx::DeepImageOutputTile tile(displayWindow, dataWindow, true, chanSet, chanCtx);
        Imf::Header header;
        tile.setOutputFile(filename.c_str(), header);
        const U32 width = film.width();
        size_t idx = 0;
        std::vector<const DeepFilm::Segment*> order;
        for (int outY = tile.minY(), filmY = 0; outY <= tile.maxY(); ++outY, ++filmY) {
            for (int outX = tile.minX(), filmX = 0; outX <= tile.maxX(); ++outX, ++filmX, ++idx) {
                const size_t pixel = size_t(filmY) * width + size_t(filmX);
                const auto& segs = film.segments(pixel);
                if (segs.empty()) {
                    tile.clearDeepPixel(outX, outY);
                    continue;
                }
                dcx::DeepPixel dcxpixel(chanSet);
                auto put = [&](const DeepFilm::Segment& s, float alpha, float scale, dcx::Pixelf& dp) {
                    dp[dcx::Chan_A] = alpha;
                    for (size_t i = 0; i < idChannels.size(); ++i) dp[idChannels[i]] = s.ids[i];
                    for (const Out& o : outs) {
                        // DeepBuffer's channels: the beauty's sums, or the alpha set as the segment's alpha (then divided
                        // by the samples again, as MoonRay does).
                        for (size_t c = 0; c < o.channels.size(); ++c) {
                            const float v = o.result == ResultBeauty ? (c < 3 ? s.rgb[c] : 0.0f) : alpha;
                            dp[o.channels[c]] = v / s.weight * scale;
                        }
                    }
                };
                if (masks) {
                    for (const DeepFilm::Segment& s : segs) {
                        dcx::DeepSegment ds;
                        dcx::Pixelf dp(chanSet);
                        ds.Zf = -s.front * s.rayZ;
                        ds.Zb = -s.back * s.rayZ;
                        ds.index = -1;
                        ds.metadata.spmask = (unsigned long long)s.maskLo | ((unsigned long long)s.maskHi << 32);
                        ds.metadata.flags = dcx::DeepFlags::LINEAR_INTERP;
                        const float alpha = s.alpha / s.weight;
                        put(s, alpha, 1.0f, dp);
                        dcxpixel.append(ds, dp);
                    }
                } else {
                    // Front to back, each one's alpha from its coverage of what is left (writeHardSurfaceSegmentsNoMask).
                    order.clear();
                    for (const DeepFilm::Segment& s : segs) order.push_back(&s);
                    std::sort(order.begin(), order.end(), [](const DeepFilm::Segment* a, const DeepFilm::Segment* b) {
                        return -a->back * a->rayZ < -b->back * b->rayZ;
                    });
                    float total = 0.0f;
                    const int spp = std::max(1, int(pixel < samples.size() ? samples[pixel] : 1u));
                    for (const DeepFilm::Segment* s : order) {
                        dcx::DeepSegment ds;
                        dcx::Pixelf dp(chanSet);
                        ds.Zf = ds.Zb = -s->back * s->rayZ;
                        ds.index = -1;
                        ds.metadata.spmask = 0xffffffffffffffffull;
                        ds.metadata.flags = dcx::DeepFlags::LINEAR_INTERP;
                        const float coverage = s->weight / float(spp);
                        const float alpha = coverage / (1.0f - total);
                        total += coverage;
                        put(*s, alpha, alpha, dp);
                        dcxpixel.append(ds, dp);
                        if (total >= 1.0f) break;
                    }
                }
                tile.setDeepPixel(outX, outY, dcxpixel);
            }
            tile.writeScanline(outY, true);
        }
        return "";
    } catch (const std::exception& e) {
        return filename + ": " + e.what();
    }
}

}  // namespace xpu2::interactive::aov
