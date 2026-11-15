// The render outputs' files as MoonRay writes them (rndr/RenderOutputWriter.cc, ExrUtils.cc): a file each, a part each
// of its images (multi-part EXR where parts are named), each output's channels in order with its own half or float
// format, the part's compression (and DWA level), its metadata (exr_header_attributes), the display window the camera's
// aperture and the data window the render region.
#pragma once
#include "aov_plan.h"
#include "async_render.h"
#include "aov_crypto.h"
#include <OpenImageIO/imageio.h>
#include <array>
#include <functional>
#include <sstream>

namespace xpu2::interactive::aov {

// ExrUtils.cc writeExrHeader: one metadata attribute by its type.
inline void writeExrHeader(OIIO::ImageSpec& spec, const std::string& name, const std::string& type, const std::string& value,
                           std::vector<std::string>& warnings, const std::string& metadataName) {
    auto values = [&](size_t n) {
        std::istringstream in(value);
        std::vector<double> v;
        double x;
        while (v.size() < n && in >> x) v.push_back(x);
        if (v.size() != n) throw std::runtime_error("error converting to array of values");
        return v;
    };
    try {
        if (type == "float") spec.attribute(name, std::stof(value));
        else if (type == "int") spec.attribute(name, std::stoi(value));
        else if (type == "string") spec.attribute(name, value);
        else if (type == "double") spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::DOUBLE, OIIO::TypeDesc::SCALAR), value);
        else if (type == "chromaticities") {
            const auto v = values(8);
            float f[8];
            for (int i = 0; i < 8; ++i) f[i] = float(v[i]);
            spec.attribute(type, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::SCALAR, 8), f);  // MoonRay names it by its type
        } else if (type == "v2i" || type == "v3i") {
            const size_t n = type == "v2i" ? 2 : 3;
            const auto v = values(n);
            int a[3];
            for (size_t i = 0; i < n; ++i) a[i] = int(v[i]);
            spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::INT, n == 2 ? OIIO::TypeDesc::VEC2 : OIIO::TypeDesc::VEC3), a);
        } else if (type == "v2f" || type == "v3f") {
            const size_t n = type == "v2f" ? 2 : 3;
            const auto v = values(n);
            float a[3];
            for (size_t i = 0; i < n; ++i) a[i] = float(v[i]);
            spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, n == 2 ? OIIO::TypeDesc::VEC2 : OIIO::TypeDesc::VEC3), a);
        } else if (type == "m33f" || type == "m44f") {
            const size_t n = type == "m33f" ? 9 : 16;
            const auto v = values(n);
            float a[16];
            for (size_t i = 0; i < n; ++i) a[i] = float(v[i]);
            spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, n == 9 ? OIIO::TypeDesc::MATRIX33 : OIIO::TypeDesc::MATRIX44), a);
        } else if (type == "box2i") {
            const auto v = values(4);
            int a[4] = {int(v[0]), int(v[1]), int(v[2]), int(v[3])};
            if (a[0] > a[2]) std::swap(a[0], a[2]);
            if (a[1] > a[3]) std::swap(a[1], a[3]);
            spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::INT, OIIO::TypeDesc::VEC2, 2), a);
        } else if (type == "box2f") {
            const auto v = values(4);
            float a[4] = {float(v[0]), float(v[1]), float(v[2]), float(v[3])};
            if (a[0] > a[2]) std::swap(a[0], a[2]);
            if (a[1] > a[3]) std::swap(a[1], a[3]);
            spec.attribute(name, OIIO::TypeDesc(OIIO::TypeDesc::FLOAT, OIIO::TypeDesc::VEC2, 2), a);
        } else {
            throw std::runtime_error("datatype " + type + " is not supported");
        }
    } catch (const std::exception& e) {
        warnings.push_back("Metadata(\"" + metadataName + "\") : \"" + name + "\": " + e.what());
    }
}

// An output's channel values at a pixel (fillPixBufferSingleEntry), the render's rows from the bottom.
inline void outputValues(const Plan& plan, const Output& out, const async::RenderResult& r, size_t pixel, float* v) {
    switch (out.result) {
    case ResultBeauty:
        for (int c = 0; c < 3; ++c) v[c] = r.rgba[pixel * 4 + c];
        return;
    case ResultAlpha: v[0] = r.rgba[pixel * 4 + 3]; return;
    case ResultBeautyAux:
        for (int c = 0; c < 3; ++c) v[c] = r.odd.empty() ? 0.0f : r.odd[pixel * 4 + c];
        return;
    case ResultAlphaAux: v[0] = r.odd.empty() ? 0.0f : r.odd[pixel * 4 + 3]; return;
    case ResultHeatMap: v[0] = r.heat.empty() ? 0.0f : r.heat[pixel]; return;
    case ResultDisplayFilter: {
        const auto f = r.displayFilters.find(out.name);
        for (int c = 0; c < 3; ++c) v[c] = f == r.displayFilters.end() ? 0.0f : f->second[pixel * 3 + c];
        return;
    }
    case ResultCryptomatte: {
        // outputFragments: the regular block (vector mode records no reflected or refracted one: theirs are empty),
        // two fragments a layer of ID and coverage (scaled for the coverage of fragments left out), then each
        // fragment's extra data, zeros past the last.
        size_t at = 0;
        for (int type = 0; type < 3; ++type) {
            if ((type == 1 && !out.cryptoReflected) || (type == 2 && !out.cryptoRefracted)) continue;
            static const std::vector<CryptoBuffer::Fragment> none;
            const auto& list = type == 0 && r.crypto ? r.crypto->fragments(U32(out.cryptoCategory), pixel) : none;
            const size_t most = size_t(2 * out.cryptoLayers);
            float total = 0.0f, kept = 0.0f;
            for (const auto& f : list) total += f.coverage;
            size_t n = 0;
            for (const auto& f : list) {
                kept += f.coverage;
                if (++n >= most) break;
            }
            const float scale = total / kept;
            n = 0;
            for (const auto& f : list) {
                v[at++] = f.id;
                v[at++] = f.coverage * scale;
                if (++n >= most) break;
            }
            for (; n < most; ++n) {
                v[at++] = 0.0f;
                v[at++] = 0.0f;
            }
            const bool* x = out.cryptoExtra;
            const U32 extra = (x[0] ? 4 : 0) + (x[1] ? 4 : 0) + (x[2] ? 4 : 0) + (x[3] ? 4 : 0) + (x[4] ? 4 : 0) + (x[5] ? 4 : 0) +
                              (x[6] ? 2 : 0) + (x[7] ? 2 : 0);
            if (!extra) continue;
            n = 0;
            for (const auto& f : list) {
                auto put4 = [&](V3 a, float w) {
                    v[at++] = a.x;
                    v[at++] = a.y;
                    v[at++] = a.z;
                    v[at++] = w;
                };
                if (x[0]) put4(f.position, 0.0f);
                if (x[1]) put4(f.p0, 0.0f);
                if (x[2]) put4(f.normal, 0.0f);
                if (x[3]) {
                    for (int k = 0; k < 4; ++k) v[at++] = f.beauty[k];
                }
                if (x[4]) put4(f.refP, 0.0f);
                if (x[5]) put4(f.refN, 0.0f);
                if (x[6]) {
                    v[at++] = f.uv[0];
                    v[at++] = f.uv[1];
                }
                if (x[7]) {
                    v[at++] = float(f.presenceDepth);
                    v[at++] = float(f.samples);
                }
                if (++n >= most) break;
            }
            for (size_t i = n * extra; i < most * extra; ++i) v[at++] = 0.0f;
        }
        return;
    }
    case ResultWeight: v[0] = float(r.counts[pixel]); return;
    default: break;
    }
    if (out.entry < 0) {
        for (size_t c = 0; c < out.channels.size(); ++c) v[c] = 0.0f;
        return;
    }
    const Entry& e = plan.entries[size_t(out.entry)];
    const float* a = r.aov.data() + pixel * plan.channels + e.offset;
    if (e.type == TypeVisibility) {
        if (plan.resumable) {
            v[0] = a[0];
            v[1] = a[1];
            v[2] = a[1] > 0.0f ? a[0] / a[1] : 0.0f;
        } else {
            v[0] = a[1] > 0.0f ? a[0] / a[1] : 0.0f;
        }
        return;
    }
    for (U32 c = 0; c < e.channels; ++c) v[c] = a[c];
    if (out.channels.size() > e.channels) v[e.channels] = 0.0f;  // a resumable closest output's depth (not kept here)
}

// RenderOutputWriter: every file of the plan, or with checkpoint each under its checkpoint name (a file without one
// is left out). written (or none): each file's name once it is written. Returns the first error ("" when all were).
inline std::string writeOutputs(const Plan& plan, const async::RenderResult& r, unsigned width, unsigned height,
                                std::vector<std::string>& warnings, bool checkpoint = false,
                                const std::function<void(const std::string&)>& written = {}) {
    static const char* const compressions[] = {"none", "zip", "rle", "zips", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"};
    for (const File& planned : plan.files) {
        if (checkpoint && planned.checkpoint.empty()) continue;
        // A deep file is written apart (aov_deep_write.h).
        {
            bool deep = false;
            for (const Image& i : planned.images)
                for (const Output& o : i.outputs) deep |= o.deep;
            if (deep) continue;
        }
        File file = planned;
        if (checkpoint) file.name = planned.checkpoint;
        std::vector<OIIO::ImageSpec> specs;
        for (const Image& image : file.images) {
            int channels = 0;
            for (const Output& out : image.outputs) channels += int(out.channels.size());
            OIIO::ImageSpec spec(int(width), int(height), channels, OIIO::TypeDesc::FLOAT);
            spec.x = plan.region[0];
            spec.y = plan.aperture[3] - plan.region[3];  // y flipped against the display window
            spec.full_x = plan.aperture[0];
            spec.full_y = plan.aperture[1];
            spec.full_width = plan.aperture[2] - plan.aperture[0];
            spec.full_height = plan.aperture[3] - plan.aperture[1];
            spec.channelformats.clear();
            spec.channelnames.clear();
            for (const Output& out : image.outputs)
                for (const auto& name : out.channels) {
                    spec.channelformats.push_back(out.half ? OIIO::TypeDesc::HALF : OIIO::TypeDesc::FLOAT);
                    spec.channelnames.push_back(name);
                }
            if (!image.part.empty()) spec.attribute("name", image.part);
            spec.attribute("oiio::ColorSpace", "Linear");
            const Output& first = image.outputs.front();
            const int comp = first.compression >= 0 && first.compression < 10 ? first.compression : 1;
            spec.attribute("compression", compressions[comp]);
            if (comp == 8 || comp == 9) spec.attribute("openexr:dwaCompressionLevel", std::max(0.0f, first.compressionLevel));
            for (const Output& out : image.outputs) {
                if (!out.metadata) continue;
                for (size_t i = 0; i < out.metadataNames.size() && i < out.metadataTypes.size() && i < out.metadataValues.size(); ++i)
                    writeExrHeader(spec, out.metadataNames[i], out.metadataTypes[i], out.metadataValues[i], warnings, out.metadataName);
                break;
            }
            specs.push_back(spec);
        }
        auto io = OIIO::ImageOutput::create(file.name);
        if (!io) return "Cannot write " + file.name;
        const bool multi = specs.size() > 1;
        if (multi && !io->supports("multiimage")) return "Cannot write the parts of " + file.name;
        if (!(multi ? io->open(file.name, int(specs.size()), specs.data()) : io->open(file.name, specs[0])))
            return "Cannot write " + file.name + ": " + io->geterror();
        for (size_t s = 0; s < file.images.size(); ++s) {
            const Image& image = file.images[s];
            if (s > 0 && !io->open(file.name, specs[s], OIIO::ImageOutput::AppendSubimage))
                return "Cannot write " + file.name + ": " + io->geterror();
            const int channels = specs[s].nchannels;
            std::vector<float> data(size_t(width) * height * size_t(channels));
            std::vector<float> values(size_t(channels) + 16);
            float* v = values.data();
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x) {
                    const size_t pixel = (height - 1 - y) * width + x;  // the film's rows from the bottom
                    float* dst = data.data() + (y * width + x) * size_t(channels);
                    for (const Output& out : image.outputs) {
                        outputValues(plan, out, r, pixel, v);
                        for (size_t c = 0; c < out.channels.size(); ++c) *dst++ = v[c];
                    }
                }
            if (!io->write_image(OIIO::TypeDesc::FLOAT, data.data())) return "Cannot write " + file.name + ": " + io->geterror();
        }
        if (!io->close()) return "Cannot write " + file.name + ": " + io->geterror();
        if (written) written(file.name);
    }
    return "";
}

}  // namespace xpu2::interactive::aov
