// Checks of the texture levels read on demand (host_texture.h's Pyramid::loadTile): a read decodes the file's own tile
// that holds the tile asked for, once, and keeps every tile of ours in it, with the same texels a whole level's decode
// gives. Sponza (2026-11-18) stalled XPU2 for want of this: read 32 x 32 at a time through an OpenImageIO cache too small
// for the scene, each of a .tx's 64 x 64 tiles was decoded again for each of our four in it (19 GB read from 8.8 GB of
// files), a few thousand tiles a second, while the card waited for 1.65 million of them. And a texture file's size and
// time for the tessellation cache's key (texture::File::stamp), read once: read for every mesh piece, they took 18-30 s of
// Sponza's preparation.
//
// Built and run from the MoonRayForModo tree (MSYS2 UCRT64 toolchain on PATH):
//   g++ -std=c++20 -O2 -Itoolchain/msys64/ucrt64/include -o host_texture_test.exe xpu2/interactive/host_texture_test.cc
//       -lOpenImageIO -lOpenImageIO_Util
//   host_texture_test.exe
// Prints a line a check and a JSON summary; exits non-zero on a failure. Writes its textures to a folder of its own in
// the temporary folder and removes it.
#include "host_texture.h"
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <cstdio>
#include <filesystem>
#include <random>
#include <thread>

using namespace xpu2::interactive;

static int failures = 0, checks = 0;
static void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// A mipmapped float .tx of width x height and channels, its tiles fileTile a side, its texels a hash of their place (so
// that a texel read from the wrong place shows), a quarter of its first tile flat (a flat tile of ours) and one texel past
// half floats' range (a wide tile of ours).
static std::string makeTexture(const std::filesystem::path& folder, const char* name, int width, int height, int channels, int fileTile) {
    std::vector<float> texels(size_t(width) * height * channels);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c) {
                uint32_t h = uint32_t(x) * 0x9e3779b1u ^ uint32_t(y) * 0x85ebca77u ^ uint32_t(c + 1) * 0xc2b2ae3du;
                h ^= h >> 15, h *= 0x2c1b3c6du, h ^= h >> 12;
                float v = float(h >> 8) * (1.0f / 16777216.0f) * 4.0f;
                if (x < 32 && y < 32) v = 0.25f;  // the first tile of ours flat
                texels[(size_t(y) * width + x) * channels + c] = v;
            }
    for (int c = 0; c < channels; ++c) texels[(size_t(height - 1) * width + width - 1) * channels + c] = 100000.0f;  // past half floats' range
    OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::FLOAT);
    if (channels == 4) spec.alpha_channel = 3;
    OIIO::ImageBuf image(spec);
    image.set_pixels(OIIO::ROI::All(), OIIO::TypeDesc::FLOAT, texels.data());
    const std::string path = (folder / name).string();
    OIIO::ImageSpec config;
    config.tile_width = config.tile_height = fileTile;
    config.attribute("maketx:filtername", "box");
    if (!OIIO::ImageBufAlgo::make_texture(OIIO::ImageBufAlgo::MakeTxTexture, image, path, config))
        throw std::runtime_error("make_texture: " + OIIO::geterror());
    return path;
}

// Every texel of every level on demand, as tex::texel reads it from the pyramid's pages, against the level decoded whole
// from the file, held as the pyramid holds it (half floats rounded, or floats in a wide tile).
static bool levelsMatch(texture::Pyramid& p) {
    auto in = OIIO::ImageInput::open(p.file);
    if (!in) return false;
    for (unsigned level = 0; level < p.image.count; ++level) {
        const tex::Mip& m = p.image.levels[level];
        if (!m.load) continue;
        if (!in->seek_subimage(0, int(level) + p.fileBase)) return false;
        const auto& spec = in->spec();
        std::vector<float> whole(size_t(spec.width) * spec.height * spec.nchannels);
        if (!in->read_image(0, int(level) + p.fileBase, 0, spec.nchannels, OIIO::TypeDesc::FLOAT, whole.data())) return false;
        const unsigned nx = (m.width + tex::Tile - 1) / tex::Tile;
        for (unsigned y = 0; y < m.height; ++y)
            for (unsigned x = 0; x < m.width; ++x) {
                const unsigned tx = x / tex::Tile, tile = (y / tex::Tile) * nx + tx;
                const surface::V3* page = m.pages[tile];
                if (!page) return false;
                surface::V3 got;
                if (tex::texel(page, m.width, tx, x, y, got, m.format) != tex::Ready) return false;
                surface::V3 want = p.kept(&whole[(size_t(y) * spec.width + x) * spec.nchannels]);
                if (p.rounds && !tex::widePage(page)) want = texture::halfRoundedTexel(want);
                if (got.x != want.x || got.y != want.y || got.z != want.z) {
                    std::printf("     level %u texel %u, %u: %g %g %g, wanted %g %g %g\n", level, x, y, got.x, got.y, got.z, want.x, want.y, want.z);
                    return false;
                }
            }
    }
    return true;
}

static unsigned tilesOf(const tex::Mip& m) { return ((m.width + tex::Tile - 1) / tex::Tile) * ((m.height + tex::Tile - 1) / tex::Tile); }
static unsigned held(const tex::Mip& m) {
    unsigned n = 0;
    for (unsigned i = 0; i < tilesOf(m); ++i) n += m.pages[i] != nullptr;
    return n;
}

int main() {
    const auto folder = std::filesystem::temp_directory_path() / ("host_texture_test_" + std::to_string(GetCurrentProcessId()));
    std::filesystem::create_directories(folder);
    try {
        // 1. One tile asked for: the file's 64 x 64 tile holding it decoded once, its four tiles of ours kept.
        {
            const std::string file = makeTexture(folder, "rgb64.tx", 512, 512, 3, 64);
            texture::Pyramid p(file);
            const tex::Mip& m = p.image.levels[0];
            check(m.load != nullptr && p.image.levels[1].load != nullptr, "levels larger than the resident side are on demand");
            const unsigned nx = (m.width + tex::Tile - 1) / tex::Tile;
            texture::Pyramid::loadTile(m.loader, 0, nx + 1);  // tile (1, 1): in the file's first tile, with (0, 0), (1, 0), (0, 1)
            check(p.regionsRead == 1, "one region read for one tile asked for");
            check(m.pages[0] && m.pages[1] && m.pages[nx] && m.pages[nx + 1] && held(m) == 4, "the four tiles of the file's tile kept, no others");
            check(tex::flatPage(m.pages[0]) && !tex::flatPage(m.pages[1]), "a flat tile of the four kept as its value");
            texture::Pyramid::loadTile(m.loader, 0, 0);
            texture::Pyramid::loadTile(m.loader, 0, 1);
            check(p.regionsRead == 1, "its neighbours in the same file tile not read again");
            texture::Pyramid::loadTile(m.loader, 0, 2);
            check(p.regionsRead == 2 && held(m) == 8, "the next file tile read on its own");
            for (unsigned level = 0; level < p.image.count; ++level)
                if (p.image.levels[level].load)
                    for (unsigned t = 0; t < tilesOf(p.image.levels[level]); ++t) texture::Pyramid::loadTile(p.image.levels[level].loader, level, t);
            unsigned regions = 0;
            for (unsigned level = 0; level < p.image.count; ++level)
                if (p.image.levels[level].load) regions += (p.image.levels[level].width / 64) * (p.image.levels[level].height / 64);
            check(p.regionsRead == regions, "every file tile read once (" + std::to_string(p.regionsRead.load()) + " of " + std::to_string(regions) + ")");
            check(p.lazyLoaded == tilesOf(m) + tilesOf(p.image.levels[1]), "every tile of ours kept once");
            check(levelsMatch(p), "every texel the whole level's (half floats rounded, a wide tile in floats)");
            check(p.lazyWide >= 1, "the tile past half floats' range held in floats");
        }
        // 2. Many threads asking for every tile in turns of their own: each region read once, the texels the same.
        {
            const std::string file = makeTexture(folder, "grey64.tx", 1024, 768, 1, 64);
            texture::Pyramid p(file);
            std::vector<std::pair<unsigned, unsigned>> all;
            for (unsigned level = 0; level < p.image.count; ++level)
                if (p.image.levels[level].load)
                    for (unsigned t = 0; t < tilesOf(p.image.levels[level]); ++t) all.push_back({level, t});
            std::vector<std::thread> threads;
            for (unsigned k = 0; k < 12; ++k)
                threads.emplace_back([&, k] {
                    auto mine = all;
                    std::shuffle(mine.begin(), mine.end(), std::mt19937(k));
                    for (auto [level, t] : mine) texture::Pyramid::loadTile(p.image.levels[level].loader, level, t);
                });
            for (auto& t : threads) t.join();
            unsigned regions = 0;
            for (unsigned level = 0; level < p.image.count; ++level)
                if (p.image.levels[level].load) {
                    const auto& m = p.image.levels[level];
                    regions += ((m.width + 63) / 64) * ((m.height + 63) / 64);
                }
            check(p.regionsRead == regions, "12 threads: each file tile read once (" + std::to_string(p.regionsRead.load()) + " of " + std::to_string(regions) + ")");
            check(p.lazyLoaded == all.size(), "12 threads: each tile of ours kept once");
            check(levelsMatch(p), "12 threads: every texel the whole level's (grey, one channel)");
        }
        // 3. Files with other tiles: smaller than ours (each of ours its own read), larger (four of ours a side at most),
        // and a size whose edge tiles are partial.
        {
            const std::string small = makeTexture(folder, "rgb16.tx", 256, 256, 3, 16), large = makeTexture(folder, "rgb256.tx", 512, 512, 3, 256),
                              odd = makeTexture(folder, "rgb64odd.tx", 600, 360, 4, 64);
            texture::Pyramid ps(small), pl(large), po(odd);
            texture::Pyramid::loadTile(ps.image.levels[0].loader, 0, 0);
            check(ps.regionsRead == 1 && held(ps.image.levels[0]) == 1, "16 x 16 file tiles: a tile of ours read alone");
            texture::Pyramid::loadTile(pl.image.levels[0].loader, 0, 0);
            check(pl.regionsRead == 1 && held(pl.image.levels[0]) == 16, "256 x 256 file tiles: four of ours a side read at once");
            const tex::Mip& m = po.image.levels[0];
            const unsigned nx = (m.width + tex::Tile - 1) / tex::Tile, ny = (m.height + tex::Tile - 1) / tex::Tile;
            texture::Pyramid::loadTile(m.loader, 0, nx * ny - 1);  // the last, partial both ways
            check(po.regionsRead == 1 && m.pages[nx * ny - 1] != nullptr, "the corner tile of a 600 x 360 texture read");
            for (auto* p : {&ps, &pl, &po})
                for (unsigned level = 0; level < p->image.count; ++level)
                    if (p->image.levels[level].load)
                        for (unsigned t = 0; t < tilesOf(p->image.levels[level]); ++t) texture::Pyramid::loadTile(p->image.levels[level].loader, level, t);
            check(levelsMatch(ps) && levelsMatch(pl) && levelsMatch(po), "every texel the whole level's (16, 256 and 64 x 64 file tiles, partial edges, alpha)");
        }
        // 4. A tile let go under memory pressure is read again, its region with it.
        {
            const std::string file = makeTexture(folder, "rgb64b.tx", 256, 256, 3, 64);
            texture::Pyramid p(file);
            const tex::Mip& m = p.image.levels[0];
            texture::Pyramid::loadTile(m.loader, 0, 0);
            std::vector<texture::Block> grave;
            p.letGo(size_t(1) << 40, grave);
            grave.clear();
            check(held(m) == 0, "every tile let go");
            texture::Pyramid::loadTile(m.loader, 0, 1);
            check(p.regionsRead == 2 && held(m) == 4, "a tile let go read again with its region");
        }
        // 5. A texture file's size and time for the tessellation cache's key (cpu_scene.h geometryKey), read once while the
        // session holds the file: read for every mesh piece, they took 18-30 s of Sponza's preparation.
        {
            const std::string file = makeTexture(folder, "stamp.tx", 128, 128, 3, 64);
            texture::File f(file, folder, texture::system());
            std::error_code ec;
            const auto size = std::filesystem::file_size(file, ec);
            const auto when = std::filesystem::last_write_time(file, ec);
            const auto& s = f.stamp();
            check(s.size == size && s.when == when.time_since_epoch().count(), "the stamp is the file's size and time, as the key read them");
            const HANDLE writer = CreateFileA(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            check(writer == INVALID_HANDLE_VALUE, "the lease keeps writers out while the file is held");
            if (writer != INVALID_HANDLE_VALUE) CloseHandle(writer);
            // A time set by hand (attributes pass the lease), seen by a new read and not by the stamp.
            bool touched = false;
            if (const HANDLE h = CreateFileA(file.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                             OPEN_EXISTING, 0, nullptr);
                h != INVALID_HANDLE_VALUE) {
                FILETIME t{};
                GetSystemTimeAsFileTime(&t);
                ULARGE_INTEGER u{{t.dwLowDateTime, t.dwHighDateTime}};
                u.QuadPart += 36000000000ull;  // an hour on
                t = {u.LowPart, u.HighPart};
                touched = SetFileTime(h, nullptr, nullptr, &t) != 0;
                CloseHandle(h);
            }
            check(touched && std::filesystem::last_write_time(file, ec) != when && &f.stamp() == &s && f.stamp().when == when.time_since_epoch().count(),
                  "the stamp read once, not again for each key");
        }
    } catch (const std::exception& e) {
        check(false, std::string("exception: ") + e.what());
    }
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::printf("{\"checks\": %d, \"failures\": %d}\n", checks, failures);
    return failures ? 1 : 0;
}
