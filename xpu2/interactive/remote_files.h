// Files read without being on this machine's disk whole: their bytes come in blocks from a source, each block fetched
// the first time a read touches it and kept (Rhizome, RHIZOME.md plan 16: a node's textures from the swarm's memory as it
// renders, in place of a copy of each). Registered by path; OpenImageIO reads them through an IOProxy, so the tiles a
// render never touches are never fetched. The texture code asks here before it opens a file (texture::File, Pyramid,
// the tile cache).
#pragma once
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace xpu2::remote {

using Block = std::shared_ptr<const std::vector<unsigned char>>;

// Blocks fetched by every source so far (a render that fetched some waited for them: Rhizome leaves its time out of the
// machine's rate).
inline std::atomic<uint64_t>& fetchedBlocks() {
    static std::atomic<uint64_t> n{0};
    return n;
}

// A file's bytes as blocks of blockBytes (the last one shorter). block(i): block i, waiting for it as long as it takes
// (null: it cannot be had; the read fails as a missing file's would).
class Source {
public:
    Source(uint64_t size, size_t blockBytes, std::function<Block(size_t)> block) : size_(size), blockBytes_(blockBytes), block_(std::move(block)) {}
    uint64_t size() const { return size_; }
    // Bytes [at, at + n) into buf; how many were read.
    size_t read(void* buf, size_t n, uint64_t at) {
        if (at >= size_) return 0;
        n = size_t(std::min<uint64_t>(n, size_ - at));
        size_t done = 0;
        while (done < n) {
            const uint64_t pos = at + done;
            const size_t i = size_t(pos / blockBytes_), within = size_t(pos % blockBytes_);
            const Block b = get(i);
            if (!b || b->size() <= within) break;
            const size_t take = std::min(n - done, b->size() - within);
            std::memcpy(static_cast<unsigned char*>(buf) + done, b->data() + within, take);
            done += take;
        }
        return done;
    }
    // Blocks fetched so far, and their bytes.
    size_t blocksHeld() {
        std::lock_guard<std::mutex> g(m_);
        return held_.size();
    }
    uint64_t bytesHeld() {
        std::lock_guard<std::mutex> g(m_);
        uint64_t n = 0;
        for (const auto& [i, b] : held_) n += b ? b->size() : 0;
        return n;
    }
    size_t blocks() const { return size_t((size_ + blockBytes_ - 1) / blockBytes_); }

private:
    Block get(size_t i) {
        {
            std::lock_guard<std::mutex> g(m_);
            if (auto it = held_.find(i); it != held_.end()) return it->second;
        }
        // Fetched outside the lock: threads reading other blocks go on meanwhile (two asking for one block fetch it
        // twice at worst, and the second is kept).
        Block b = block_(i);
        if (!b) return nullptr;
        fetchedBlocks().fetch_add(1);
        std::lock_guard<std::mutex> g(m_);
        return held_.emplace(i, std::move(b)).first->second;
    }
    uint64_t size_;
    size_t blockBytes_;
    std::function<Block(size_t)> block_;
    std::mutex m_;
    std::map<size_t, Block> held_;
};

// OpenImageIO's view of a source: reads and positioned reads (thread-safe, as the readers use them).
class Proxy : public OIIO::Filesystem::IOProxy {
public:
    Proxy(const std::string& name, std::shared_ptr<Source> s) : IOProxy(name, Read), s_(std::move(s)) {}
    const char* proxytype() const override { return "remote"; }
    size_t read(void* buf, size_t size) override {
        const size_t n = s_->read(buf, size, uint64_t(m_pos));
        m_pos += int64_t(n);
        return n;
    }
    size_t pread(void* buf, size_t size, int64_t offset) override { return offset < 0 ? 0 : s_->read(buf, size, uint64_t(offset)); }
    size_t write(const void*, size_t) override { return 0; }
    size_t pwrite(const void*, size_t, int64_t) override { return 0; }
    size_t size() const override { return size_t(s_->size()); }

private:
    std::shared_ptr<Source> s_;
};

namespace detail {
inline std::string key(const std::string& path) { return std::filesystem::path(path).lexically_normal().make_preferred().string(); }
struct Registry {
    std::mutex m;
    std::map<std::string, std::shared_ptr<Source>> files;
    // The proxies each image cache was given (it keeps them open for as long as it lives).
    std::map<std::pair<const void*, std::string>, std::unique_ptr<Proxy>> cached;
};
inline Registry& registry() {
    static Registry r;
    return r;
}
}  // namespace detail

// A file at path read from source (its own path need not exist).
inline void add(const std::string& path, std::shared_ptr<Source> source) {
    auto& r = detail::registry();
    std::lock_guard<std::mutex> g(r.m);
    r.files[detail::key(path)] = std::move(source);
}
// The files under folder let go (a job's, as its staging folder is emptied). Image caches keep their proxies, as they
// keep the files open; a path registered again is a cache entry of its own only once the cache lets the old one go.
inline void removeUnder(const std::string& folder) {
    auto& r = detail::registry();
    const std::string k = detail::key(folder);
    std::lock_guard<std::mutex> g(r.m);
    for (auto it = r.files.begin(); it != r.files.end();)
        it = it->first.compare(0, k.size(), k) == 0 ? r.files.erase(it) : std::next(it);
}
inline std::shared_ptr<Source> find(const std::string& path) {
    auto& r = detail::registry();
    std::lock_guard<std::mutex> g(r.m);
    if (r.files.empty()) return nullptr;
    auto it = r.files.find(detail::key(path));
    return it == r.files.end() ? nullptr : it->second;
}
inline bool any() {
    auto& r = detail::registry();
    std::lock_guard<std::mutex> g(r.m);
    return !r.files.empty();
}

// An image opened from its source when it has one, else from disk. keep holds the proxy for as long as the image is
// open (destroyed after it).
inline std::unique_ptr<OIIO::ImageInput> open(const std::string& path, std::unique_ptr<Proxy>& keep) {
    if (auto s = find(path)) {
        keep = std::make_unique<Proxy>(path, std::move(s));
        OIIO::ImageSpec config;
        OIIO::Filesystem::IOProxy* p = keep.get();
        config.attribute("oiio:ioproxy", OIIO::TypeDesc::PTR, &p);
        return OIIO::ImageInput::open(path, &config, keep.get());
    }
    return OIIO::ImageInput::open(path);
}
// An image cache told to read path from its source (once a cache and path; nothing for a file on disk).
inline void addToCache(OIIO::ImageCache& cache, const std::string& path) {
    auto s = find(path);
    if (!s) return;
    auto& r = detail::registry();
    std::lock_guard<std::mutex> g(r.m);
    auto& slot = r.cached[{&cache, detail::key(path)}];
    if (slot) return;
    slot = std::make_unique<Proxy>(path, std::move(s));
    OIIO::ImageSpec config;
    OIIO::Filesystem::IOProxy* p = slot.get();
    config.attribute("oiio:ioproxy", OIIO::TypeDesc::PTR, &p);
    cache.add_file(OIIO::ustring(path), nullptr, &config, true);
}

}  // namespace xpu2::remote
