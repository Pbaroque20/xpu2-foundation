// Rhizome's jobs (RHIZOME.md, plan 13): frame ranges, a frame's files and outputs by their #### patterns, and the job
// queue, a folder of job files a master renders one after another.
//
// A job is a scene (its -in files), the changes each frame makes to it (--delta files, read on top of the frame before,
// as Modo's session sends its changes), the frames, and the picture to write. A path with a run of # (or %04d) names
// a file a frame: shot.####.rdla is shot.0012.rdla for frame 12. A scene given a frame (-in shot.####.rdla) is made
// again each frame, its files that did not change not sent again (the pool's pieces are their own hashes); a scene
// given once with a delta a frame (-in shot.rdla --delta shot.####.delta.rdla) is kept on every machine between frames
// and the delta alone is sent and read.
//
// The queue: a folder of .job files, rendered in their names' order. A job being rendered is in running\, a finished
// one moves to done\ (failed\ with why). moonray_xpu2 rhizome submit writes one.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace xpu2::interactive::rhizome {

// "12", "1-24", "1-24:2" (every second), "1,5,9-12", and lists of those by commas. Empty: none (the scene as it is).
inline std::vector<int> parseFrames(const std::string& spec) {
    std::vector<int> out;
    std::stringstream parts(spec);
    std::string part;
    while (std::getline(parts, part, ',')) {
        part.erase(std::remove_if(part.begin(), part.end(), [](char c) { return c == ' ' || c == '\t'; }), part.end());
        if (part.empty()) continue;
        int step = 1;
        const auto colon = part.find(':');
        if (colon != std::string::npos) {
            step = std::stoi(part.substr(colon + 1));
            part = part.substr(0, colon);
            if (step < 1) throw std::runtime_error("a frame step is at least 1: " + spec);
        }
        // A dash after the first character: a range (a leading one is a negative frame).
        const auto dash = part.find('-', 1);
        if (dash == std::string::npos) {
            out.push_back(std::stoi(part));
            continue;
        }
        const int a = std::stoi(part.substr(0, dash)), b = std::stoi(part.substr(dash + 1));
        if (b < a) throw std::runtime_error("a frame range goes up: " + spec);
        if (size_t(b - a) / size_t(step) > 1000000) throw std::runtime_error("too many frames: " + spec);
        for (int f = a; f <= b; f += step) out.push_back(f);
    }
    return out;
}

// Where a path's frame number goes: its last run of #, or a printf %d / %0Nd. {position, length, width}; length 0: none.
struct FrameSlot {
    size_t at = 0, length = 0;
    int width = 0;
};
inline FrameSlot frameSlot(const std::string& path) {
    FrameSlot s;
    // The file's name only: a folder may have a # in it.
    const size_t name = path.find_last_of("/\\") == std::string::npos ? 0 : path.find_last_of("/\\") + 1;
    const size_t hash = path.find_last_of('#');
    if (hash != std::string::npos && hash >= name) {
        size_t a = hash;
        while (a > name && path[a - 1] == '#') --a;
        s.at = a, s.length = hash + 1 - a, s.width = int(s.length);
        return s;
    }
    for (size_t p = path.rfind('%'); p != std::string::npos && p >= name; p = p ? path.rfind('%', p - 1) : std::string::npos) {
        size_t q = p + 1;
        int width = 0;
        while (q < path.size() && path[q] >= '0' && path[q] <= '9') width = width * 10 + (path[q] - '0'), ++q;
        if (q < path.size() && path[q] == 'd' && width <= 12) {
            s.at = p, s.length = q + 1 - p, s.width = width;
            return s;
        }
        if (!p) break;
    }
    return s;
}
inline bool hasFrameSlot(const std::string& path) { return frameSlot(path).length != 0; }
inline std::string padded(int frame, int width) {
    char b[32];
    std::snprintf(b, sizeof b, "%0*d", std::max(1, width), frame);
    return b;
}
// The path for a frame (a path without a slot: as it is).
inline std::string framePath(const std::string& path, int frame) {
    const FrameSlot s = frameSlot(path);
    if (!s.length) return path;
    return path.substr(0, s.at) + padded(frame, s.width) + path.substr(s.at + s.length);
}
// A picture's path for a frame of a job: its slot filled; without one, in a job of more than one frame, .NNNN before
// its extension (beauty.exr: beauty.0012.exr), so the frames do not write over each other.
inline std::string frameOutput(const std::string& path, int frame, bool manyFrames) {
    if (hasFrameSlot(path)) return framePath(path, frame);
    if (!manyFrames) return path;
    const size_t name = path.find_last_of("/\\") == std::string::npos ? 0 : path.find_last_of("/\\") + 1;
    const size_t dot = path.find_last_of('.');
    const size_t at = dot == std::string::npos || dot < name ? path.size() : dot;
    return path.substr(0, at) + "." + padded(frame, 4) + path.substr(at);
}

struct Job {
    std::string name;                  // as status shows it (the job file's, else the first scene's)
    std::vector<std::string> scenes;   // -in: the scene's files, each with a slot or not
    std::vector<std::string> deltas;   // --delta: a frame's changes, each with a slot (read on top of the frame before)
    std::string frames;                // as given ("1-24"); empty: the scene as it is, one frame
    std::string out;                   // the picture (a slot, or .NNNN added for more than one frame)
    std::vector<int> frameList() const { return parseFrames(frames); }
    // The files of the scene for a frame, and the frame's delta files (none for a job without frames).
    std::vector<std::string> scenesFor(int frame) const {
        std::vector<std::string> v;
        for (const auto& s : scenes) v.push_back(framePath(s, frame));
        return v;
    }
    std::vector<std::string> deltasFor(int frame) const {
        std::vector<std::string> v;
        for (const auto& d : deltas) v.push_back(framePath(d, frame));
        return v;
    }
    void check() const {
        if (scenes.empty()) throw std::runtime_error("a job without a scene (in)");
        if (out.empty()) throw std::runtime_error("a job without a picture to write (out)");
        if (!deltas.empty() && frames.empty()) throw std::runtime_error("deltas need frames (--frames)");
        for (const auto& d : deltas)
            if (!hasFrameSlot(d)) throw std::runtime_error("a delta names a file a frame (a run of #, or %04d): " + d);
        const auto list = frameList();
        if (!frames.empty() && list.empty()) throw std::runtime_error("no frames in " + frames);
    }
};

// A job file: a line a field, "in", "delta" (each again for more), "frames", "out", "name"; # comments.
inline std::string writeJob(const Job& j) {
    std::string t = "# a Rhizome job (moonray_xpu2 rhizome submit)\n";
    if (!j.name.empty()) t += "name " + j.name + "\n";
    for (const auto& s : j.scenes) t += "in " + s + "\n";
    for (const auto& d : j.deltas) t += "delta " + d + "\n";
    if (!j.frames.empty()) t += "frames " + j.frames + "\n";
    t += "out " + j.out + "\n";
    return t;
}
inline Job readJob(const std::string& text) {
    Job j;
    std::stringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        size_t a = line.find_first_not_of(" \t");
        if (a == std::string::npos || line[a] == '#') continue;
        const size_t sp = line.find_first_of(" \t", a);
        const std::string key = line.substr(a, sp == std::string::npos ? std::string::npos : sp - a);
        const size_t v = sp == std::string::npos ? std::string::npos : line.find_first_not_of(" \t", sp);
        const std::string value = v == std::string::npos ? "" : line.substr(v);
        if (key == "in") j.scenes.push_back(value);
        else if (key == "delta") j.deltas.push_back(value);
        else if (key == "frames") j.frames = value;
        else if (key == "out") j.out = value;
        else if (key == "name") j.name = value;
        else throw std::runtime_error("a job file's line not understood: " + line);
    }
    return j;
}

// The queue folder: its .job files waiting, by name; a job taken (moved to running\), and finished (to done\ or
// failed\, with what came of it).
namespace queue {
namespace fs = std::filesystem;
inline std::vector<fs::path> waiting(const fs::path& folder) {
    std::vector<fs::path> v;
    std::error_code e;
    for (fs::directory_iterator it(folder, e), end; !e && it != end; it.increment(e))
        if (it->is_regular_file(e) && it->path().extension() == ".job") v.push_back(it->path());
    std::sort(v.begin(), v.end());
    return v;
}
inline fs::path move(const fs::path& from, const fs::path& folder) {
    std::error_code e;
    fs::create_directories(folder, e);
    fs::path to = folder / from.filename();
    for (int k = 1; fs::exists(to, e); ++k) to = folder / (from.stem().string() + "-" + std::to_string(k) + from.extension().string());
    fs::rename(from, to, e);
    if (e) throw std::runtime_error("cannot move " + from.string() + " to " + folder.string() + ": " + e.message());
    return to;
}
// The next job: moved to running\ (none waiting: an empty path). Its file read into j.
inline fs::path take(const fs::path& folder, Job& j) {
    for (const fs::path& p : waiting(folder)) {
        std::ifstream in(p, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        const fs::path running = move(p, folder / "running");
        j = readJob(text);
        if (j.name.empty()) j.name = p.stem().string();
        return running;
    }
    return {};
}
inline void finish(const fs::path& running, const fs::path& folder, bool ok, const std::string& note) {
    {
        std::ofstream out(running, std::ios::binary | std::ios::app);
        out << "# " << (ok ? "done: " : "failed: ") << note << "\n";
    }
    move(running, folder / (ok ? "done" : "failed"));
}
// A job added to the queue: its file written whole, then named (a master never reads half of one), its name its
// time so the queue keeps the order jobs came in.
inline fs::path submit(const fs::path& folder, const Job& j) {
    j.check();
    std::error_code e;
    fs::create_directories(folder, e);
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    const long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_s(&tm, &t);
    char stamp[64];
    std::snprintf(stamp, sizeof stamp, "%04d%02d%02d-%02d%02d%02d-%03lld", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min,
                  tm.tm_sec, ms);
    // A number after the time, so jobs given in the same millisecond keep their order too.
    fs::path to;
    for (int k = 0;; ++k) {
        char n[16];
        std::snprintf(n, sizeof n, "-%03d.job", k);
        to = folder / (std::string(stamp) + n);
        if (!fs::exists(to, e) && !fs::exists(to.string() + ".part", e)) break;
    }
    const fs::path part = to.string() + ".part";
    {
        std::ofstream out(part, std::ios::binary | std::ios::trunc);
        out << writeJob(j);
        if (!out) throw std::runtime_error("cannot write " + part.string());
    }
    fs::rename(part, to, e);
    if (e) throw std::runtime_error("cannot name " + to.string() + ": " + e.message());
    return to;
}
}  // namespace queue

}  // namespace xpu2::interactive::rhizome
