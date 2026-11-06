// A Hive node (HIVE.md): its identity, its connections to the other members over TCP, and the protocol, as a worker
// or as the master integrator. One frame at a time.
//
// Joining: Hello (key, what it has, a challenge) -> Welcome (key, the proof for that challenge, a challenge back) ->
// Join (the proof for it). Each side checks the other's key hashes to its hash and its proof holds the hive key.
// The master then sends the members (Members), and every pair of workers connects in the same way, for steals and
// pieces. A frame: the master's plan (Assign) -> each node renders its chunks and sends their films (Contribution)
// -> a node out of work asks its neighbours (StealAsk -> StealGrant, the grant copied to the master) -> the master
// audits each film, adds it once, and gives a lost or quarantined node's chunks to the nearest (Reassign). The
// memory pool's copies move by the master's word (PieceMoves -> PieceAsk -> PieceData).
//
// Threads: an accepting thread, a reading thread a connection, the loop (messages, heartbeats, sweeps) and the work
// thread (rendering), the state under one mutex. The renderer is the caller's (a node's XPU2; a stand-in in tests).
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include "hive_audit.h"
#include "hive_memory.h"
#include "hive_schedule.h"
#include "hive_swarm.h"
#include "hive_wire.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace xpu2::interactive::hive {

enum : uint16_t { TypeJoin = 15, TypePieceMoves = 16, TypePieceAsk = 17, TypePieceData = 18, TypeReassign = 19 };

// A node's renderer: the film of a cell's rectangle for samples [first, first + count).
using CellRenderer = std::function<Film(const uint32_t rect[4], uint32_t first, uint32_t count)>;

struct NodeOptions {
    std::string hiveKey;
    bool master = false;
    std::string masterAddress;  // a worker's way in: host:port
    Capabilities caps;
    uint64_t ramBytes = 0, vramBytes = 0;  // what it gives the memory pool
    CellRenderer render;
    PixelRenderer renderPixel;  // the master's, for audits
    double heartbeat = 0.25, suspectAfter = 1.0, goneAfter = 2.0;
    bool masterRenders = true;  // the master takes cells too
    std::function<void(const std::string&)> log;
};

inline double nowSeconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

class Winsock {
public:
    Winsock() {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) != 0) throw std::runtime_error("Winsock did not start");
    }
    ~Winsock() { WSACleanup(); }
};

class HiveNode {
public:
    explicit HiveNode(NodeOptions o, std::unique_ptr<Identity> id = nullptr) : o_(std::move(o)), id_(id ? std::move(id) : Identity::make()) {
        members_.suspectAfter = o_.suspectAfter, members_.goneAfter = o_.goneAfter;
    }
    ~HiveNode() { stop(); }

    const Hash& hash() const { return id_->hash(); }
    std::string address() const { return address_; }

    // Listens on 127.0.0.1 (a port the system picks) and, as a worker, joins the master.
    void start() {
        listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = 0;
        inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (bind(listener_, (sockaddr*)&a, sizeof a) != 0 || listen(listener_, 64) != 0) throw std::runtime_error("cannot listen");
        int len = sizeof a;
        getsockname(listener_, (sockaddr*)&a, &len);
        address_ = "127.0.0.1:" + std::to_string(ntohs(a.sin_port));
        o_.caps.address = address_;
        running_ = true;
        threads_.emplace_back([this] { acceptLoop(); });
        threads_.emplace_back([this] { loop(); });
        threads_.emplace_back([this] { work(); });
        if (o_.master) {
            std::lock_guard<std::mutex> g(m_);
            pool_.addNode(hash(), o_.ramBytes, o_.vramBytes);
        } else {
            connectTo(o_.masterAddress, true);
        }
    }
    // Stops everything (a node leaving, or one that fails in a test: its peers then find it silent).
    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        closesocket(listener_);
        {
            std::lock_guard<std::mutex> g(m_);
            for (auto& c : conns_) c->close();
        }
        for (auto& t : threads_)
            if (t.joinable()) t.join();
        threads_.clear();
        std::vector<ConnPtr> conns;
        {
            std::lock_guard<std::mutex> g(m_);
            conns = conns_;
        }
        for (auto& c : conns)
            if (c->reader.joinable()) c->reader.join();
    }

    // Master: the members that have joined (itself not counted).
    size_t joined() {
        std::lock_guard<std::mutex> g(m_);
        return members_.working().size();
    }
    // Master: a frame with the members there now. importance: a value a hive cell (empty: flat).
    void beginFrame(const FrameShape& shape, const std::vector<float>& importance) {
        std::lock_guard<std::mutex> g(m_);
        std::vector<Node> nodes;
        if (o_.masterRenders) nodes.push_back(Node{hash(), o_.caps.rate, 0, false});
        for (const Member* mm : members_.working()) nodes.push_back(Node{mm->hash, mm->caps.rate, std::max(0.0, mm->rtt), false});
        plan_ = std::make_unique<Plan>(Plan::make(shape, nodes, importance));
        ledger_ = std::make_unique<Ledger>(*plan_);
        film_.assign(size_t(shape.width) * shape.height * 4, 0.0);
        counts_.assign(size_t(shape.width) * shape.height, 0);
        byNode_.assign(plan_->nodes.size(), 0);
        importantByFast_ = importantBySlow_ = 0;
        frameDone_ = false;
        Writer w;
        writePlan(w, *plan_);
        broadcast(Type::Assign, w.b);
        startPlan();
        log("frame: " + std::to_string(shape.cells()) + " cells, " + std::to_string(plan_->nodes.size()) + " nodes");
    }
    // Master: waits for the picture (false: timed out).
    bool waitFrame(double seconds) {
        std::unique_lock<std::mutex> g(m_);
        return doneCv_.wait_for(g, std::chrono::duration<double>(seconds), [&] { return frameDone_; });
    }
    // Master: the picture (sums a pixel) and counts.
    std::vector<double> film() {
        std::lock_guard<std::mutex> g(m_);
        return film_;
    }
    std::vector<uint32_t> counts() {
        std::lock_guard<std::mutex> g(m_);
        return counts_;
    }
    // Master: chunks each node's films brought (by the plan's node order), and the important cells' chunks by tier.
    std::vector<uint64_t> chunksByNode() {
        std::lock_guard<std::mutex> g(m_);
        return byNode_;
    }
    std::pair<uint64_t, uint64_t> importantChunks() {
        std::lock_guard<std::mutex> g(m_);
        return {importantByFast_, importantBySlow_};
    }
    const Plan* plan() const { return plan_.get(); }
    bool quarantined(const Hash& h) {
        std::lock_guard<std::mutex> g(m_);
        return members_.quarantined(h);
    }
    // The memory pool: pieces to keep across the swarm (master), and what this node holds of them.
    void addPiece(const Piece& p, const Bytes& data) {
        std::lock_guard<std::mutex> g(m_);
        store_[p.hash] = data;  // the master keeps the scene's pieces as their source
        sourcePieces_[p.hash] = p;
        pool_.add(p);
        sendPlacements(nullptr);
    }
    Pool& pool() { return pool_; }
    bool holds(const Hash& piece) {
        std::lock_guard<std::mutex> g(m_);
        return store_.count(piece) != 0;
    }
    uint64_t audits() {
        std::lock_guard<std::mutex> g(m_);
        return auditsDone_;
    }

private:
    struct Conn {
        SOCKET s = INVALID_SOCKET;
        Hash peer{};
        bool known = false, joined = false;
        Hash challenge{};  // the one this side sent
        std::mutex send;
        std::thread reader;
        std::atomic<bool> open{true};
        void close() {
            if (open.exchange(false)) {
                shutdown(s, SD_BOTH);
                closesocket(s);
            }
        }
    };
    using ConnPtr = std::shared_ptr<Conn>;

    void log(const std::string& s) {
        if (o_.log) o_.log((o_.master ? "master " : "node ") + shortHex(hash()) + ": " + s);
    }

    // ---- connections
    void acceptLoop() {
        while (running_) {
            SOCKET s = accept(listener_, nullptr, nullptr);
            if (s == INVALID_SOCKET) break;
            adopt(s);
        }
    }
    ConnPtr adopt(SOCKET s) {
        auto c = std::make_shared<Conn>();
        c->s = s;
        BOOL on = 1;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*)&on, sizeof on);
        {
            std::lock_guard<std::mutex> g(m_);
            conns_.push_back(c);
        }
        c->reader = std::thread([this, c] { read(c); });
        return c;
    }
    void connectTo(const std::string& address, bool toMaster) {
        const auto colon = address.rfind(':');
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(uint16_t(std::stoi(address.substr(colon + 1))));
        inet_pton(AF_INET, address.substr(0, colon).c_str(), &a.sin_addr);
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (connect(s, (sockaddr*)&a, sizeof a) != 0) {
            closesocket(s);
            log("cannot reach " + address);
            return;
        }
        ConnPtr c = adopt(s);
        Hash challenge;
        randomBytes(challenge.data(), challenge.size());
        Writer w;
        w.bytes(id_->publicKey());
        o_.caps.write(w);
        w.u64(o_.ramBytes), w.u64(o_.vramBytes);
        w.hash(challenge);
        w.u8(toMaster ? 1 : 0);
        std::lock_guard<std::mutex> g(m_);
        c->challenge = challenge;
        sendOn(c, Type::Hello, w.b);
    }
    static bool readAll(SOCKET s, unsigned char* p, size_t n) {
        while (n) {
            const int got = recv(s, (char*)p, int(std::min<size_t>(n, 1 << 20)), 0);
            if (got <= 0) return false;
            p += got, n -= size_t(got);
        }
        return true;
    }
    void read(ConnPtr c) {
        for (;;) {
            Bytes frame(HeaderBytes);
            if (!readAll(c->s, frame.data(), HeaderBytes)) break;
            const auto n = payloadLength(frame.data());
            if (!n) break;
            frame.resize(HeaderBytes + *n + SignatureBytes);
            if (!readAll(c->s, frame.data() + HeaderBytes, *n + SignatureBytes)) break;
            {
                std::lock_guard<std::mutex> g(m_);
                inbox_.push_back({c, std::move(frame)});
            }
            cv_.notify_all();
        }
        c->close();
    }
    void sendOn(const ConnPtr& c, uint16_t type, const Bytes& payload) {
        if (!c || !c->open) return;
        const Bytes frame = encode(*id_, Type(type), ++sequence_, payload);
        std::lock_guard<std::mutex> g(c->send);
        size_t at = 0;
        while (at < frame.size()) {
            const int k = ::send(c->s, (const char*)frame.data() + at, int(std::min<size_t>(frame.size() - at, 1 << 20)), 0);
            if (k <= 0) {
                c->close();
                return;
            }
            at += size_t(k);
        }
    }
    void sendOn(const ConnPtr& c, Type type, const Bytes& payload) { sendOn(c, uint16_t(type), payload); }
    ConnPtr connTo(const Hash& h) {
        for (auto& c : conns_)
            if (c->joined && c->peer == h && c->open) return c;
        return nullptr;
    }
    void sendTo(const Hash& h, uint16_t type, const Bytes& payload) {
        if (h == hash()) {
            // To itself (the master as a node): taken straight in.
            const Bytes frame = encode(*id_, Type(type), ++sequence_, payload);
            self_.push_back(frame);
            cv_.notify_all();
            return;
        }
        sendOn(connTo(h), type, payload);
    }
    void sendTo(const Hash& h, Type t, const Bytes& p) { sendTo(h, uint16_t(t), p); }
    void broadcast(Type t, const Bytes& p) {
        for (auto& c : conns_)
            if (c->joined && c->open && members_.find(c->peer) && !members_.quarantined(c->peer)) sendOn(c, t, p);
    }

    // ---- the loop
    void loop() {
        double nextBeat = 0;
        std::unique_lock<std::mutex> g(m_);
        while (running_) {
            cv_.wait_for(g, std::chrono::milliseconds(20), [&] { return !inbox_.empty() || !self_.empty() || !running_; });
            while (!self_.empty()) {
                Bytes f = std::move(self_.front());
                self_.pop_front();
                auto m = peek(f);
                if (m) handle(nullptr, f, *m);
            }
            while (!inbox_.empty()) {
                auto [c, f] = std::move(inbox_.front());
                inbox_.pop_front();
                auto m = peek(f);
                if (!m) {
                    c->close();
                    continue;
                }
                handle(c, f, *m);
            }
            const double now = nowSeconds();
            if (now >= nextBeat) {
                nextBeat = now + o_.heartbeat;
                Writer w;
                w.f64(now);
                broadcast(Type::Heartbeat, w.b);
                for (const Hash& h : members_.sweep(now)) lose(h, "silent");
            }
        }
    }

    void handle(const ConnPtr& c, const Bytes& frame, const Message& m) {
        const double now = nowSeconds();
        const uint16_t type = uint16_t(m.type);
        // The handshake: taken before the sender is a member, each checked against the key it carries.
        if (m.type == Type::Hello || m.type == Type::Welcome || type == TypeJoin) {
            handshake(c, frame, m, now);
            return;
        }
        if (c) {
            if (!c->joined || !members_.accept(frame, m, now)) return;
        }
        Reader r(m.payload);
        switch (type) {
        case uint16_t(Type::Heartbeat): {
            Writer w;
            w.f64(r.f64());
            sendOn(c, Type::Echo, w.b);
            break;
        }
        case uint16_t(Type::Echo):
            members_.roundTrip(m.sender, now - r.f64());
            break;
        case uint16_t(Type::Members): {
            const uint32_t n = r.u32();
            for (uint32_t i = 0; i < n && !r.bad; ++i) {
                const Hash h = r.hash();
                const Capabilities caps = Capabilities::read(r);
                // Workers connect to each other: the one with the lower hash dials, so each pair has one connection.
                if (h != hash() && !members_.find(h) && hash() < h && !caps.address.empty()) {
                    g_unlock([&] { connectTo(caps.address, false); });
                }
            }
            break;
        }
        case uint16_t(Type::Assign): {
            auto p = std::make_unique<Plan>();
            if (readPlan(r, *p)) {
                plan_ = std::move(p);
                frameDone_ = false;
                startPlan();
            }
            break;
        }
        case uint16_t(Type::StealAsk): {
            const int thief = plan_ ? plan_->nodeIndex(m.sender) : -1;
            Writer w;
            std::vector<Chunk> given;
            if (thief >= 0 && holder_) given = holder_->grant(thief);
            writeChunks(w, given);
            sendTo(m.sender, Type::StealGrant, w.b);
            if (!given.empty() && !o_.master) {
                // The master learns of the grant.
                Writer rw;
                rw.hash(m.sender);
                writeChunks(rw, given);
                sendTo(masterHash_, uint16_t(TypeReassign), rw.b);
            } else if (!given.empty() && ledger_) {
                ledger_->granted(given, thief);
            }
            break;
        }
        case uint16_t(Type::StealGrant): {
            const auto chunks = readChunks(r);
            asking_ = false;
            if (holder_ && !chunks.empty()) holder_->receive(chunks);
            else if (chunks.empty()) ++victim_;
            cv_.notify_all();
            break;
        }
        case uint16_t(TypeReassign): {
            const Hash to = r.hash();
            const auto chunks = readChunks(r);
            if (o_.master && m.sender != hash()) {
                // A node's grant to another (or to the master, which took it in from the grant), reported.
                if (ledger_ && plan_) ledger_->granted(chunks, plan_->nodeIndex(to));
            } else if (holder_ && to == hash()) {
                holder_->receive(chunks);  // a lost node's chunks, from the master
                cv_.notify_all();
            }
            break;
        }
        case uint16_t(Type::Contribution):
            contribution(m.sender, r);
            break;
        case uint16_t(Type::FrameDone):
            frameDone_ = true;
            holder_.reset();
            break;
        case uint16_t(Type::Quarantine):
            members_.quarantine(r.hash());
            break;
        case TypePieceMoves: {
            const uint32_t n = r.u32();
            for (uint32_t i = 0; i < n && !r.bad; ++i) {
                const Hash piece = r.hash(), from = r.hash(), to = r.hash();
                if (to == hash() && !store_.count(piece)) {
                    Writer w;
                    w.hash(piece);
                    sendTo(from == Hash{} ? masterHash_ : from, uint16_t(TypePieceAsk), w.b);
                }
            }
            break;
        }
        case TypePieceAsk: {
            const Hash piece = r.hash();
            auto it = store_.find(piece);
            if (it != store_.end()) {
                Writer w;
                w.hash(piece);
                w.bytes(it->second);
                sendTo(m.sender, uint16_t(TypePieceData), w.b);
            }
            break;
        }
        case TypePieceData: {
            const Hash piece = r.hash();
            Bytes data = r.bytes();
            if (sha256(data) == piece) store_[piece] = std::move(data);  // a piece is its own check
            break;
        }
        default:
            break;
        }
    }
    // Runs f with the state's lock let go (connecting blocks).
    template <class F>
    void g_unlock(F&& f) {
        m_.unlock();
        f();
        m_.lock();
    }

    void handshake(const ConnPtr& c, const Bytes& frame, const Message& m, double now) {
        if (!c) return;
        Reader r(m.payload);
        if (m.type == Type::Hello) {
            const Bytes key = r.bytes();
            const Capabilities caps = Capabilities::read(r);
            const uint64_t ram = r.u64(), vram = r.u64();
            const Hash challenge = r.hash();
            r.u8();
            if (r.bad || !authentic(frame, m, key) || members_.quarantined(m.sender)) return c->close();
            pendingKey_[c.get()] = {key, caps, ram, vram};
            c->peer = m.sender;
            Hash back;
            randomBytes(back.data(), back.size());
            c->challenge = back;
            Writer w;
            w.bytes(id_->publicKey());
            o_.caps.write(w);
            w.u64(o_.ramBytes), w.u64(o_.vramBytes);
            w.hash(membershipProof(o_.hiveKey, challenge, hash()));
            w.hash(back);
            w.u8(o_.master ? 1 : 0);
            sendOn(c, Type::Welcome, w.b);
        } else if (m.type == Type::Welcome) {
            const Bytes key = r.bytes();
            const Capabilities caps = Capabilities::read(r);
            const uint64_t ram = r.u64(), vram = r.u64();
            const Hash proof = r.hash(), back = r.hash();
            const bool isMaster = r.u8() != 0;
            if (r.bad || !authentic(frame, m, key) || proof != membershipProof(o_.hiveKey, c->challenge, m.sender)) {
                log("refused " + shortHex(m.sender) + ": no proof of the hive key");
                return c->close();
            }
            Writer w;
            w.hash(membershipProof(o_.hiveKey, back, hash()));
            sendOn(c, uint16_t(TypeJoin), w.b);
            admit(c, m.sender, key, caps, ram, vram, now, isMaster);
            if (isMaster) masterHash_ = m.sender;
        } else {
            auto it = pendingKey_.find(c.get());
            if (it == pendingKey_.end() || c->peer != m.sender) return c->close();
            const Hash proof = r.hash();
            const auto info = it->second;
            pendingKey_.erase(it);
            if (!authentic(frame, m, info.key) || proof != membershipProof(o_.hiveKey, c->challenge, m.sender)) {
                log("refused " + shortHex(m.sender) + ": no proof of the hive key");
                return c->close();
            }
            admit(c, m.sender, info.key, info.caps, info.ram, info.vram, now, false);
            if (o_.master) {
                // The members to the newcomer; the newcomer to the members (each dials by the lower hash).
                Writer w;
                const auto list = members_.working();
                w.u32(uint32_t(list.size()));
                for (const Member* mm : list) w.hash(mm->hash), mm->caps.write(w);
                broadcast(Type::Members, w.b);
                pool_.addNode(m.sender, info.ram, info.vram);
                pool_.joined();  // pieces short of copies, placed now there is room
                sendPlacements(nullptr);
            }
        }
    }
    void admit(const ConnPtr& c, const Hash& h, const Bytes& key, const Capabilities& caps, uint64_t, uint64_t, double now, bool isMaster) {
        if (!members_.introduce(h, key, caps, now, isMaster)) return c->close();
        if (Member* mm = members_.find(h)) mm->lastSequence = 0;
        c->peer = h, c->known = true, c->joined = true;
        log("joined " + shortHex(h));
    }

    // ---- frames
    void writePlan(Writer& w, const Plan& p) {
        w.u32(p.shape.width), w.u32(p.shape.height), w.u32(p.shape.cellSize), w.u32(p.shape.spp), w.u32(p.shape.chunkSamples);
        w.u32(uint32_t(p.nodes.size()));
        for (const auto& n : p.nodes) w.hash(n.hash), w.f64(n.rate), w.f64(n.rtt), w.u8(n.fast);
        w.array(p.owner);
        w.array(p.important);
        w.array(p.grantChunks);
        w.hash(hash());
    }
    bool readPlan(Reader& r, Plan& p) {
        p.shape.width = r.u32(), p.shape.height = r.u32(), p.shape.cellSize = r.u32(), p.shape.spp = r.u32(), p.shape.chunkSamples = r.u32();
        const uint32_t n = r.u32();
        for (uint32_t i = 0; i < n && !r.bad; ++i) {
            Node x;
            x.hash = r.hash(), x.rate = r.f64(), x.rtt = r.f64(), x.fast = r.u8() != 0;
            p.nodes.push_back(x);
        }
        p.owner = r.array<int>();
        p.important = r.array<unsigned char>();
        p.grantChunks = r.array<U32>();
        masterHash_ = r.hash();
        return !r.bad && p.owner.size() == p.shape.cells() && p.grantChunks.size() == p.nodes.size();
    }
    static void writeChunks(Writer& w, const std::vector<Chunk>& v) {
        w.u32(uint32_t(v.size()));
        for (const auto& c : v) w.u32(c.cell), w.u32(c.index);
    }
    static std::vector<Chunk> readChunks(Reader& r) {
        std::vector<Chunk> v(r.u32());
        for (auto& c : v) c.cell = r.u32(), c.index = r.u32();
        if (r.bad) v.clear();
        return v;
    }
    void startPlan() {
        const int self = plan_->nodeIndex(hash());
        holder_ = self >= 0 ? std::make_unique<Holder>(*plan_, self) : nullptr;
        victim_ = 0;
        asking_ = false;
        cv_.notify_all();
    }

    // The work thread: render the next chunk, send its film; out of work, ask the next victim.
    void work() {
        std::unique_lock<std::mutex> g(m_);
        while (running_) {
            if (!holder_ || frameDone_) {
                cv_.wait_for(g, std::chrono::milliseconds(20));
                continue;
            }
            const auto chunk = holder_->next();
            if (chunk) {
                const Plan plan = *plan_;
                const Hash master = o_.master ? hash() : masterHash_;
                U32 rect[4];
                plan.shape.rect(chunk->cell, rect);
                const U32 first = plan.shape.first(chunk->index), count = plan.shape.count(chunk->index);
                g.unlock();
                const Film f = o_.render(rect, first, count);
                Writer w;
                w.u32(chunk->cell), w.u32(chunk->index);
                for (int k = 0; k < 4; ++k) w.u32(f.rect[k]);
                w.u32(f.first), w.u32(f.count);
                w.array(f.sums);
                w.array(f.counts);
                g.lock();
                sendTo(master, Type::Contribution, w.b);
                continue;
            }
            // Out of work: ask the victims in turn, one at a time.
            if (!asking_) {
                const auto victims = holder_->victims();
                if (victim_ < victims.size()) {
                    asking_ = true;
                    askedAt_ = nowSeconds();
                    sendTo(plan_->nodes[size_t(victims[victim_])].hash, Type::StealAsk, {});
                } else {
                    victim_ = 0;  // everyone asked: wait a little, then again (a lost node's chunks may come)
                    cv_.wait_for(g, std::chrono::milliseconds(50));
                }
            } else if (nowSeconds() - askedAt_ > 0.5) {
                asking_ = false;  // a victim that does not answer
                ++victim_;
            }
            cv_.wait_for(g, std::chrono::milliseconds(5));
        }
    }

    // Master: a film. Audited, then added once.
    void contribution(const Hash& from, Reader& r) {
        if (!o_.master || !ledger_ || frameDone_) return;
        Chunk c;
        c.cell = r.u32(), c.index = r.u32();
        Film f;
        for (int k = 0; k < 4; ++k) f.rect[k] = r.u32();
        f.first = r.u32(), f.count = r.u32();
        f.sums = r.array<float>();
        f.counts = r.array<uint32_t>();
        if (r.bad) return;
        U32 rect[4];
        plan_->shape.rect(c.cell, rect);
        bool shaped = f.first == plan_->shape.first(c.index) && f.count == plan_->shape.count(c.index);
        for (int k = 0; k < 4; ++k) shaped = shaped && f.rect[k] == rect[k];
        if (ledger_->returned(c)) return;  // rendered twice after a loss: the first film stands
        Verdict v;
        if (shaped && from != hash()) {
            v = auditor_.check(from, f, o_.renderPixel);
            ++auditsDone_;
        }
        if (!shaped || !v.passed) {
            log("audit failed for " + shortHex(from) + " (" + std::to_string(v.failed) + " of " + std::to_string(v.checked) +
                " pixels): quarantined");
            members_.quarantine(from);
            Writer w;
            w.hash(from);
            broadcast(Type::Quarantine, w.b);
            for (auto& cc : conns_)
                if (cc->peer == from) cc->close();
            lose(from, "quarantined");
            return;
        }
        if (!ledger_->arrived(c)) return;
        const int who = plan_->nodeIndex(from);
        if (who >= 0) {
            ++byNode_[size_t(who)];
            if (plan_->important[c.cell]) (plan_->nodes[size_t(who)].fast ? importantByFast_ : importantBySlow_)++;
        }
        const U32 W = plan_->shape.width;
        for (U32 y = f.rect[1]; y < f.rect[3]; ++y)
            for (U32 x = f.rect[0]; x < f.rect[2]; ++x) {
                const size_t i = size_t(y - f.rect[1]) * f.width() + (x - f.rect[0]), o = size_t(y) * W + x;
                for (int k = 0; k < 4; ++k) film_[o * 4 + k] += f.sums[i * 4 + k];
                counts_[o] += f.counts[i];
            }
        if (ledger_->complete()) {
            frameDone_ = true;
            broadcast(Type::FrameDone, {});
            holder_.reset();
            doneCv_.notify_all();
            log("frame complete");
        }
    }

    // Master: a node gone or quarantined: its chunks to their heirs, its memory pool copies made again elsewhere.
    void lose(const Hash& h, const char* why) {
        if (!o_.master) return;
        log(std::string("lost ") + shortHex(h) + " (" + why + ")");
        const auto moves = pool_.fail(h);
        sendPlacements(&moves);
        if (!ledger_ || frameDone_) return;
        const int dead = plan_->nodeIndex(h);
        if (dead < 0) return;
        std::vector<bool> alive(plan_->nodes.size(), false);
        for (size_t i = 0; i < plan_->nodes.size(); ++i) {
            const Hash& n = plan_->nodes[i].hash;
            const Member* mm = members_.find(n);
            alive[i] = n == hash() ? o_.masterRenders : (mm && mm->state != State::Gone && !mm->quarantined);
        }
        for (const auto& [to, chunks] : ledger_->lost(dead, alive)) {
            Writer w;
            w.hash(plan_->nodes[size_t(to)].hash);
            writeChunks(w, chunks);
            sendTo(plan_->nodes[size_t(to)].hash, uint16_t(TypeReassign), w.b);
            log("gave " + std::to_string(chunks.size()) + " chunks to " + shortHex(plan_->nodes[size_t(to)].hash));
        }
    }
    // Master: where copies are to be made (all of the pool's placements when moves is null, each to its node).
    void sendPlacements(const std::vector<Move>* moves) {
        if (!o_.master) return;
        std::vector<Move> all;
        if (!moves) {
            for (const auto& [h, p] : sourcePieces_)
                for (const auto& c : pool_.holders(h)) all.push_back(Move{h, hash(), c});
            moves = &all;
        }
        Writer w;
        w.u32(uint32_t(moves->size()));
        for (const auto& mv : *moves) w.hash(mv.piece), w.hash(mv.from), w.hash(mv.to.node);
        broadcast(Type(TypePieceMoves), w.b);
    }

    struct Pending {
        Bytes key;
        Capabilities caps;
        uint64_t ram = 0, vram = 0;
    };

    NodeOptions o_;
    std::unique_ptr<Identity> id_;
    Winsock winsock_;
    SOCKET listener_ = INVALID_SOCKET;
    std::string address_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    std::vector<ConnPtr> conns_;
    std::deque<std::pair<ConnPtr, Bytes>> inbox_;
    std::deque<Bytes> self_;
    std::map<Conn*, Pending> pendingKey_;
    std::atomic<uint64_t> sequence_{0};
    Members members_;
    Hash masterHash_{};
    std::unique_ptr<Plan> plan_;
    std::unique_ptr<Ledger> ledger_;
    std::unique_ptr<Holder> holder_;
    size_t victim_ = 0;
    bool asking_ = false;
    double askedAt_ = 0;
    bool frameDone_ = false;
    std::vector<double> film_;
    std::vector<uint32_t> counts_;
    std::vector<uint64_t> byNode_;
    uint64_t importantByFast_ = 0, importantBySlow_ = 0, auditsDone_ = 0;
    Auditor auditor_;
    Pool pool_;
    std::map<Hash, Bytes> store_;
    std::map<Hash, Piece> sourcePieces_;
};

}  // namespace xpu2::interactive::hive
