// A Rhizome node (RHIZOME.md): its identity, its connections to the other members over TCP, and the protocol, as a worker
// or as the master integrator. One frame at a time.
//
// Joining: Hello (key, what it has, a challenge) -> Welcome (key, the proof for that challenge, a challenge back) ->
// Join (the proof for it). Each side checks the other's key hashes to its hash and its proof holds the rhizome key.
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
#include "rhizome_audit.h"
#include "rhizome_memory.h"
#include "rhizome_schedule.h"
#include "rhizome_swarm.h"
#include "rhizome_wire.h"
#include "rhizome_trust.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace xpu2::interactive::rhizome {

enum : uint16_t {
    TypeJoin = 15, TypePieceMoves = 16, TypePieceAsk = 17, TypePieceData = 18, TypeReassign = 19, TypeManifest = 20, TypeReady = 21,
    TypeEndorse = 22,  // master -> node: its endorsement (and the master's own, for a delegation)
    TypeRevoke = 23,   // a revocation, passed through the swarm
    TypeSuccession = 24,  // master: who takes over if it fails, in order, signed
    TypeReplica = 25,     // master -> deputy: a film it took in (the deputy keeps the picture and the ledger too)
    TypeTakeover = 26,    // a successor: it is the master now (the old master's signed succession as its right)
    TypeHeld = 27,        // a node -> its new master: the chunks it holds
    TypeReplicaState = 28,  // master -> a new deputy: the picture, counts and chunks back so far, and the caller's merge
};

// A node's renderer: the film of a cell's rectangle for samples [first, first + count).
using CellRenderer = std::function<Film(const uint32_t rect[4], uint32_t first, uint32_t count)>;

struct NodeOptions {
    std::string rhizomeKey;
    bool master = false;
    std::string masterAddress;  // a worker's way in: host:port
    Capabilities caps;
    uint64_t ramBytes = 0, vramBytes = 0;  // what it gives the memory pool
    CellRenderer render;
    PixelRenderer renderPixel;  // the master's, for audits
    // The master's: each film as it is taken into the picture (audited, the first of its chunk), under the node's
    // lock: the AOVs and aux outputs merged by the caller, which knows the scene's schema.
    std::function<void(const Film&)> onFilm;
    // Failover: the caller's merge (AOVs, aux) as bytes for a new deputy, taken in by it; and what the caller does when
    // this node takes over as master (it then waits for the picture and writes it).
    std::function<Bytes()> exportMerge;
    std::function<void(const Bytes&)> importMerge;
    std::function<void()> onTakeover;
    // A node's: the master's manifest (what the frame needs: the scene's files as pieces), as it comes; and whether
    // a node takes cells before it says it is ready (markReady), as the stand-in renderer's tests do.
    std::function<void(const Bytes&)> onManifest;
    bool readyAtJoin = true;
    // Where the node writes its status each second (rhizome status reads it; empty: nowhere), and what it says of the
    // job (the scene's name, set by the caller).
    std::string statusPath, job;
    // Signed approvals (rhizome_trust.h): this member's trust (none: the rhizome key alone admits), and, for a master,
    // whether it endorses each machine it meets on this run (trust first). The XPU2 build it runs (empty: not checked).
    std::shared_ptr<Trust> trust;
    bool trustFirst = false;
    std::string build;
    double heartbeat = 0.25, suspectAfter = 1.0, goneAfter = 2.0;
    bool masterRenders = true;  // the master takes cells too
    // Where it listens: an address of this machine (127.0.0.1 for a swarm on one machine, 0.0.0.0 for every network)
    // and a port (0: one the system picks), and the address the others are told to reach it at (empty: the bound one).
    std::string bind = "127.0.0.1", advertise;
    uint16_t port = 0;
    bool portFallback = false;  // its port taken: one the system picks (a node's or master's default port)
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

class RhizomeNode {
public:
    explicit RhizomeNode(NodeOptions o, std::unique_ptr<Identity> id = nullptr) : o_(std::move(o)), id_(id ? std::move(id) : Identity::make()) {
        members_.suspectAfter = o_.suspectAfter, members_.goneAfter = o_.goneAfter;
    }
    ~RhizomeNode() { stop(); }

    const Hash& hash() const { return id_->hash(); }
    std::string address() const { return address_; }

    // Listens on 127.0.0.1 (a port the system picks) and, as a worker, joins the master.
    void start() {
        listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(o_.port);
        inet_pton(AF_INET, o_.bind.c_str(), &a.sin_addr);
        if (::bind(listener_, (sockaddr*)&a, sizeof a) != 0 || listen(listener_, 64) != 0) {
            // Its port taken (another node or master on this machine, started at the same moment): one the system
            // picks, when it may (portFallback); discovery tells the master which.
            if (!o_.portFallback || o_.port == 0) throw std::runtime_error("cannot listen on " + o_.bind + ":" + std::to_string(o_.port));
            log("port " + std::to_string(o_.port) + " is taken here: another one is used");
            closesocket(listener_);
            listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            a.sin_port = 0;
            if (::bind(listener_, (sockaddr*)&a, sizeof a) != 0 || listen(listener_, 64) != 0)
                throw std::runtime_error("cannot listen on " + o_.bind);
        }
        int len = sizeof a;
        getsockname(listener_, (sockaddr*)&a, &len);
        address_ = (o_.advertise.empty() ? (o_.bind == "0.0.0.0" ? std::string("127.0.0.1") : o_.bind) : o_.advertise) + ":" +
                   std::to_string(ntohs(a.sin_port));
        o_.caps.address = address_;
        running_ = true;
        threads_.emplace_back([this] { acceptLoop(); });
        threads_.emplace_back([this] { loop(); });
        threads_.emplace_back([this] { work(); });
        threads_.emplace_back([this] { auditLoop(); });  // every node: one may become the master (failover)
        if (o_.master) {
            std::lock_guard<std::mutex> g(m_);
            pool_.addNode(hash(), o_.ramBytes, o_.vramBytes);
        } else {
            // Given a master: it may not listen yet (it reads the scene first) or may go and come back: tried until it
            // answers. Not given one (a headless node): it waits to be found.
            if (!o_.masterAddress.empty()) threads_.emplace_back([this] { reachMaster(); });
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
        {
            std::lock_guard<std::mutex> g(dialsMutex_);
            for (auto& t : dials_)
                if (t.joinable()) t.join();
            dials_.clear();
        }
        // Every connection, those adopted while the threads above ended too (a peer accepted just as this node
        // stopped, or reached by a dial): closed, then their threads joined.
        std::vector<ConnPtr> conns;
        {
            std::lock_guard<std::mutex> g(m_);
            conns = conns_;
        }
        for (auto& c : conns) c->close();
        for (auto& c : conns) {
            if (c->reader.joinable()) c->reader.join();
            if (c->writer.joinable()) c->writer.join();
        }
    }

    // Master: the members that have joined (itself not counted).
    size_t joined() {
        std::lock_guard<std::mutex> g(m_);
        return members_.working().size();
    }
    // Master: a frame with the members there now. importance: a value a rhizome cell (empty: flat).
    void beginFrame(const FrameShape& shape, const std::vector<float>& importance) {
        std::lock_guard<std::mutex> g(m_);
        std::vector<Node> nodes;
        if (o_.masterRenders) nodes.push_back(Node{hash(), o_.caps.rate, 0, false});
        for (const Member* mm : members_.working())
            if (o_.readyAtJoin || ready_.count(mm->hash)) nodes.push_back(Node{mm->hash, mm->caps.rate, std::max(0.0, mm->rtt), false});
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
        frameId_ = nowSeconds() * 1e6;
        announceSuccession();
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
    // Where the pool keeps a piece now (a copy, under the node's lock).
    std::vector<Copy> holdersOf(const Hash& piece) {
        std::lock_guard<std::mutex> g(m_);
        return pool_.holders(piece);
    }
    bool holds(const Hash& piece) {
        std::lock_guard<std::mutex> g(m_);
        return store_.count(piece) != 0;
    }
    uint64_t audits() {
        std::lock_guard<std::mutex> g(m_);
        return auditsDone_;
    }
    // Reaches a member at host:port (a master dialling a node discovery found, or one given by address); the usual
    // handshake follows. On a thread of its own: connecting may take a while.
    void dial(const std::string& address) {
        std::lock_guard<std::mutex> g(dialsMutex_);
        dials_.emplace_back([this, address] {
            if (running_) connectTo(address, false);
        });
    }
    // The machines waiting for approval (master).
    std::vector<Hash> waitingForApproval() {
        std::lock_guard<std::mutex> g(m_);
        std::vector<Hash> out;
        for (const auto& [h, w] : waiting_) out.push_back(h);
        return out;
    }
    bool isMember(const Hash& h) {
        std::lock_guard<std::mutex> g(m_);
        const Member* mm = members_.find(h);
        return mm && mm->state != State::Gone && !mm->quarantined;
    }
    // Whether this node is the master now (it may have taken over), who it takes as master, and its succession.
    bool isMaster() {
        std::lock_guard<std::mutex> g(m_);
        return o_.master;
    }
    Hash masterOf() {
        std::lock_guard<std::mutex> g(m_);
        return o_.master ? hash() : masterHash_;
    }
    std::vector<Hash> succession() {
        std::lock_guard<std::mutex> g(m_);
        return succession_;
    }
    // The deputy's copy: chunks of the frame it holds the films of.
    size_t replicaChunks() {
        std::lock_guard<std::mutex> g(m_);
        return replicaLedger_ ? size_t(plan_->shape.cells()) * plan_->shape.chunksPerCell() - replicaLedger_->outstanding() : 0;
    }
    // Master: what the frame needs (the scene's files, as pieces in the pool), to every node now and each one joining.
    void publishManifest(const Bytes& manifest) {
        std::lock_guard<std::mutex> g(m_);
        manifest_ = manifest;
        Writer w;
        w.bytes(manifest_);
        broadcast(Type(TypeManifest), w.b);
    }
    // A node: the pieces the pool placed here not landed yet, asked again every two seconds, of the next holder in turn
    // (the one first asked may not have it yet itself), the master's source once each holder has been asked twice.
    void askPlaced(double now) {
        for (auto it = placed_.begin(); it != placed_.end();) {
            if (store_.count(it->first)) {
                it = placed_.erase(it);
                continue;
            }
            auto& [when, tries] = it->second;
            if (now - when >= 2.0) {
                std::vector<Hash> from;
                for (const Hash& h : holders_[it->first])
                    if (h != hash() && h != masterHash_ && !members_.quarantined(h)) from.push_back(h);
                const Hash to = !from.empty() && tries < 2 * from.size() ? from[tries % from.size()] : masterHash_;
                Writer w;
                w.hash(it->first);
                sendTo(to, uint16_t(TypePieceAsk), w.b);
                when = now, ++tries;
            }
            ++it;
        }
    }
    // Master: the nodes that have said they are ready (their scene made).
    size_t ready() {
        std::lock_guard<std::mutex> g(m_);
        size_t n = 0;
        for (const Member* mm : members_.working()) n += ready_.count(mm->hash);
        return n;
    }
    // A node: its scene is made; it may be given cells.
    void markReady() {
        std::lock_guard<std::mutex> g(m_);
        sendTo(masterHash_, uint16_t(TypeReady), {});
    }
    // A node: the pieces asked for, each fetched from a node the pool placed it on (spread over them, the master's
    // source last), again from another every two seconds until it lands. False: not all within the time.
    bool fetch(const std::vector<Hash>& pieces, double seconds) {
        std::unique_lock<std::mutex> g(m_);
        const double end = nowSeconds() + seconds;
        std::map<Hash, std::pair<double, size_t>> asked;  // when, and how many times
        for (;;) {
            size_t missing = 0;
            const double now = nowSeconds();
            for (const Hash& p : pieces) {
                if (store_.count(p)) continue;
                ++missing;
                auto& a = asked[p];
                if (a.second && now - a.first < 2.0) continue;
                std::vector<Hash> from;
                for (const Hash& h : holders_[p])
                    if (h != hash() && h != masterHash_ && !members_.quarantined(h)) from.push_back(h);
                // Spread over the holders by the piece and the attempt; the master (the scene's source) once they fail.
                Hash to = masterHash_;
                if (!from.empty() && a.second < 2 * from.size()) to = from[(size_t(p[0]) + a.second) % from.size()];
                Writer w;
                w.hash(p);
                sendTo(to, uint16_t(TypePieceAsk), w.b);
                a = {now, a.second + 1};
            }
            if (!missing) return true;
            if (nowSeconds() > end || !running_) return false;
            cv_.wait_for(g, std::chrono::milliseconds(50));
        }
    }
    Bytes piece(const Hash& h) {
        std::lock_guard<std::mutex> g(m_);
        auto it = store_.find(h);
        return it == store_.end() ? Bytes{} : it->second;
    }

private:
    struct Conn {
        SOCKET s = INVALID_SOCKET;
        Hash peer{};
        bool known = false, joined = false;
        Hash challenge{};  // the one this side sent
        uint64_t lastSequence = 0;  // the last message's number taken on this connection
        // Outgoing frames, in the order they were numbered, sent by the connection's writer thread: no thread holding
        // the node's lock ever waits on the network (two nodes sending each other large pieces would otherwise each
        // block in send while the other's reader waits for its node's lock).
        std::mutex send;
        std::condition_variable sendCv;
        std::deque<Bytes> out;
        std::thread reader, writer;
        std::atomic<bool> open{true};
        void close() {
            if (open.exchange(false)) {
                shutdown(s, SD_BOTH);
                closesocket(s);
            }
            sendCv.notify_all();
        }
        void write() {
            for (;;) {
                Bytes frame;
                {
                    std::unique_lock<std::mutex> g(send);
                    sendCv.wait(g, [&] { return !out.empty() || !open; });
                    if (!open) return;
                    frame = std::move(out.front());
                    out.pop_front();
                }
                size_t at = 0;
                while (at < frame.size()) {
                    const int k = ::send(s, (const char*)frame.data() + at, int(std::min<size_t>(frame.size() - at, 1 << 20)), 0);
                    if (k <= 0) {
                        close();
                        return;
                    }
                    at += size_t(k);
                }
            }
        }
    };
    using ConnPtr = std::shared_ptr<Conn>;
    // A machine shaking hands: what it said of itself, kept until it is admitted.
    struct Pending {
        Bytes key;
        Capabilities caps;
        uint64_t ram = 0, vram = 0;
        bool isMaster = false;
        std::string build;
        std::vector<Bytes> shown;  // its endorsements
    };
    struct Waiting {
        ConnPtr conn;
        Pending info;
    };

    // RHIZOME_TRACE=1: every message taken or refused (heartbeats aside) in the log.
    static bool trace() {
        static const bool on = std::getenv("RHIZOME_TRACE") && std::getenv("RHIZOME_TRACE")[0] == '1';
        return on;
    }
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
        // Under the lock, as stop takes the connections: one adopted once the node is stopping is closed at once, and
        // every other has its threads before stop can see it.
        std::lock_guard<std::mutex> g(m_);
        if (!running_) {
            c->close();
            return c;
        }
        c->reader = std::thread([this, c] { read(c); });
        c->writer = std::thread([c] { c->write(); });
        conns_.push_back(c);
        return c;
    }
    void reachMaster() {
        bool said = false;
        while (running_) {
            bool connected = false;
            {
                std::lock_guard<std::mutex> g(m_);
                for (auto& c : conns_) connected = connected || (c->open && (!c->joined || c->peer == masterHash_));
            }
            if (!connected) {
                if (!connectTo(o_.masterAddress, true, !said)) said = true;
                else said = false;
            }
            for (int i = 0; i < 10 && running_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    // False: not reached (said in the log when quiet is false).
    bool connectTo(const std::string& address, bool toMaster, bool tell = true) {
        const auto colon = address.rfind(':');
        int port = -1;
        if (colon != std::string::npos && colon + 1 < address.size() &&
            address.find_first_not_of("0123456789", colon + 1) == std::string::npos && address.size() - colon - 1 <= 5)
            port = std::stoi(address.substr(colon + 1));
        if (port <= 0 || port > 65535) {
            if (tell) log("not an address (host:port): " + address);
            return false;
        }
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(uint16_t(port));
        inet_pton(AF_INET, address.substr(0, colon).c_str(), &a.sin_addr);
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (connect(s, (sockaddr*)&a, sizeof a) != 0) {
            closesocket(s);
            if (tell) log("cannot reach " + address + "; trying again");
            return false;
        }
        ConnPtr c = adopt(s);
        Hash challenge;
        randomBytes(challenge.data(), challenge.size());
        Writer w;
        w.bytes(id_->publicKey());
        o_.caps.write(w);
        w.u64(o_.ramBytes), w.u64(o_.vramBytes);
        w.hash(challenge);
        w.u8(o_.master ? 1 : 0);
        w.str(o_.build);
        const std::vector<Bytes> shown = o_.trust ? o_.trust->shown() : std::vector<Bytes>{};
        w.u32(uint32_t(shown.size()));
        for (const auto& s : shown) w.bytes(s);
        (void)toMaster;
        std::lock_guard<std::mutex> g(m_);
        c->challenge = challenge;
        sendOn(c, Type::Hello, w.b);
        return true;
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
                inbox_.push_back({c, std::move(frame), nowSeconds()});
            }
            cv_.notify_all();
        }
        c->close();
    }
    void sendOn(const ConnPtr& c, uint16_t type, const Bytes& payload) {
        if (!c || !c->open) return;
        // Numbered under the connection's lock: two threads sending to one peer send in the order they number, and
        // the peer refuses a number at or below the last it took (a message played again).
        std::lock_guard<std::mutex> g(c->send);
        const Bytes frame = encode(*id_, Type(type), ++sequence_, payload);
        if (trace() && type != uint16_t(Type::Heartbeat) && type != uint16_t(Type::Echo))
            log("sent type " + std::to_string(type) + " to " + shortHex(c->peer) + ", " + std::to_string(payload.size()) + " bytes" +
                (payload.size() >= 6 ? " [" + hex(payload.data(), 6) + "]" : std::string()));
        c->out.push_back(std::move(frame));
        c->sendCv.notify_one();
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
        double nextBeat = 0, nextTrust = 0;
        std::unique_lock<std::mutex> g(m_);
        while (running_) {
            cv_.wait_for(g, std::chrono::milliseconds(20), [&] { return !inbox_.empty() || !self_.empty() || !running_; });
            while (!self_.empty()) {
                Bytes f = std::move(self_.front());
                self_.pop_front();
                auto m = peek(f);
                if (m) handle(nullptr, f, *m, nowSeconds());
            }
            while (!inbox_.empty()) {
                Arrived a = std::move(inbox_.front());
                inbox_.pop_front();
                auto m = peek(a.frame);
                if (!m) {
                    a.conn->close();
                    continue;
                }
                handle(a.conn, a.frame, *m, a.at);
            }
            const double now = nowSeconds();
            if (now >= nextBeat) {
                nextBeat = now + o_.heartbeat;
                Writer w;
                w.f64(now);
                broadcast(Type::Heartbeat, w.b);
                for (const Hash& h : members_.sweep(now)) {
                    if (!o_.master && h == masterHash_) masterGone(now);
                    else {
                        const bool successor = std::find(succession_.begin(), succession_.end(), h) != succession_.end();
                        lose(h, "silent");
                        if (successor && o_.master) announceSuccession();
                    }
                }
                takeOverDone(now);
                if (o_.master && !takeoverMessage_.empty() && now - takeoverAt_ < 10.0) broadcast(Type(TypeTakeover), takeoverMessage_);
                askPlaced(now);
                if (now >= nextTrust) {
                    nextTrust = now + 1.0;
                    reviewTrust(now);
                    writeStatus(now);
                }
            }
        }
    }

    void handle(const ConnPtr& c, const Bytes& frame, const Message& m, double now) {
        const uint16_t type = uint16_t(m.type);
        // The handshake: taken before the sender is a member, each checked against the key it carries.
        if (m.type == Type::Hello || m.type == Type::Welcome || type == TypeJoin) {
            handshake(c, frame, m, now);
            return;
        }
        if (c) {
            if (!c->joined || !members_.accept(frame, m, now, &c->lastSequence)) {
                if (trace()) log("refused type " + std::to_string(type) + " from " + shortHex(m.sender) + (c->joined ? " (not accepted: sequence " +
                                 std::to_string(m.sequence) + ")" : " (not joined)"));
                return;
            }
        }
        if (trace() && type != uint16_t(Type::Heartbeat) && type != uint16_t(Type::Echo))
            log("took type " + std::to_string(type) + " from " + shortHex(m.sender) + ", " + std::to_string(m.payload.size()) + " bytes" +
                (m.payload.size() >= 6 ? " [" + hex(m.payload.data(), 6) + "]" : std::string()));
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
                if (h != hash() && !members_.find(h) && hash() < h && !caps.address.empty() && dialing_.insert(h).second) {
                    // Dialled once: a member list that comes again while this dial shakes hands does not dial twice.
                    g_unlock([&] { connectTo(caps.address, false); });
                }
            }
            break;
        }
        case uint16_t(Type::Assign): {
            auto p = std::make_unique<Plan>();
            if (readPlan(r, *p)) {
                replicaLedger_.reset();  // it holds the plan it was made of
                succession_.clear();
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
                auto& known = holders_[piece];
                if (std::find(known.begin(), known.end(), to) == known.end()) known.push_back(to);
                if (to == hash() && !store_.count(piece)) {
                    Writer w;
                    w.hash(piece);
                    sendTo(from == Hash{} ? masterHash_ : from, uint16_t(TypePieceAsk), w.b);
                    placed_[piece] = {nowSeconds(), 1};  // asked again until it lands (askPlaced)
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
            if (sha256(data) == piece) {  // a piece is its own check
                store_[piece] = std::move(data);
                cv_.notify_all();
            }
            break;
        }
        case TypeManifest:
            manifest_ = r.bytes();
            if (o_.onManifest && !r.bad) {
                const Bytes copy = manifest_;
                g_unlock([&] { o_.onManifest(copy); });
            }
            break;
        case TypeEndorse:
            if (o_.trust && m.sender == masterHash_) {
                const Bytes e = r.bytes();
                std::vector<Bytes> chain(std::min<uint32_t>(r.u32(), 64));
                for (auto& s : chain) s = r.bytes();
                if (!r.bad && o_.trust->take(e, m.sender, chain)) log("endorsed by " + shortHex(m.sender));
            }
            break;
        case TypeRevoke:
            if (o_.trust) {
                const Bytes v = r.bytes();
                if (auto who = o_.trust->takeRevocation(v)) {
                    if (*who == hash()) log("this machine has been revoked");
                    else if (!members_.quarantined(*who)) {
                        log("revoked " + shortHex(*who));
                        refuse(*who, "revoked");
                    }
                }
            }
            break;
        case TypeSuccession:
            if (!o_.master && m.sender == masterHash_) {
                const Bytes body = r.bytes(), sig = r.bytes();
                const Member* mm = members_.find(m.sender);
                if (!r.bad && mm && verify(mm->publicKey, body.data(), body.size(), sig)) {
                    Reader br(body);
                    if (br.u32() == 0x31535A52u && br.hash() == m.sender) {
                        br.u64();
                        std::vector<Hash> list(std::min<uint32_t>(br.u32(), 16));
                        for (auto& h : list) h = br.hash();
                        if (!br.bad) {
                            succession_ = list;
                            successionBody_ = body;
                            successionSig_ = sig;
                        }
                    }
                }
            }
            break;
        case TypeReplica:
            if (m.sender == masterHash_) takeReplica(r);
            break;
        case TypeReplicaState:
            if (m.sender == masterHash_) takeReplicaState(r);
            break;
        case TypeTakeover:
            takeTakeover(m.sender, r, now);
            break;
        case TypeHeld:
            if (o_.master) heldReports_[m.sender] = readChunks(r);
            break;
        case TypeReady:
            if (o_.master) {
                ready_.insert(m.sender);
                log("ready " + shortHex(m.sender));
            }
            break;
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

    // The handshake (both sides alike): Hello (key, what it has, memory it gives, a challenge, whether it is a master,
    // its build, its endorsements) -> Welcome (the same, with the proof for the challenge and a challenge back) -> Join
    // (the proof for that). Then decide: a build other than this one's is refused (its samples would differ); one
    // that is not approved (Trust::admits) is refused by a node, held by a master until an approver endorses it (or
    // endorsed at once with trustFirst); the rest admitted, and a master then welcomes it into the swarm.
    void handshake(const ConnPtr& c, const Bytes& frame, const Message& m, double now) {
        if (!c) return;
        Reader r(m.payload);
        auto readShown = [&] {
            std::vector<Bytes> shown(std::min<uint32_t>(r.u32(), 64));
            for (auto& s : shown) s = r.bytes();
            return shown;
        };
        auto writeOwn = [&](Writer& w) {
            w.u8(o_.master ? 1 : 0);
            w.str(o_.build);
            const std::vector<Bytes> shown = o_.trust ? o_.trust->shown() : std::vector<Bytes>{};
            w.u32(uint32_t(shown.size()));
            for (const auto& s : shown) w.bytes(s);
        };
        if (m.type == Type::Hello) {
            Pending info;
            info.key = r.bytes();
            info.caps = Capabilities::read(r);
            info.ram = r.u64(), info.vram = r.u64();
            const Hash challenge = r.hash();
            info.isMaster = r.u8() != 0;
            info.build = r.str();
            info.shown = readShown();
            if (r.bad || !authentic(frame, m, info.key) || members_.quarantined(m.sender) || revoked(m.sender)) return c->close();
            c->peer = m.sender;
            Hash back;
            randomBytes(back.data(), back.size());
            c->challenge = back;
            pendingKey_[c.get()] = std::move(info);
            Writer w;
            w.bytes(id_->publicKey());
            o_.caps.write(w);
            w.u64(o_.ramBytes), w.u64(o_.vramBytes);
            w.hash(membershipProof(o_.rhizomeKey, challenge, hash()));
            w.hash(back);
            writeOwn(w);
            sendOn(c, Type::Welcome, w.b);
        } else if (m.type == Type::Welcome) {
            Pending info;
            info.key = r.bytes();
            info.caps = Capabilities::read(r);
            info.ram = r.u64(), info.vram = r.u64();
            const Hash proof = r.hash(), back = r.hash();
            info.isMaster = r.u8() != 0;
            info.build = r.str();
            info.shown = readShown();
            if (r.bad || !authentic(frame, m, info.key) || proof != membershipProof(o_.rhizomeKey, c->challenge, m.sender)) {
                log("refused " + shortHex(m.sender) + ": no proof of the rhizome key");
                return c->close();
            }
            c->peer = m.sender;
            Writer w;
            w.hash(membershipProof(o_.rhizomeKey, back, hash()));
            sendOn(c, uint16_t(TypeJoin), w.b);
            decide(c, m.sender, info, now);
        } else {
            auto it = pendingKey_.find(c.get());
            if (it == pendingKey_.end() || c->peer != m.sender) return c->close();
            const Hash proof = r.hash();
            const Pending info = it->second;
            pendingKey_.erase(it);
            if (!authentic(frame, m, info.key) || proof != membershipProof(o_.rhizomeKey, c->challenge, m.sender)) {
                log("refused " + shortHex(m.sender) + ": no proof of the rhizome key");
                return c->close();
            }
            decide(c, m.sender, info, now);
        }
    }
    void decide(const ConnPtr& c, const Hash& h, const Pending& info, double now) {
        if (!o_.build.empty() && !info.build.empty() && info.build != o_.build) {
            log("refused " + shortHex(h) + ": it runs XPU2 build " + info.build.substr(0, 12) + ", this machine " + o_.build.substr(0, 12) +
                " (a sample would differ)");
            return c->close();
        }
        if (revoked(h)) {
            log("refused " + shortHex(h) + ": revoked");
            return c->close();
        }
        if (o_.trust && !o_.trust->admits(h, info.shown)) {
            if (!o_.master) {
                log("refused " + shortHex(h) + ": not approved by an approver this machine trusts");
                return c->close();
            }
            if (o_.trustFirst) {
                o_.trust->endorse(h);
                log("approved " + shortHex(h) + " (trust first)");
            } else {
                if (!waiting_.count(h))
                    log("waiting for approval: " + hex(h) + " at " + info.caps.address + "; approve it with: moonray_xpu2 rhizome approve " + hex(h));
                waiting_[h] = Waiting{c, info};
                return;
            }
        }
        admitMember(c, h, info, now);
    }
    // Where a member is reached: the address it gave, or, where it gave this machine's loopback (it listens on every
    // adapter), the address its connection comes from, with its port: what the members are told to dial.
    std::string reachableAt(const ConnPtr& c, const std::string& given) const {
        const auto colon = given.rfind(':');
        if (colon == std::string::npos) return given;
        const std::string host = given.substr(0, colon);
        if (host != "127.0.0.1" && host != "0.0.0.0") return given;
        sockaddr_in peer{};
        int len = sizeof peer;
        if (getpeername(c->s, (sockaddr*)&peer, &len) != 0) return given;
        char ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof ip);
        return std::string(ip) + given.substr(colon);
    }
    void admitMember(const ConnPtr& c, const Hash& h, const Pending& info0, double now) {
        Pending info = info0;
        info.caps.address = reachableAt(c, info.caps.address);
        if (!members_.introduce(h, info.key, info.caps, now, info.isMaster)) return c->close();
        if (Member* mm = members_.find(h)) mm->lastSequence = 0;
        c->peer = h, c->known = true, c->joined = true;
        if (info.isMaster && !o_.master) masterHash_ = h;
        given_[h] = {info.ram, info.vram};
        dialing_.erase(h);
        log("joined " + shortHex(h));
        if (o_.master) welcome(c, h, info);
    }
    // Master: a member admitted: its endorsement (one an approver issued it), the revocations known, the members to it
    // and it to them (each pair dials by the lower hash), its place in the memory pool, the manifest.
    void welcome(const ConnPtr& c, const Hash& h, const Pending& info) {
        if (o_.trust) {
            if (auto e = o_.trust->issued(h)) {
                Writer w;
                w.bytes(*e);
                const std::vector<Bytes> chain = o_.trust->shown();
                w.u32(uint32_t(chain.size()));
                for (const auto& s : chain) w.bytes(s);
                sendOn(c, uint16_t(TypeEndorse), w.b);
            }
            for (const Bytes& v : o_.trust->revocations()) {
                Writer w;
                w.bytes(v);
                sendOn(c, uint16_t(TypeRevoke), w.b);
            }
        }
        Writer w;
        const auto list = members_.working();
        w.u32(uint32_t(list.size()));
        for (const Member* mm : list) w.hash(mm->hash), mm->caps.write(w);
        broadcast(Type::Members, w.b);
        pool_.addNode(h, info.ram, info.vram);
        if (!manifest_.empty()) {
            Writer mw;
            mw.bytes(manifest_);
            sendOn(c, uint16_t(TypeManifest), mw.b);
        }
        pool_.joined();  // pieces short of copies, placed now there is room
        sendPlacements(nullptr);
    }
    bool revoked(const Hash& h) const { return o_.trust && o_.trust->revokedAfter(h, 0); }
    // Master, each second: a machine waiting for approval admitted once an approver has endorsed it (the approve
    // command writes the endorsement into this master's folder); revocations the folder has gained passed on and
    // enforced. A node: revocations its folder has gained (its own blocks) enforced.
    void reviewTrust(double now) {
        if (!o_.trust) return;
        for (auto it = waiting_.begin(); it != waiting_.end();) {
            if (!it->second.conn->open) {
                it = waiting_.erase(it);
                continue;
            }
            if (revoked(it->first)) {
                log("refused " + shortHex(it->first) + ": revoked");
                it->second.conn->close();
                it = waiting_.erase(it);
                continue;
            }
            if (o_.trust->issued(it->first) || o_.trust->admits(it->first, it->second.info.shown)) {
                log("approved " + shortHex(it->first));
                const Waiting w = it->second;
                const Hash h = it->first;
                it = waiting_.erase(it);
                admitMember(w.conn, h, w.info, now);
                continue;
            }
            ++it;
        }
        for (const Hash& h : o_.trust->refreshRevocations()) {
            log("revoked " + shortHex(h));
            if (o_.trust->approver())
                for (const Bytes& v : o_.trust->revocations()) {
                    Writer w;
                    w.bytes(v);
                    broadcast(Type(TypeRevoke), w.b);
                }
            refuse(h, "revoked");
        }
    }
    // status.json (rhizome status): this machine, the frame, each member's role, state, tier, rate, distance and progress,
    // and the memory pool, as this machine sees them. Written whole each second, then renamed into place.
    static std::string jsonText(const std::string& s) {
        std::string o = "\"";
        for (char c : s) {
            if (c == '"' || c == '\\') o += '\\';
            if (static_cast<unsigned char>(c) >= 0x20) o += c;
        }
        return o + "\"";
    }
    void writeStatus(double now) {
        if (o_.statusPath.empty()) return;
        auto num = [](double v) {
            char t[32];
            std::snprintf(t, sizeof t, "%.6g", std::isfinite(v) ? v : 0.0);
            return std::string(t);
        };
        std::string j = "{\n";
        j += "  \"hash\": " + jsonText(hex(hash())) + ",\n";
        j += "  \"role\": " + jsonText(o_.master ? "master" : "node") + ",\n";
        j += "  \"address\": " + jsonText(address_) + ",\n";
        j += "  \"approver\": " + std::string(o_.trust && o_.trust->approver() ? "true" : "false") + ",\n";
        j += "  \"master\": " + jsonText(o_.master ? hex(hash()) : masterHash_ == Hash{} ? "" : hex(masterHash_)) + ",\n";
        j += "  \"job\": " + jsonText(o_.job) + ",\n";
        j += "  \"time\": " + num(now) + ",\n";
        // The frame: chunks in all and back (the master's ledger), the chunks this machine holds still.
        size_t total = 0, back = 0;
        if (plan_) total = size_t(plan_->shape.cells()) * plan_->shape.chunksPerCell();
        if (ledger_) back = total - ledger_->outstanding();
        j += "  \"frame\": {\"running\": " + std::string(plan_ && !frameDone_ ? "true" : "false") + ", \"done\": " +
             std::string(frameDone_ ? "true" : "false") + ", \"chunks\": " + std::to_string(total) + ", \"back\": " + std::to_string(back) +
             ", \"held\": " + std::to_string(holder_ ? holder_->remaining() : 0) + ", \"audits\": " + std::to_string(auditsDone_) +
             (plan_ ? ", \"width\": " + std::to_string(plan_->shape.width) + ", \"height\": " + std::to_string(plan_->shape.height) +
                          ", \"spp\": " + std::to_string(plan_->shape.spp) + ", \"cells\": " + std::to_string(plan_->shape.cells())
                    : std::string()) +
             "},\n";
        // The memory pool (the master knows its whole layout).
        j += "  \"pool\": {\"ram\": " + std::to_string(pool_.total(Tier::Ram)) + ", \"vram\": " + std::to_string(pool_.total(Tier::Vram)) +
             ", \"pieces\": " + std::to_string(sourcePieces_.size()) + ", \"held\": " + std::to_string(store_.size()) + "},\n";
        j += "  \"members\": [\n";
        auto row = [&](const Hash& h, const std::string& addr, const std::string& role, const std::string& state, double rate, double rtt) {
            const int i = plan_ ? plan_->nodeIndex(h) : -1;
            std::string tier = i < 0 ? "" : plan_->nodes[size_t(i)].fast ? "fast" : "slow";
            const uint64_t done = i >= 0 && size_t(i) < byNode_.size() ? byNode_[size_t(i)] : 0;
            return "    {\"hash\": " + jsonText(hex(h)) + ", \"address\": " + jsonText(addr) + ", \"role\": " + jsonText(role) +
                   ", \"state\": " + jsonText(state) + ", \"tier\": " + jsonText(tier) + ", \"rate\": " + num(rate) + ", \"rtt\": " + num(rtt) +
                   ", \"chunks\": " + std::to_string(done) + ", \"ram\": " + std::to_string(pool_.used(h, Tier::Ram)) +
                   ", \"vram\": " + std::to_string(pool_.used(h, Tier::Vram)) + ", \"ramGiven\": " + std::to_string(pool_.capacity(h, Tier::Ram)) +
                   ", \"vramGiven\": " + std::to_string(pool_.capacity(h, Tier::Vram)) + "}";
        };
        std::vector<std::string> rows;
        // The same as text (rhizome status prints it): a line a member.
        char line[320];
        std::string text;
        auto mb = [](uint64_t v) {
            char t[24];
            std::snprintf(t, sizeof t, "%.1f", double(v) / (1 << 20));
            return std::string(t);
        };
        auto textRow = [&](const Hash& h, const std::string& addr, const std::string& role, const std::string& state, double rate, double rtt) {
            const int i = plan_ ? plan_->nodeIndex(h) : -1;
            const char* tier = i < 0 ? "-" : plan_->nodes[size_t(i)].fast ? "fast" : "slow";
            const uint64_t done = i >= 0 && size_t(i) < byNode_.size() ? byNode_[size_t(i)] : 0;
            char rt[24], rr[24];
            std::snprintf(rt, sizeof rt, rtt >= 0 ? "%.1f ms" : "-", rtt * 1000.0);
            std::snprintf(rr, sizeof rr, rate > 0 ? "%.2f" : "-", rate / 1e6);
            std::snprintf(line, sizeof line, "%-12s  %-21s  %-22s  %-8s  %-5s  %10s  %9s  %6llu  %s of %s MB\n", shortHex(h).c_str(), addr.c_str(),
                          role.c_str(), state.c_str(), tier, rr, rt, (unsigned long long)done, mb(pool_.used(h, Tier::Ram) + pool_.used(h, Tier::Vram)).c_str(),
                          mb(pool_.capacity(h, Tier::Ram) + pool_.capacity(h, Tier::Vram)).c_str());
            text += line;
        };
        rows.push_back(row(hash(), address_, o_.master ? "master" : "node (this machine)", "alive", o_.caps.rate, 0));
        textRow(hash(), address_, o_.master ? "master (this machine)" : "node (this machine)", "alive", o_.caps.rate, -1);
        for (const Member* mm : members_.all()) {
            const char* st = mm->quarantined ? "refused" : mm->state == State::Alive ? "alive" : mm->state == State::Suspect ? "suspect" : "gone";
            std::string role = mm->quarantined ? (revoked(mm->hash) ? "revoked" : "quarantined") : mm->master || mm->hash == masterHash_ ? "master" : "node";
            if (o_.trust && o_.trust->blocked(mm->hash)) role = "blocked (here)";
            rows.push_back(row(mm->hash, mm->caps.address, role, st, mm->caps.rate, mm->rtt));
            textRow(mm->hash, mm->caps.address, role, st, mm->caps.rate, mm->rtt);
        }
        for (const auto& [h, w] : waiting_) {
            rows.push_back(row(h, w.info.caps.address, "waiting for approval", "held", w.info.caps.rate, -1));
            textRow(h, w.info.caps.address, "waiting for approval", "held", w.info.caps.rate, -1);
        }
        for (size_t i = 0; i < rows.size(); ++i) j += rows[i] + (i + 1 < rows.size() ? ",\n" : "\n");
        j += "  ]\n}\n";
        const std::string temp = o_.statusPath + ".part";
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            out << j;
        }
        std::error_code e;
        std::filesystem::rename(temp, o_.statusPath, e);
        // The text: this machine, the frame, the pool, then the members.
        std::string head;
        std::snprintf(line, sizeof line, "This machine  %s  %s%s   %s\n", hex(hash()).c_str(), o_.master ? "master" : "node",
                      o_.trust && o_.trust->approver() ? ", approver" : "", address_.c_str());
        head += line;
        if (!o_.master && masterHash_ != Hash{}) {
            std::snprintf(line, sizeof line, "Master        %s\n", hex(masterHash_).c_str());
            head += line;
        }
        if (plan_) {
            std::snprintf(line, sizeof line, "Frame         %s %ux%u, %u spp   %s   %zu of %zu chunks back (%.0f%%), %llu films audited\n", o_.job.c_str(),
                          plan_->shape.width, plan_->shape.height, plan_->shape.spp, frameDone_ ? "done" : "rendering", back, total,
                          total ? 100.0 * double(back) / double(total) : 0.0, (unsigned long long)auditsDone_);
            head += line;
        } else {
            head += "Frame         none (waiting for a job)\n";
        }
        std::snprintf(line, sizeof line, "Memory pool   %zu pieces of the scene, %s MB given in RAM and %s MB in card memory across the swarm; %zu held here\n",
                      sourcePieces_.size(), mb(pool_.total(Tier::Ram)).c_str(), mb(pool_.total(Tier::Vram)).c_str(), store_.size());
        head += line;
        // Failover: who takes over, in order, and the deputy's copy of the frame.
        if (!succession_.empty()) {
            std::string order;
            for (const auto& h : succession_) order += (order.empty() ? "" : " -> ") + shortHex(h);
            std::snprintf(line, sizeof line, "Succession    %s\n", order.c_str());
            head += line;
            if (o_.master) {
                std::snprintf(line, sizeof line, "Deputy        %s   sent %llu films of the %zu taken in\n", shortHex(succession_[0]).c_str(),
                              (unsigned long long)replicasSent_, back);
                head += line;
            } else if (succession_[0] == hash()) {
                const size_t copy = replicaLedger_ ? total - replicaLedger_->outstanding() : 0;
                std::snprintf(line, sizeof line, "Deputy        this machine   holds the films of %zu of %zu chunks, the picture so far and its AOVs\n", copy, total);
                head += line;
            }
        }
        if (tookOverFrom_ != Hash{} || (o_.master && !takeoverMessage_.empty())) {
            head += "Master        this machine since it took over (the master before it failed)\n";
        }
        head += "\nHash          Address                Role                    State     Tier   Rate Ms/s        RTT  Chunks  Pool\n";
        const std::string textPath = o_.statusPath.substr(0, o_.statusPath.size() - 5) + ".txt";
        {
            std::ofstream out(textPath + ".part", std::ios::binary | std::ios::trunc);
            out << head << text;
        }
        std::filesystem::rename(textPath + ".part", textPath, e);
    }
    // A member refused from now on (quarantined, revoked or blocked): its connections closed, its work given out.
    void refuse(const Hash& h, const char* why) {
        members_.quarantine(h);
        waiting_.erase(h);
        for (auto& cc : conns_)
            if (cc->peer == h) cc->close();
        lose(h, why);
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
                U32 rect[4];
                plan.shape.rect(chunk->cell, rect);
                const U32 first = plan.shape.first(chunk->index), count = plan.shape.count(chunk->index);
                g.unlock();
                const double began = nowSeconds();
                Film f;
                try {
                    f = o_.render(rect, first, count);
                } catch (const std::exception& e) {
                    log(std::string("a chunk failed: ") + e.what());
                    g.lock();
                    holder_->receive({*chunk});  // kept, for a steal or the master's reassignment if this node goes
                    cv_.wait_for(g, std::chrono::milliseconds(200));
                    continue;
                }
                const double seconds = nowSeconds() - began;
                Writer w;
                w.u32(chunk->cell), w.u32(chunk->index);
                for (int k = 0; k < 4; ++k) w.u32(f.rect[k]);
                w.u32(f.first), w.u32(f.count);
                w.array(f.sums);
                w.array(f.counts);
                w.f64(seconds);
                w.array(f.odd);
                w.array(f.aovs);
                w.array(f.aovDepths);
                g.lock();
                // To the master as it is now (one may have taken over while the film was rendering).
                sendTo(o_.master ? hash() : masterHash_, Type::Contribution, w.b);
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
    static void writeFilm(Writer& w, const Chunk& c, const Film& f, double seconds) {
        w.u32(c.cell), w.u32(c.index);
        for (int k = 0; k < 4; ++k) w.u32(f.rect[k]);
        w.u32(f.first), w.u32(f.count);
        w.array(f.sums);
        w.array(f.counts);
        w.f64(seconds);
        w.array(f.odd);
        w.array(f.aovs);
        w.array(f.aovDepths);
    }
    static bool readFilm(Reader& r, Chunk& c, Film& f, double& seconds) {
        c.cell = r.u32(), c.index = r.u32();
        for (int k = 0; k < 4; ++k) f.rect[k] = r.u32();
        f.first = r.u32(), f.count = r.u32();
        f.sums = r.array<float>();
        f.counts = r.array<uint32_t>();
        seconds = r.f64();
        f.odd = r.array<float>();
        f.aovs = r.array<double>();
        f.aovDepths = r.array<float>();
        return !r.bad;
    }
    void contribution(const Hash& from, Reader& r) {
        if (!o_.master || !ledger_ || frameDone_) return;
        Chunk c;
        Film f;
        double seconds = 0;
        if (!readFilm(r, c, f, seconds)) return;
        // The node's rate, measured: the next frame's plan uses it.
        if (seconds > 0 && from != hash()) members_.rate(from, double(f.width()) * f.height() * f.count / seconds);
        if (seconds > 0 && from == hash()) o_.caps.rate = o_.caps.rate > 0 ? 0.7 * o_.caps.rate + 0.3 * double(f.width()) * f.height() * f.count / seconds
                                                                             : double(f.width()) * f.height() * f.count / seconds;
        U32 rect[4];
        plan_->shape.rect(c.cell, rect);
        bool shaped = f.first == plan_->shape.first(c.index) && f.count == plan_->shape.count(c.index);
        for (int k = 0; k < 4; ++k) shaped = shaped && f.rect[k] == rect[k];
        if (ledger_->returned(c)) return;  // rendered twice after a loss: the first film stands
        if (shaped && from != hash()) {
            // Audited on a thread of its own: rendering pixels again takes time the network loop must not lose.
            auditQueue_.push_back(Audit{from, c, std::move(f)});
            cv_.notify_all();
            return;
        }
        settle(from, c, f, shaped, Verdict{});
    }
    struct Audit {
        Hash from{};
        Chunk chunk;
        Film film;
    };
    // The audit thread: each film's picks rendered again with the state let go, then settled.
    void auditLoop() {
        std::unique_lock<std::mutex> g(m_);
        while (running_) {
            if (auditQueue_.empty()) {
                cv_.wait_for(g, std::chrono::milliseconds(20));
                continue;
            }
            Audit a = std::move(auditQueue_.front());
            auditQueue_.pop_front();
            if (!ledger_ || frameDone_ || members_.quarantined(a.from) || ledger_->returned(a.chunk)) continue;
            g.unlock();
            Verdict v;
            try {
                v = auditor_.check(a.from, a.film, o_.renderPixel);
            } catch (const std::exception& e) {
                g.lock();
                log(std::string("an audit could not render: ") + e.what() + "; the film is taken unchecked");
                settle(a.from, a.chunk, a.film, true, Verdict{});
                continue;
            }
            g.lock();
            ++auditsDone_;
            if (ledger_ && !frameDone_) settle(a.from, a.chunk, a.film, true, v);
        }
    }
    // A film's end: added once, or its node quarantined.
    void settle(const Hash& from, const Chunk& c, const Film& f, bool shaped, const Verdict& v) {
        if (!shaped || !v.passed) {
            log("audit failed for " + shortHex(from) + " (" + std::to_string(v.failed) + " of " + std::to_string(v.checked) +
                " pixels, chunk " + std::to_string(c.cell) + "/" + std::to_string(c.index) + "):" + v.detail + " quarantined");
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
        addFilm(film_, counts_, f);
        if (o_.onFilm) o_.onFilm(f);
        replicate(from, c, f);
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
    void addFilm(std::vector<double>& film, std::vector<uint32_t>& counts, const Film& f) const {
        const U32 W = plan_->shape.width;
        for (U32 y = f.rect[1]; y < f.rect[3]; ++y)
            for (U32 x = f.rect[0]; x < f.rect[2]; ++x) {
                const size_t i = size_t(y - f.rect[1]) * f.width() + (x - f.rect[0]), o = size_t(y) * W + x;
                if (o >= counts.size()) continue;
                for (int k = 0; k < 4; ++k) film[o * 4 + k] += f.sums[i * 4 + k];
                counts[o] += f.counts[i];
            }
    }

    // ---- failover (RHIZOME.md, 3b)
    // Master: who takes over if it fails: the frame's members by rate, fastest first (three at most), signed. With
    // approvals each is endorsed as an approver, so a master after a takeover approves and revokes in its turn.
    void announceSuccession() {
        if (!o_.master || !plan_) return;
        std::vector<std::pair<double, Hash>> by;
        for (const auto& n : plan_->nodes) {
            if (n.hash == hash()) continue;
            const Member* mm = members_.find(n.hash);
            if (!mm || mm->state == State::Gone || mm->quarantined) continue;
            by.push_back({-mm->caps.rate, n.hash});
        }
        std::sort(by.begin(), by.end());
        std::vector<Hash> list;
        for (size_t i = 0; i < by.size() && i < 3; ++i) list.push_back(by[i].second);
        const Hash oldDeputy = succession_.empty() ? Hash{} : succession_[0];
        succession_ = list;
        Writer body;
        body.u32(0x31535A52u);  // "RZS1"
        body.hash(hash());
        body.u64(uint64_t(frameId_));
        body.u32(uint32_t(list.size()));
        for (const auto& h : list) body.hash(h);
        successionBody_ = body.b;
        successionSig_ = id_->sign(body.b.data(), body.b.size());
        Writer w;
        w.bytes(successionBody_);
        w.bytes(successionSig_);
        broadcast(Type(TypeSuccession), w.b);
        if (o_.trust && o_.trust->approver())
            for (const Hash& h : list) {
                if (approverEndorsed_.count(h)) continue;
                approverEndorsed_.insert(h);
                const Bytes e = o_.trust->endorse(h, Role::Approver);
                Writer ew;
                ew.bytes(e);
                const std::vector<Bytes> chain = o_.trust->shown();
                ew.u32(uint32_t(chain.size()));
                for (const auto& s : chain) ew.bytes(s);
                sendTo(h, uint16_t(TypeEndorse), ew.b);
            }
        // A new deputy has none of the films taken in so far: the picture as it stands, the chunks back, the merge.
        if (!list.empty() && list[0] != oldDeputy && ledger_) {
            Writer s;
            s.array(film_);
            s.array(counts_);
            std::vector<U32> back;
            for (U32 c = 0; c < plan_->shape.cells(); ++c)
                for (U32 k = 0; k < plan_->shape.chunksPerCell(); ++k)
                    if (ledger_->returned(Chunk{c, k})) back.push_back(c), back.push_back(k);
            s.array(back);
            s.array(byNode_);
            s.bytes(o_.exportMerge ? o_.exportMerge() : Bytes{});
            sendTo(list[0], uint16_t(TypeReplicaState), s.b);
            log("deputy " + shortHex(list[0]) + " given the frame so far (" + std::to_string(back.size() / 2) + " chunks)");
        }
        std::string names;
        for (const auto& h : list) names += (names.empty() ? "" : " -> ") + shortHex(h);
        log("succession: " + (names.empty() ? std::string("none") : names));
    }
    // Master: a film taken in, to the deputy.
    void replicate(const Hash& from, const Chunk& c, const Film& f) {
        if (!o_.master || succession_.empty()) return;
        Writer w;
        w.hash(from);
        writeFilm(w, c, f, 0.0);
        sendTo(succession_[0], uint16_t(TypeReplica), w.b);
        ++replicasSent_;
    }
    void makeReplica() {
        if (replicaLedger_ || !plan_) return;
        replicaLedger_ = std::make_unique<Ledger>(*plan_);
        replicaFilm_.assign(size_t(plan_->shape.width) * plan_->shape.height * 4, 0.0);
        replicaCounts_.assign(size_t(plan_->shape.width) * plan_->shape.height, 0);
        replicaBy_.assign(plan_->nodes.size(), 0);
    }
    // The deputy: a film the master took in, kept as the master keeps it.
    void takeReplica(Reader& r) {
        if (o_.master || !plan_) return;
        const Hash from = r.hash();
        Chunk c;
        Film f;
        double seconds = 0;
        if (!readFilm(r, c, f, seconds)) return;
        makeReplica();
        if (!replicaLedger_->arrived(c)) return;
        addFilm(replicaFilm_, replicaCounts_, f);
        const int who = plan_->nodeIndex(from);
        if (who >= 0) ++replicaBy_[size_t(who)];
        if (o_.onFilm) o_.onFilm(f);
    }
    void takeReplicaState(Reader& r) {
        if (o_.master || !plan_) return;
        replicaLedger_.reset();
        makeReplica();
        replicaFilm_ = r.array<double>();
        replicaCounts_ = r.array<uint32_t>();
        const auto back = r.array<U32>();
        replicaBy_ = r.array<uint64_t>();
        const Bytes merge = r.bytes();
        if (r.bad || replicaFilm_.size() != size_t(plan_->shape.width) * plan_->shape.height * 4) {
            replicaLedger_.reset();
            return;
        }
        for (size_t i = 0; i + 1 < back.size(); i += 2) replicaLedger_->arrived(Chunk{back[i], back[i + 1]});
        if (o_.importMerge) o_.importMerge(merge);
        log("deputy: the frame so far taken (" + std::to_string(back.size() / 2) + " chunks)");
    }
    // A node: its master gone. The first successor of the signed list still here takes over: this node, or one it
    // waits to hear from (its takeover is checked as it comes).
    void masterGone(double now) {
        if (o_.master || succession_.empty() || !plan_ || frameDone_) return;
        for (const Hash& h : succession_) {
            if (h == hash()) {
                takeOver(now);
                return;
            }
            const Member* mm = members_.find(h);
            if (mm && mm->state != State::Gone && !mm->quarantined) {
                log("master " + shortHex(masterHash_) + " gone: waiting for " + shortHex(h) + " to take over");
                return;
            }
        }
        log("master " + shortHex(masterHash_) + " gone, and no successor is here");
    }
    void takeOver(double now) {
        const Hash old = masterHash_;
        log("master " + shortHex(old) + " gone: taking over as master");
        o_.master = true;
        o_.masterRenders = true;
        masterHash_ = hash();
        makeReplica();
        ledger_ = std::move(replicaLedger_);
        film_ = std::move(replicaFilm_);
        counts_ = std::move(replicaCounts_);
        byNode_ = std::move(replicaBy_);
        byNode_.resize(plan_->nodes.size(), 0);
        frameDone_ = false;
        auditor_ = Auditor();
        tookOverFrom_ = old;
        takeoverAt_ = now;
        heldReports_.clear();
        // The memory pool again, from the members here and the scene this node holds whole.
        pool_ = Pool();
        pool_.addNode(hash(), o_.ramBytes, o_.vramBytes);
        for (const Member* mm : members_.working())
            if (mm->hash != old) {
                auto it = given_.find(mm->hash);
                pool_.addNode(mm->hash, it == given_.end() ? 0 : it->second.first, it == given_.end() ? 0 : it->second.second);
            }
        sourcePieces_.clear();
        for (const auto& [h, data] : store_) {
            Piece pc{h, data.size(), Tier::Ram};
            sourcePieces_[h] = pc;
            pool_.add(pc);
        }
        sendPlacements(nullptr);
        Writer w;
        w.hash(old);
        w.bytes(successionBody_);
        w.bytes(successionSig_);
        takeoverMessage_ = w.b;
        broadcast(Type(TypeTakeover), takeoverMessage_);
        // The others report what they hold; what no one holds and did not come back is given out again (takeOverDone).
        if (o_.onTakeover) {
            auto f = o_.onTakeover;
            g_unlock([&] { f(); });
        }
    }
    // A node: a successor says it is the master now. Taken when the old master signed the succession it shows, it is
    // in it, and every successor before it is gone here.
    void takeTakeover(const Hash& from, Reader& r, double now) {
        const Hash old = r.hash();
        const Bytes body = r.bytes(), sig = r.bytes();
        if (r.bad || o_.master || from == old) return;
        if (masterHash_ == from) {
            sendHeld(from);  // the takeover again (it repeats it a while): the report again, in case it was lost
            return;
        }
        const Member* om = members_.find(old);
        if (!om || !verify(om->publicKey, body.data(), body.size(), sig)) {
            log("refused a takeover by " + shortHex(from) + ": its succession is not the old master's");
            return;
        }
        Reader br(body);
        if (br.u32() != 0x31535A52u || br.hash() != old) return;
        br.u64();
        const uint32_t n = br.u32();
        bool ok = false;
        for (uint32_t i = 0; i < n && !br.bad; ++i) {
            const Hash h = br.hash();
            if (h == from) {
                ok = true;
                break;
            }
            if (h == hash()) return;  // this node is ahead of it: it takes over itself, as its master went (masterGone)
            const Member* mm = members_.find(h);
            if (mm && mm->state != State::Gone && !mm->quarantined) return;  // an earlier successor is still here
        }
        if (!ok || (om->state != State::Gone && now - om->lastHeard < o_.goneAfter)) return;
        log("master " + shortHex(from) + " took over from " + shortHex(old));
        masterHash_ = from;
        if (Member* mm = members_.find(from)) mm->master = true;
        sendHeld(from);
    }
    void sendHeld(const Hash& to) {
        Writer w;
        writeChunks(w, holder_ ? holder_->held() : std::vector<Chunk>{});
        sendTo(to, uint16_t(TypeHeld), w.b);
    }
    // The new master, two seconds on: the chunks neither back nor held by anyone who reported, given out again.
    void takeOverDone(double now) {
        if (!o_.master || tookOverFrom_ == Hash{} || now - takeoverAt_ < 2.0 || !ledger_ || frameDone_) return;
        tookOverFrom_ = Hash{};
        std::set<Chunk> held;
        if (holder_)
            for (const auto& c : holder_->held()) held.insert(c);
        for (const auto& [h, cs] : heldReports_)
            for (const auto& c : cs) held.insert(c);
        std::vector<Chunk> orphans;
        for (U32 c = 0; c < plan_->shape.cells(); ++c)
            for (U32 k = 0; k < plan_->shape.chunksPerCell(); ++k)
                if (!ledger_->returned(Chunk{c, k}) && !held.count(Chunk{c, k})) orphans.push_back(Chunk{c, k});
        // To the reporting members by the plan's owner where it reported, else to this master.
        std::map<Hash, std::vector<Chunk>> to;
        for (const auto& c : orphans) {
            const int owner = plan_->owner[c.cell];
            const Hash h = owner >= 0 ? plan_->nodes[size_t(owner)].hash : hash();
            to[heldReports_.count(h) || h == hash() ? h : hash()].push_back(c);
        }
        for (const auto& [h, cs] : to) {
            if (h == hash()) {
                if (holder_) holder_->receive(cs);
                cv_.notify_all();
            } else {
                Writer w;
                w.hash(h);
                writeChunks(w, cs);
                sendTo(h, uint16_t(TypeReassign), w.b);
            }
        }
        for (const auto& [h, cs] : heldReports_) ledger_->granted(cs, plan_->nodeIndex(h));
        announceSuccession();
        log("takeover done: " + std::to_string(orphans.size()) + " chunks no one held given out again; " +
            std::to_string(ledger_->outstanding()) + " to come");
        if (ledger_->complete()) {
            frameDone_ = true;
            broadcast(Type::FrameDone, {});
            holder_.reset();
            doneCv_.notify_all();
            log("frame complete");
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


    NodeOptions o_;
    std::unique_ptr<Identity> id_;
    Winsock winsock_;
    SOCKET listener_ = INVALID_SOCKET;
    std::string address_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
    std::mutex dialsMutex_;
    std::vector<std::thread> dials_;
    std::mutex m_;
    std::condition_variable cv_, doneCv_;
    std::vector<ConnPtr> conns_;
    struct Arrived {
        ConnPtr conn;
        Bytes frame;
        double at = 0;  // when it was read: a member is heard then, however long the loop takes to come to it
    };
    std::deque<Arrived> inbox_;
    std::deque<Bytes> self_;
    std::map<Conn*, Pending> pendingKey_;
    std::map<Hash, Waiting> waiting_;  // master: admitted by the key, waiting for an approver's endorsement
    std::set<Hash> dialing_;           // peers this node has dialled (one connection a pair)
    // Failover: the signed succession (its list, body and signature), the deputy's copy of the master's frame, a
    // takeover's state, and what each member gives the pool.
    std::vector<Hash> succession_;
    Bytes successionBody_, successionSig_, takeoverMessage_;
    double frameId_ = 0, takeoverAt_ = 0;
    std::unique_ptr<Ledger> replicaLedger_;
    std::vector<double> replicaFilm_;
    std::vector<uint32_t> replicaCounts_;
    std::vector<uint64_t> replicaBy_;
    uint64_t replicasSent_ = 0;
    Hash tookOverFrom_{};
    std::map<Hash, std::vector<Chunk>> heldReports_;
    std::set<Hash> approverEndorsed_;
    std::map<Hash, std::pair<uint64_t, uint64_t>> given_;
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
    std::deque<Audit> auditQueue_;
    Pool pool_;
    std::map<Hash, Bytes> store_;
    std::map<Hash, std::vector<Hash>> holders_;  // where the pool placed each piece (PieceMoves)
    std::map<Hash, std::pair<double, size_t>> placed_;  // pieces placed here not landed yet: when last asked, how often
    Bytes manifest_;
    std::set<Hash> ready_;
    std::map<Hash, Piece> sourcePieces_;
};

}  // namespace xpu2::interactive::rhizome
