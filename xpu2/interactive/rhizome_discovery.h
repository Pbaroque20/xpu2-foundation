// Rhizome's discovery (RHIZOME.md, 2): a master finds the headless nodes on its networks, a direct cable between two
// machines among them. A node listens for probes on a UDP port (DiscoveryPort); the master sends one, about every
// second, to the broadcast address of each of its network adapters (each IPv4 address's subnet: a cable with no router
// on it has the address Windows gives itself, 169.254.x.y, and its subnet's broadcast reaches the other end) and to
// 255.255.255.255. A node answers the master straight back with its hash, the TCP port it listens on, and a keyed hash
// under the rhizome key: an answer without it is not taken, so only a member of the rhizome is found. The master then reaches
// the node at the address the answer came from and the usual handshake (rhizome_node.h) follows.
//
// Beyond broadcast (plan 18): a router does not pass a broadcast, nor do most VPNs, so the master also probes machines
// one address at a time: the ones it reached before (remembered in its folder, known.txt), and with --scan whole ranges
// (10.8.0.0/24, or each adapter's own subnet up to 4,096 addresses) at a steady rate, a few hundred a second. A machine
// it cannot probe at all (another site's network) is still reached by address: --node host:port.
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include "rhizome_identity.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace xpu2::interactive::rhizome {

constexpr uint16_t DiscoveryPort = 47101;
// Nodes on one machine (several, or a master's machine that also runs one) each take the first free port of
// DiscoveryPort and the next ones, and the master probes them all: a probe sent to one address (plan 18, beyond
// broadcast) reaches only one of the sockets sharing a port, where a broadcast reaches each.
constexpr uint16_t DiscoveryPorts = 4;
constexpr uint32_t ProbeMagic = 0x505A4852u, AnswerMagic = 0x415A4852u;  // "RHZP", "RHZA"

// The broadcast address of each IPv4 address of each adapter that is up (loopback's left out), and the limited one.
inline std::vector<in_addr> broadcastAddresses() {
    std::vector<in_addr> out;
    ULONG bytes = 16 << 10;
    std::vector<unsigned char> buffer;
    ULONG got = ERROR_BUFFER_OVERFLOW;
    for (int tries = 0; tries < 3 && got == ERROR_BUFFER_OVERFLOW; ++tries) {
        buffer.resize(bytes);
        got = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr,
                                   reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()), &bytes);
    }
    std::set<uint32_t> seen;
    if (got == NO_ERROR)
        for (auto* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()); a; a = a->Next) {
            if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (auto* u = a->FirstUnicastAddress; u; u = u->Next) {
                if (u->Address.lpSockaddr->sa_family != AF_INET) continue;
                const uint32_t ip = ntohl(reinterpret_cast<sockaddr_in*>(u->Address.lpSockaddr)->sin_addr.s_addr);
                const unsigned bits = std::min<unsigned>(u->OnLinkPrefixLength, 32);
                const uint32_t mask = bits == 0 ? 0u : ~0u << (32 - bits);
                const uint32_t b = (ip & mask) | ~mask;
                if (seen.insert(b).second) {
                    in_addr x;
                    x.s_addr = htonl(b);
                    out.push_back(x);
                }
            }
        }
    in_addr all;
    all.s_addr = htonl(INADDR_BROADCAST);
    if (seen.insert(INADDR_BROADCAST).second) out.push_back(all);
    return out;
}

// A range of IPv4 addresses (host order): first and how many.
struct Range {
    uint32_t first = 0, count = 0;
};
// "10.8.0.0/24" (its hosts: the network and broadcast addresses left out from /30 down), "10.8.0.7" (that one), at most
// /16. False: not one.
inline bool parseRange(const std::string& text, Range& r) {
    const size_t slash = text.find('/');
    const std::string host = text.substr(0, slash);
    in_addr a{};
    if (inet_pton(AF_INET, host.c_str(), &a) != 1) return false;
    unsigned bits = 32;
    if (slash != std::string::npos) {
        try {
            size_t used = 0;
            bits = unsigned(std::stoul(text.substr(slash + 1), &used));
            if (used != text.size() - slash - 1) return false;
        } catch (...) {
            return false;
        }
    }
    if (bits < 16 || bits > 32) return false;
    const uint32_t mask = bits == 32 ? ~0u : ~0u << (32 - bits);
    const uint32_t base = ntohl(a.s_addr) & mask;
    const uint64_t size = uint64_t(1) << (32 - bits);
    r.first = size >= 4 ? base + 1 : base;
    r.count = uint32_t(size >= 4 ? size - 2 : size);
    return true;
}
// Each adapter's own subnet that is up (loopback's and the self-assigned 169.254/16's left out), as ranges of their
// hosts, no wider than widest bits (a /20: 4,094 addresses).
inline std::vector<Range> adapterSubnets(unsigned widest = 20) {
    std::vector<Range> out;
    ULONG bytes = 16 << 10;
    std::vector<unsigned char> buffer;
    ULONG got = ERROR_BUFFER_OVERFLOW;
    for (int tries = 0; tries < 3 && got == ERROR_BUFFER_OVERFLOW; ++tries) {
        buffer.resize(bytes);
        got = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr,
                                   reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()), &bytes);
    }
    std::set<uint32_t> seen;
    if (got == NO_ERROR)
        for (auto* a = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data()); a; a = a->Next) {
            if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
            for (auto* u = a->FirstUnicastAddress; u; u = u->Next) {
                if (u->Address.lpSockaddr->sa_family != AF_INET) continue;
                const uint32_t ip = ntohl(reinterpret_cast<sockaddr_in*>(u->Address.lpSockaddr)->sin_addr.s_addr);
                if ((ip >> 16) == 0xA9FEu) continue;  // 169.254/16: a cable's, reached by its broadcast
                const unsigned bits = std::max(widest, std::min<unsigned>(u->OnLinkPrefixLength, 32));
                const uint32_t mask = bits == 32 ? ~0u : ~0u << (32 - bits);
                if (!seen.insert((ip & mask) | bits).second) continue;
                in_addr x;
                x.s_addr = htonl(ip);
                char t[INET_ADDRSTRLEN] = {};
                inet_ntop(AF_INET, &x, t, sizeof t);
                Range r;
                if (parseRange(std::string(t) + "/" + std::to_string(bits), r)) out.push_back(r);
            }
        }
    return out;
}
// What a master probes beyond its adapters' broadcasts (plan 18).
struct ProbeOptions {
    bool broadcast = true;            // every adapter's broadcast and the limited one (--no-broadcast: not)
    std::vector<uint32_t> hosts;      // machines probed each round (remembered ones; host order)
    std::vector<Range> ranges;        // swept, address by address (--scan ranges)
    bool adapters = false;            // each adapter's subnet swept too (--scan alone)
    double perSecond = 400;           // addresses a second of a sweep
};

// The answer's check: HMAC-SHA-256 under the rhizome key of the probe's nonce, the node's hash and its port.
inline Hash answerCheck(const std::string& rhizomeKey, const Hash& nonce, const Hash& node, uint16_t port) {
    unsigned char m[66];
    std::memcpy(m, nonce.data(), 32);
    std::memcpy(m + 32, node.data(), 32);
    std::memcpy(m + 64, &port, 2);
    return sha256(m, sizeof m, rhizomeKey.data(), rhizomeKey.size());
}

inline SOCKET udpSocket(uint16_t port, bool reuse) {
    SOCKET s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) throw std::runtime_error("no UDP socket");
    BOOL on = 1;
    setsockopt(s, SOL_SOCKET, SO_BROADCAST, (const char*)&on, sizeof on);
    // Several nodes on one machine (tests) share the port: each gets the broadcast probes.
    if (reuse) setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*)&on, sizeof on);
    DWORD wait = 250;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char*)&wait, sizeof wait);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(s, (sockaddr*)&a, sizeof a) != 0) {
        closesocket(s);
        throw std::runtime_error("cannot listen for discovery on UDP port " + std::to_string(port));
    }
    return s;
}

// A node's side: answers each probe from a master of its rhizome.
class Responder {
public:
    Responder(std::string rhizomeKey, Hash self, uint16_t tcpPort, uint16_t port = DiscoveryPort)
        : key_(std::move(rhizomeKey)), self_(self), tcp_(tcpPort), s_(responderSocket(port)) {
        thread_ = std::thread([this] { run(); });
    }
    ~Responder() {
        running_ = false;
        closesocket(s_);
        if (thread_.joinable()) thread_.join();
    }
    uint64_t answered() const { return answered_; }

private:
    // The default port: the first of the discovery ports free here (shared with the others as the last resort); a
    // port given (tests): shared.
    static SOCKET responderSocket(uint16_t port) {
        if (port == DiscoveryPort)
            for (uint16_t k = 0; k < DiscoveryPorts; ++k) {
                try {
                    return udpSocket(uint16_t(port + k), false);
                } catch (...) {
                }
            }
        return udpSocket(port, true);
    }
    void run() {
        while (running_) {
            unsigned char b[4 + 32];
            sockaddr_in from{};
            int len = sizeof from;
            const int n = recvfrom(s_, (char*)b, sizeof b, 0, (sockaddr*)&from, &len);
            if (n != int(sizeof b)) continue;
            uint32_t magic;
            std::memcpy(&magic, b, 4);
            if (magic != ProbeMagic) continue;
            Hash nonce;
            std::memcpy(nonce.data(), b + 4, 32);
            unsigned char a[4 + 32 + 2 + 32];
            std::memcpy(a, &AnswerMagic, 4);
            std::memcpy(a + 4, self_.data(), 32);
            std::memcpy(a + 36, &tcp_, 2);
            const Hash check = answerCheck(key_, nonce, self_, tcp_);
            std::memcpy(a + 38, check.data(), 32);
            sendto(s_, (const char*)a, sizeof a, 0, (sockaddr*)&from, len);
            ++answered_;
        }
    }
    std::string key_;
    Hash self_;
    uint16_t tcp_;
    SOCKET s_;
    std::atomic<bool> running_{true};
    std::atomic<uint64_t> answered_{0};
    std::thread thread_;
};

// The master's side: probes every adapter's broadcast about every second; each node of the rhizome that answers is found
// once (found: its hash and host:port, the address its answer came from).
class Prober {
public:
    Prober(std::string rhizomeKey, std::function<void(const Hash&, const std::string&)> found, uint16_t port = DiscoveryPort,
           double interval = 1.0, ProbeOptions options = {})
        : key_(std::move(rhizomeKey)), found_(std::move(found)), port_(port), interval_(interval), o_(std::move(options)),
          s_(udpSocket(0, false)) {
        thread_ = std::thread([this] { run(); });
    }
    // A machine probed each round from now on (host order).
    void addHost(uint32_t ip) {
        std::lock_guard<std::mutex> g(m_);
        if (std::find(o_.hosts.begin(), o_.hosts.end(), ip) == o_.hosts.end()) o_.hosts.push_back(ip);
    }
    // Addresses probed one by one so far (remembered machines and sweeps).
    uint64_t unicastProbes() const { return unicast_; }
    ~Prober() {
        running_ = false;
        closesocket(s_);
        if (thread_.joinable()) thread_.join();
    }
    // Probe again for a node already found (it came back after it was lost).
    void forget(const Hash& node) {
        std::lock_guard<std::mutex> g(m_);
        known_.erase(node);
    }

private:
    // A probe to an address at each discovery port (one port for a port given: tests).
    void toEachPort(sockaddr_in a, const unsigned char* b, size_t n) {
        const uint16_t ports = port_ == DiscoveryPort ? DiscoveryPorts : 1;
        for (uint16_t k = 0; k < ports; ++k) {
            a.sin_port = htons(uint16_t(port_ + k));
            sendto(s_, (const char*)b, int(n), 0, (sockaddr*)&a, sizeof a);
        }
    }
    void probe(uint32_t ip, const unsigned char* b, size_t n) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(ip);
        toEachPort(a, b, n);
        ++unicast_;
    }
    // A round's share of the sweep: the next addresses of the ranges (and the adapters' subnets, read again each pass).
    void sweep(const unsigned char* b, size_t n) {
        if (o_.ranges.empty() && !o_.adapters) return;
        size_t budget = size_t(std::max(1.0, o_.perSecond * interval_));
        for (size_t guard = 0; budget > 0 && guard < 64; ++guard) {
            if (rangeAt_ >= pass_.size()) {
                pass_ = o_.ranges;
                if (o_.adapters)
                    for (const Range& r : adapterSubnets()) pass_.push_back(r);
                rangeAt_ = 0, hostAt_ = 0;
                if (pass_.empty()) return;
            }
            const Range& r = pass_[rangeAt_];
            while (budget > 0 && hostAt_ < r.count) probe(r.first + hostAt_++, b, n), --budget;
            if (hostAt_ >= r.count) ++rangeAt_, hostAt_ = 0;
        }
    }
    void run() {
        auto last = std::chrono::steady_clock::now() - std::chrono::hours(1);
        Hash nonce{};
        while (running_) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last).count() >= interval_) {
                last = now;
                randomBytes(nonce.data(), nonce.size());
                unsigned char b[4 + 32];
                std::memcpy(b, &ProbeMagic, 4);
                std::memcpy(b + 4, nonce.data(), 32);
                // The adapters are read again each time: a cable plugged in since, its address given, is probed too.
                if (o_.broadcast)
                    for (const in_addr& to : broadcastAddresses()) {
                        sockaddr_in a{};
                        a.sin_family = AF_INET;
                        a.sin_addr = to;
                        toEachPort(a, b, sizeof b);
                    }
                // The machines reached before, one by one, then the sweep's next addresses (plan 18).
                std::vector<uint32_t> hosts;
                {
                    std::lock_guard<std::mutex> g(m_);
                    hosts = o_.hosts;
                }
                for (uint32_t ip : hosts) probe(ip, b, sizeof b);
                sweep(b, sizeof b);
                // This machine's own nodes (tests, or a master's machine that also runs a node), with the broadcasts.
                if (o_.broadcast) {
                    sockaddr_in self{};
                    self.sin_family = AF_INET;
                    inet_pton(AF_INET, "127.0.0.1", &self.sin_addr);
                    toEachPort(self, b, sizeof b);
                }
                // A sweep's answers come back over the next rounds: the last few probes' answers are taken.
                recent_.push_front(nonce);
                if (recent_.size() > 8) recent_.pop_back();
            }
            unsigned char a[4 + 32 + 2 + 32];
            sockaddr_in from{};
            int len = sizeof from;
            const int n = recvfrom(s_, (char*)a, sizeof a, 0, (sockaddr*)&from, &len);
            if (n != int(sizeof a)) continue;
            uint32_t magic;
            std::memcpy(&magic, a, 4);
            if (magic != AnswerMagic) continue;
            Hash node, check;
            uint16_t tcp;
            std::memcpy(node.data(), a + 4, 32);
            std::memcpy(&tcp, a + 36, 2);
            std::memcpy(check.data(), a + 38, 32);
            bool ours = false;  // of this rhizome, and of a recent probe
            for (const Hash& n : recent_) ours = ours || check == answerCheck(key_, n, node, tcp);
            if (!ours) continue;
            char host[INET_ADDRSTRLEN] = {};
            inet_ntop(AF_INET, &from.sin_addr, host, sizeof host);
            {
                std::lock_guard<std::mutex> g(m_);
                if (!known_.insert(node).second) continue;
            }
            found_(node, std::string(host) + ":" + std::to_string(tcp));
        }
    }
    std::string key_;
    std::function<void(const Hash&, const std::string&)> found_;
    uint16_t port_;
    double interval_;
    ProbeOptions o_;
    SOCKET s_;
    std::deque<Hash> recent_;
    std::vector<Range> pass_;
    size_t rangeAt_ = 0;
    uint32_t hostAt_ = 0;
    std::atomic<uint64_t> unicast_{0};
    std::mutex m_;
    std::set<Hash> known_;
    std::atomic<bool> running_{true};
    std::thread thread_;
};

}  // namespace xpu2::interactive::rhizome
