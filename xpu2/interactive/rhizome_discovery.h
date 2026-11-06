// Rhizome's discovery (RHIZOME.md, 2): a master finds the headless nodes on its networks, a direct cable between two
// machines among them. A node listens for probes on a UDP port (DiscoveryPort); the master sends one, about every
// second, to the broadcast address of each of its network adapters (each IPv4 address's subnet: a cable with no router
// on it has the address Windows gives itself, 169.254.x.y, and its subnet's broadcast reaches the other end) and to
// 255.255.255.255. A node answers the master straight back with its hash, the TCP port it listens on, and a keyed hash
// under the rhizome key: an answer without it is not taken, so only a member of the rhizome is found. The master then reaches
// the node at the address the answer came from and the usual handshake (rhizome_node.h) follows.
#pragma once
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include "rhizome_identity.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace xpu2::interactive::rhizome {

constexpr uint16_t DiscoveryPort = 47101;
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
        : key_(std::move(rhizomeKey)), self_(self), tcp_(tcpPort), s_(udpSocket(port, true)) {
        thread_ = std::thread([this] { run(); });
    }
    ~Responder() {
        running_ = false;
        closesocket(s_);
        if (thread_.joinable()) thread_.join();
    }
    uint64_t answered() const { return answered_; }

private:
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
           double interval = 1.0)
        : key_(std::move(rhizomeKey)), found_(std::move(found)), port_(port), interval_(interval), s_(udpSocket(0, false)) {
        thread_ = std::thread([this] { run(); });
    }
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
                for (const in_addr& to : broadcastAddresses()) {
                    sockaddr_in a{};
                    a.sin_family = AF_INET;
                    a.sin_port = htons(port_);
                    a.sin_addr = to;
                    sendto(s_, (const char*)b, sizeof b, 0, (sockaddr*)&a, sizeof a);
                }
                // This machine's own nodes (tests, or a master's machine that also runs a node).
                sockaddr_in self{};
                self.sin_family = AF_INET;
                self.sin_port = htons(port_);
                inet_pton(AF_INET, "127.0.0.1", &self.sin_addr);
                sendto(s_, (const char*)b, sizeof b, 0, (sockaddr*)&self, sizeof self);
                current_ = nonce;
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
            if (check != answerCheck(key_, current_, node, tcp)) continue;  // not of this rhizome, or an old probe's
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
    SOCKET s_;
    Hash current_{};
    std::mutex m_;
    std::set<Hash> known_;
    std::atomic<bool> running_{true};
    std::thread thread_;
};

}  // namespace xpu2::interactive::rhizome
