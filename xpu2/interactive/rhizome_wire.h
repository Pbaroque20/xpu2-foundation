// Rhizome's messages (RHIZOME.md, 1-2): a header (its magic, the type, a sequence number, the sender's hash, the payload's
// length), the payload, and the sender's signature over both. A receiver takes a message only once it knows the
// sender's public key (from its Hello, whose key must hash to the sender's hash) and the signature verifies; the
// sequence must rise, so a message cannot be played again. Little-endian throughout.
#pragma once
#include "rhizome_identity.h"
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace xpu2::interactive::rhizome {

constexpr uint32_t Magic = 0x315A4852u;  // "RHZ1"
constexpr size_t HeaderBytes = 4 + 2 + 2 + 8 + 32 + 4;
constexpr size_t SignatureBytes = 64;
constexpr uint32_t MaxPayload = 256u << 20;  // a contribution of a large cell with its AOVs stays well under

enum class Type : uint16_t {
    Hello = 1,         // a node introduces itself: its public key, what it has, a challenge, its proof of the rhizome key
    Welcome = 2,       // the answer: the receiver's key, its proof for the challenge, the members it knows
    Heartbeat = 3,     // alive: the sender's clock (for the round trip), its members' digest
    Echo = 4,          // a heartbeat's answer: its clock back
    Members = 5,       // the members the sender knows (hash, address, rate)
    Assign = 6,        // master: a frame's cells and their owners
    StealAsk = 7,      // a node out of work asks an owner for some of a cell's ranges
    StealGrant = 8,    // the owner gives them (or none)
    Contribution = 9,  // a node's film for a cell's sample range
    AuditAsk = 10,     // master: render these samples again (a node the master trusts, or the master itself)
    AuditAnswer = 11,
    Quarantine = 12,   // master: a node's hash is refused
    FrameDone = 13,    // master: the frame's picture is complete
    Leave = 14,
};

// A byte writer and reader for payloads.
struct Writer {
    Bytes b;
    void u8(uint8_t v) { b.push_back(v); }
    void u16(uint16_t v) { raw(&v, 2); }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void f64(double v) { raw(&v, 8); }
    void hash(const Hash& h) { raw(h.data(), h.size()); }
    void bytes(const Bytes& v) {
        u32(uint32_t(v.size()));
        raw(v.data(), v.size());
    }
    void str(const std::string& s) {
        u32(uint32_t(s.size()));
        raw(s.data(), s.size());
    }
    template <class T>
    void array(const std::vector<T>& v) {
        u32(uint32_t(v.size()));
        raw(v.data(), v.size() * sizeof(T));
    }
    void raw(const void* p, size_t n) {
        const auto* c = static_cast<const unsigned char*>(p);
        b.insert(b.end(), c, c + n);
    }
};
struct Reader {
    const unsigned char* p;
    size_t n, at = 0;
    bool bad = false;
    Reader(const Bytes& b) : p(b.data()), n(b.size()) {}
    Reader(const unsigned char* data, size_t size) : p(data), n(size) {}
    bool take(void* out, size_t k) {
        if (bad || n - at < k) {
            bad = true;
            std::memset(out, 0, k);
            return false;
        }
        std::memcpy(out, p + at, k);
        at += k;
        return true;
    }
    uint8_t u8() { uint8_t v; take(&v, 1); return v; }
    uint16_t u16() { uint16_t v; take(&v, 2); return v; }
    uint32_t u32() { uint32_t v; take(&v, 4); return v; }
    uint64_t u64() { uint64_t v; take(&v, 8); return v; }
    float f32() { float v; take(&v, 4); return v; }
    double f64() { double v; take(&v, 8); return v; }
    Hash hash() { Hash h; take(h.data(), h.size()); return h; }
    Bytes bytes() {
        const uint32_t k = u32();
        if (bad || k > n - at) { bad = true; return {}; }
        Bytes v(p + at, p + at + k);
        at += k;
        return v;
    }
    std::string str() {
        const Bytes v = bytes();
        return std::string(v.begin(), v.end());
    }
    template <class T>
    std::vector<T> array() {
        const uint32_t k = u32();
        if (bad || size_t(k) * sizeof(T) > n - at) { bad = true; return {}; }
        std::vector<T> v(k);
        take(v.data(), size_t(k) * sizeof(T));
        return v;
    }
    bool done() const { return !bad && at == n; }
};

struct Message {
    Type type{};
    uint64_t sequence = 0;
    Hash sender{};
    Bytes payload;
};

// A message as bytes: header, payload, signature.
inline Bytes encode(const Identity& id, Type type, uint64_t sequence, const Bytes& payload) {
    Writer w;
    w.u32(Magic);
    w.u16(uint16_t(type));
    w.u16(0);
    w.u64(sequence);
    w.hash(id.hash());
    w.u32(uint32_t(payload.size()));
    w.raw(payload.data(), payload.size());
    const Bytes sig = id.sign(w.b.data(), w.b.size());
    w.raw(sig.data(), sig.size());
    return w.b;
}

// A frame's header read: the payload's length, or nothing (not a Rhizome frame).
inline std::optional<uint32_t> payloadLength(const unsigned char* header) {
    Reader r(header, HeaderBytes);
    if (r.u32() != Magic) return std::nullopt;
    r.u16(), r.u16(), r.u64(), r.hash();
    const uint32_t n = r.u32();
    if (n > MaxPayload) return std::nullopt;
    return n;
}
// Reads a whole frame without checking the signature (the sender's key may come in this very message: a Hello).
inline std::optional<Message> peek(const Bytes& frame) {
    if (frame.size() < HeaderBytes + SignatureBytes) return std::nullopt;
    Reader r(frame.data(), HeaderBytes);
    Message m;
    if (r.u32() != Magic) return std::nullopt;
    m.type = Type(r.u16());
    r.u16();
    m.sequence = r.u64();
    m.sender = r.hash();
    const uint32_t n = r.u32();
    if (frame.size() != HeaderBytes + n + SignatureBytes) return std::nullopt;
    m.payload.assign(frame.begin() + HeaderBytes, frame.begin() + HeaderBytes + n);
    return m;
}
// Whether a frame is signed by the key given, and that key is the sender's.
inline bool authentic(const Bytes& frame, const Message& m, const Bytes& senderKey) {
    if (!keyMatches(senderKey, m.sender)) return false;
    const size_t signedBytes = frame.size() - SignatureBytes;
    const Bytes sig(frame.begin() + signedBytes, frame.end());
    return verify(senderKey, frame.data(), signedBytes, sig);
}

// What a node has, in its Hello (and the master's view of it).
struct Capabilities {
    uint32_t cpuThreads = 0, numaNodes = 1, cards = 0;
    double rate = 0;  // samples a second of the reference work, measured (0: not yet)
    std::string address;  // where it listens, host:port
    void write(Writer& w) const {
        w.u32(cpuThreads), w.u32(numaNodes), w.u32(cards), w.f64(rate), w.str(address);
    }
    static Capabilities read(Reader& r) {
        Capabilities c;
        c.cpuThreads = r.u32(), c.numaNodes = r.u32(), c.cards = r.u32(), c.rate = r.f64(), c.address = r.str();
        return c;
    }
};

}  // namespace xpu2::interactive::rhizome
