// Rhizome's signed approvals (RHIZOME.md, 1b): who may join beyond the rhizome key. An approver (a master's machine, or a
// successor it names) signs an endorsement of a node's hash; the node keeps it and shows it in each handshake, so any
// member checks it without asking anyone: signed by an approver it trusts, of the hash shaking hands, not revoked. A
// revocation is signed the same way and passed through the swarm; one issued at or after an endorsement beats it.
//
// An endorsement: "RZE1", the subject's hash, its role (a node, or an approver: one that may endorse and revoke in its
// turn, as a master's successors do), when it was issued (seconds since 1970), the approver's public key, and the
// approver's signature over all of it. A revocation: "RZR1", the subject, when, the approver's key, its signature.
//
// A member's folder (beside node.key): approvers.txt (the approvers' hashes it trusts, a line each), endorsed/ (its own
// endorsements; an approver's issued ones as <hash>.hve), revoked/ (revocations, <hash>.hvr). A member with no approver
// trusts the one who first endorses it, if that is the master it joined (pin).
#pragma once
#include "rhizome_identity.h"
#include "rhizome_wire.h"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace xpu2::interactive::rhizome {

enum class Role : uint8_t { Node = 0, Approver = 1 };

struct Endorsement {
    Hash subject{};
    Role role = Role::Node;
    uint64_t issued = 0;
    Bytes approverKey;
    Bytes signature;
    Hash approver() const { return sha256(approverKey); }
    Bytes body() const {
        Writer w;
        w.u32(0x31455A52u);  // "RZE1"
        w.hash(subject);
        w.u8(uint8_t(role));
        w.u64(issued);
        w.bytes(approverKey);
        return w.b;
    }
    Bytes bytes() const {
        Writer w;
        w.raw(body().data(), body().size());
        w.bytes(signature);
        return w.b;
    }
    static std::optional<Endorsement> read(const Bytes& b) {
        Reader r(b);
        if (r.u32() != 0x31455A52u) return std::nullopt;
        Endorsement e;
        e.subject = r.hash();
        e.role = Role(r.u8());
        e.issued = r.u64();
        e.approverKey = r.bytes();
        e.signature = r.bytes();
        if (!r.done() || (e.role != Role::Node && e.role != Role::Approver)) return std::nullopt;
        return e;
    }
    bool signedWell() const {
        const Bytes m = body();
        return verify(approverKey, m.data(), m.size(), signature);
    }
};

struct Revocation {
    Hash subject{};
    uint64_t issued = 0;
    Bytes approverKey;
    Bytes signature;
    Hash approver() const { return sha256(approverKey); }
    Bytes body() const {
        Writer w;
        w.u32(0x31525A52u);  // "RZR1"
        w.hash(subject);
        w.u64(issued);
        w.bytes(approverKey);
        return w.b;
    }
    Bytes bytes() const {
        Writer w;
        const Bytes b = body();
        w.raw(b.data(), b.size());
        w.bytes(signature);
        return w.b;
    }
    static std::optional<Revocation> read(const Bytes& b) {
        Reader r(b);
        if (r.u32() != 0x31525A52u) return std::nullopt;
        Revocation v;
        v.subject = r.hash();
        v.issued = r.u64();
        v.approverKey = r.bytes();
        v.signature = r.bytes();
        if (!r.done()) return std::nullopt;
        return v;
    }
    bool signedWell() const {
        const Bytes m = body();
        return verify(approverKey, m.data(), m.size(), signature);
    }
};

inline std::optional<Hash> parseHash(const std::string& s) {
    if (s.size() != 64) return std::nullopt;
    Hash h{};
    for (size_t i = 0; i < 32; ++i) {
        auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
        const int a = v(s[2 * i]), b = v(s[2 * i + 1]);
        if (a < 0 || b < 0) return std::nullopt;
        h[i] = uint8_t(a * 16 + b);
    }
    return h;
}

// A member's trust: the approvers it trusts, its own endorsements, the revocations it knows; an approver's way to
// endorse and revoke. Its files are its folder's. Not thread-safe: its node's lock holds it.
class Trust {
public:
    Trust(const std::filesystem::path& folder, const Identity& self) : folder_(folder), self_(self) {
        std::filesystem::create_directories(folder_ / "endorsed");
        std::filesystem::create_directories(folder_ / "revoked");
        std::filesystem::create_directories(folder_ / "blocked");
        std::ifstream in(folder_ / "approvers.txt");
        for (std::string line; std::getline(in, line);) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (auto h = parseHash(line)) approvers_.insert(*h);
        }
        for (const auto& f : std::filesystem::directory_iterator(folder_ / "endorsed"))
            if (auto e = Endorsement::read(fileBytes(f.path()))) {
                if (e->subject == self_.hash() && e->signedWell()) own_.push_back(*e);
            }
        refreshRevocations();
    }

    // Whether this member asks endorsements of those it meets: it was told to (a master's --approve), or it trusts an
    // approver already (it was endorsed).
    bool required() const { return require_ || !approvers_.empty(); }
    void require(bool on) { require_ = on; }
    // Whether this member may endorse: it is an approver itself (a master with --approve trusts itself, a successor by
    // its approver endorsement).
    bool approver() const {
        if (approvers_.count(self_.hash())) return true;
        for (const auto& e : own_)
            if (e.role == Role::Approver && trusted(e.approver())) return true;
        return false;
    }
    // An approver trusted from now on (the trust command, --trust).
    void addApprover(const Hash& h) {
        if (approvers_.insert(h).second) saveApprovers();
    }
    // A machine this member alone refuses (the revoke command of one that is not an approver): blocked/<hash>, kept
    // across restarts, never passed on. It beats any endorsement here.
    void block(const Hash& h) {
        std::ofstream(folder_ / "blocked" / (hex(h) + ".txt")) << hex(h) << "\n";
        blocked_.insert(h);
    }
    bool blocked(const Hash& h) const { return blocked_.count(h) != 0; }
    void trustSelf() {
        if (approvers_.insert(self_.hash()).second) saveApprovers();
    }
    bool trusted(const Hash& approver) const { return approvers_.count(approver) != 0; }
    const std::set<Hash>& approvers() const { return approvers_; }

    // Whether a member shaking hands may join: its endorsements (as it showed them) hold one signed by an approver this
    // member trusts (directly, or an approver endorsed by one), of its hash, issued after any revocation of it.
    bool admits(const Hash& who, const std::vector<Bytes>& shown) const {
        if (!required()) return true;
        if (revokedAfter(who, 0)) return false;
        if (approvers_.count(who)) return true;  // an approver itself
        std::vector<Endorsement> es;
        for (const auto& b : shown)
            if (auto e = Endorsement::read(b))
                if (e->signedWell()) es.push_back(*e);
        // Approvers by delegation: an approver endorsement from a trusted approver, among those shown.
        std::set<Hash> ok = approvers_;
        for (int round = 0; round < 3; ++round)
            for (const auto& e : es)
                if (e.role == Role::Approver && ok.count(e.approver()) && !revokedAfter(e.subject, e.issued)) ok.insert(e.subject);
        for (const auto& e : es)
            if (e.subject == who && ok.count(e.approver()) && !revokedAfter(who, e.issued)) return true;
        return false;
    }
    // This member's endorsements to show: its own, and those of the approvers that endorsed it (so a peer can follow
    // a delegation it does not hold).
    std::vector<Bytes> shown() const {
        std::vector<Bytes> out;
        for (const auto& e : own_) out.push_back(e.bytes());
        for (const auto& e : chain_) out.push_back(e.bytes());
        return out;
    }
    // An endorsement given to this member by the master it joined: kept (and its approver pinned when it trusts none).
    // False: not for it, not well signed, or from an approver it does not trust (and may not pin).
    bool take(const Bytes& b, const Hash& from, const std::vector<Bytes>& chain = {}) {
        auto e = Endorsement::read(b);
        if (!e || e->subject != self_.hash() || !e->signedWell()) return false;
        if (!trusted(e->approver())) {
            if (!approvers_.empty() || e->approver() != from) {
                // Through a delegation it can check: an approver endorsed by one it trusts.
                bool delegated = false;
                for (const auto& cb : chain)
                    if (auto c = Endorsement::read(cb))
                        delegated = delegated || (c->signedWell() && c->role == Role::Approver && c->subject == e->approver() && trusted(c->approver()));
                if (!delegated) return false;
                for (const auto& cb : chain)
                    if (auto c = Endorsement::read(cb)) chain_.push_back(*c);
            } else {
                approvers_.insert(e->approver());  // pinned: the master that endorsed it, the first approver it meets
                saveApprovers();
            }
        }
        for (const auto& o : own_)
            if (o.bytes() == e->bytes()) return true;
        own_.push_back(*e);
        writeFile(folder_ / "endorsed" / ("self-" + std::to_string(own_.size()) + ".hve"), e->bytes());
        return true;
    }
    // An approver's endorsement of a hash (Role::Approver: it may approve in its turn), kept in endorsed/<hash>.hve.
    Bytes endorse(const Hash& subject, Role role = Role::Node) {
        Endorsement e;
        e.subject = subject, e.role = role, e.issued = uint64_t(std::time(nullptr));
        e.approverKey = self_.publicKey();
        const Bytes m = e.body();
        e.signature = self_.sign(m.data(), m.size());
        writeFile(folder_ / "endorsed" / (hex(subject) + ".hve"), e.bytes());
        return e.bytes();
    }
    // An endorsement this approver issued for a hash (by the --rhizome-approve command, or by this member), if any.
    std::optional<Bytes> issued(const Hash& subject) const {
        const auto p = folder_ / "endorsed" / (hex(subject) + ".hve");
        if (!std::filesystem::exists(p)) return std::nullopt;
        const Bytes b = fileBytes(p);
        auto e = Endorsement::read(b);
        if (!e || e->subject != subject || !e->signedWell()) return std::nullopt;
        if (revokedAfter(subject, e->issued)) return std::nullopt;  // revoked since: it no longer counts
        return b;
    }
    Bytes revoke(const Hash& subject) {
        Revocation v;
        v.subject = subject, v.issued = uint64_t(std::time(nullptr));
        v.approverKey = self_.publicKey();
        const Bytes m = v.body();
        v.signature = self_.sign(m.data(), m.size());
        writeFile(folder_ / "revoked" / (hex(subject) + ".hvr"), v.bytes());
        revoked_[subject] = std::max(revoked_[subject], v.issued);
        return v.bytes();
    }
    // A revocation passed on by a member: kept when an approver this member trusts signed it. Returns its subject.
    std::optional<Hash> takeRevocation(const Bytes& b) {
        auto v = Revocation::read(b);
        if (!v || !v->signedWell() || !trusted(v->approver())) return std::nullopt;
        if (revoked_.count(v->subject) && revoked_[v->subject] >= v->issued) return v->subject;
        revoked_[v->subject] = v->issued;
        writeFile(folder_ / "revoked" / (hex(v->subject) + ".hvr"), b);
        return v->subject;
    }
    // The revocations kept (to pass on to a member as it joins), and those the folder has gained since (the
    // --rhizome-revoke command, while a master runs).
    std::vector<Bytes> revocations() const {
        std::vector<Bytes> out;
        for (const auto& f : std::filesystem::directory_iterator(folder_ / "revoked")) out.push_back(fileBytes(f.path()));
        return out;
    }
    std::vector<Hash> refreshRevocations() {
        std::vector<Hash> fresh;
        for (const auto& f : std::filesystem::directory_iterator(folder_ / "blocked"))
            if (auto h = parseHash(f.path().stem().string()))
                if (blocked_.insert(*h).second) fresh.push_back(*h);
        for (const auto& f : std::filesystem::directory_iterator(folder_ / "revoked"))
            if (auto v = Revocation::read(fileBytes(f.path())))
                if (v->signedWell() && (trusted(v->approver()) || v->approver() == self_.hash())) {
                    auto it = revoked_.find(v->subject);
                    if (it == revoked_.end() || it->second < v->issued) {
                        revoked_[v->subject] = v->issued;
                        fresh.push_back(v->subject);
                    }
                }
        return fresh;
    }
    bool revokedAfter(const Hash& who, uint64_t issued) const {
        if (blocked_.count(who)) return true;
        auto it = revoked_.find(who);
        return it != revoked_.end() && it->second >= issued;
    }

    static Bytes fileBytes(const std::filesystem::path& p) {
        std::ifstream in(p, std::ios::binary);
        return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    }
    static void writeFile(const std::filesystem::path& p, const Bytes& b) {
        const auto temp = p.string() + ".part";
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
        }
        std::error_code e;
        std::filesystem::rename(temp, p, e);
    }

private:
    void saveApprovers() const {
        std::ofstream out(folder_ / "approvers.txt", std::ios::trunc);
        for (const auto& h : approvers_) out << hex(h) << "\n";
    }
    std::filesystem::path folder_;
    const Identity& self_;
    std::set<Hash> approvers_;
    std::vector<Endorsement> own_, chain_;
    std::map<Hash, uint64_t> revoked_;
    std::set<Hash> blocked_;
    bool require_ = false;
};

}  // namespace xpu2::interactive::rhizome
