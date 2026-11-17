// Rhizome's members (RHIZOME.md, 2): every node a node knows, by its hash, with its public key, what it has, when it was
// last heard from, its distance (the round trip of its heartbeats, a running mean) and the last sequence number it
// sent. A member silent for SuspectAfter is suspected, for GoneAfter dropped; a quarantined one is refused for the
// rest of the session whatever it sends. Not thread-safe: the node's network thread owns it.
#pragma once
#include "rhizome_wire.h"
#include <algorithm>
#include <map>
#include <optional>
#include <vector>

namespace xpu2::interactive::rhizome {

enum class State : uint8_t { Alive, Suspect, Gone };

// What a machine renders its jobs on, as it says (RhizomeNode::setDevices): "CPU and GPU", "the GPU alone", or "the CPU
// alone" and why (its cards cannot be used, or its devices are set so). Its rate is measured on what it renders on;
// when that changes between the processor alone and its cards, the rate it had on the other is taken up again.
struct DeviceRates {
    bool known = false;     // said once at least
    bool cpuAlone = false;  // on the processor alone now
    double other = 0;       // its rate on the devices it is not on now (0: never measured)
};
inline bool cpuAloneText(const std::string& devices) { return devices.rfind("the CPU alone", 0) == 0; }
// The first guess of a machine's rate on its processor alone, from its rate with its cards: low on purpose, as a node
// given too little steals more, and the chunks it returns correct it (Members::rate).
constexpr double CpuAloneGuess = 1.0 / 16.0;
// A machine's rate as it changes devices: the rate it had on the new ones before; else, gone to the processor alone,
// CpuAloneGuess of its rate on its cards, or, gone to its cards, unknown (0: the plan takes the others' median). The
// first time it says, its rate is left as measured.
inline double switchDevices(DeviceRates& d, double rate, bool cpuAlone) {
    if (!d.known) {
        d.known = true;
        d.cpuAlone = cpuAlone;
        return rate;
    }
    if (d.cpuAlone == cpuAlone) return rate;
    const double next = d.other > 0 ? d.other : cpuAlone ? rate * CpuAloneGuess : 0.0;
    d.other = rate;
    d.cpuAlone = cpuAlone;
    return next;
}

struct Member {
    Hash hash{};
    Bytes publicKey;
    Capabilities caps;
    double lastHeard = 0;   // seconds, the node's clock
    double rtt = -1;        // seconds; below zero: not measured yet
    uint64_t lastSequence = 0;
    State state = State::Alive;
    bool quarantined = false;
    bool master = false;
    std::string shaders;  // its GPU programs, as its heartbeats say (RhizomeNode::setShaders)
    std::string devices;  // what it renders on, as its heartbeats and Ready say (RhizomeNode::setDevices)
    DeviceRates deviceRates;
};

class Members {
public:
    double suspectAfter = 3.0, goneAfter = 10.0;

    // A member introduced (its Hello or Welcome, already authentic). False: refused (quarantined, or a key that does
    // not hash to its hash).
    bool introduce(const Hash& hash, const Bytes& key, const Capabilities& caps, double now, bool master = false) {
        if (!keyMatches(key, hash)) return false;
        auto& m = members_[hash];
        if (m.quarantined) return false;
        m.hash = hash, m.publicKey = key, m.caps = caps, m.lastHeard = now, m.state = State::Alive;
        m.master = m.master || master;
        return true;
    }
    const Member* find(const Hash& h) const {
        auto it = members_.find(h);
        return it == members_.end() ? nullptr : &it->second;
    }
    Member* find(const Hash& h) {
        auto it = members_.find(h);
        return it == members_.end() ? nullptr : &it->second;
    }
    // Whether a frame from a known member is to be taken: authentic, its sequence past the last, not quarantined.
    // Taking it marks the member heard.
    // lastOnConnection: the last sequence taken on the connection it came by (messages on one connection keep their
    // order; on two connections between the same pair they need not, so the number is checked by connection), or, as
    // here without one, the member's own.
    bool accept(const Bytes& frame, const Message& m, double now, uint64_t* lastOnConnection = nullptr) {
        Member* who = find(m.sender);
        if (!who || who->quarantined) return false;
        uint64_t& last = lastOnConnection ? *lastOnConnection : who->lastSequence;
        if (m.sequence <= last) return false;
        if (!authentic(frame, m, who->publicKey)) return false;
        last = m.sequence;
        who->lastSequence = std::max(who->lastSequence, m.sequence);
        who->lastHeard = now;
        who->state = State::Alive;  // a member taken for gone that speaks again is back (its work was given out)
        return true;
    }
    // A round trip measured (a heartbeat's echo): the distance's running mean.
    void roundTrip(const Hash& h, double seconds) {
        if (Member* m = find(h)) m->rtt = m->rtt < 0 ? seconds : 0.8 * m->rtt + 0.2 * seconds;
    }
    void rate(const Hash& h, double samplesPerSecond) {
        if (Member* m = find(h)) m->caps.rate = m->caps.rate > 0 ? 0.7 * m->caps.rate + 0.3 * samplesPerSecond : samplesPerSecond;
    }
    // What a member renders on, as it says: its rate taken over for the new devices (switchDevices), at once, so the
    // next frame's plan gives a node gone to its processor alone what it can do. True where it changed.
    bool devices(const Hash& h, const std::string& devices) {
        Member* m = find(h);
        if (!m || devices.empty() || devices == m->devices) return false;
        m->caps.rate = switchDevices(m->deviceRates, m->caps.rate, cpuAloneText(devices));
        m->devices = devices;
        return true;
    }
    // Marks the silent: suspected, then gone. Returns the members that went this sweep.
    // Bytes of a member's arrived (a message on its way, however long): it is not silent. Never brings back the gone.
    void heardBytes(const Hash& h, double at) {
        auto it = members_.find(h);
        if (it == members_.end() || it->second.state == State::Gone || it->second.quarantined) return;
        if (at > it->second.lastHeard) it->second.lastHeard = at;
        if (it->second.state == State::Suspect) it->second.state = State::Alive;
    }
    std::vector<Hash> sweep(double now) {
        std::vector<Hash> gone;
        for (auto& [h, m] : members_) {
            if (m.state == State::Gone) continue;
            const double silent = now - m.lastHeard;
            if (silent > goneAfter) {
                m.state = State::Gone;
                gone.push_back(h);
            } else if (silent > suspectAfter) {
                m.state = State::Suspect;
            }
        }
        return gone;
    }
    // Refused for the rest of the session (a failed audit); it counts as gone.
    void quarantine(const Hash& h) {
        auto& m = members_[h];
        m.hash = h;
        m.quarantined = true;
        m.state = State::Gone;
    }
    bool quarantined(const Hash& h) const {
        const Member* m = find(h);
        return m && m->quarantined;
    }
    // The members that can take work: alive or suspected (a suspect may only be slow), not quarantined.
    std::vector<const Member*> working() const {
        std::vector<const Member*> out;
        for (const auto& [h, m] : members_)
            if (m.state != State::Gone && !m.quarantined) out.push_back(&m);
        return out;
    }
    size_t size() const { return members_.size(); }
    // Every member known, gone and refused ones too (status).
    std::vector<const Member*> all() const {
        std::vector<const Member*> out;
        for (const auto& [h, m] : members_) out.push_back(&m);
        return out;
    }

private:
    std::map<Hash, Member> members_;
};

}  // namespace xpu2::interactive::rhizome
