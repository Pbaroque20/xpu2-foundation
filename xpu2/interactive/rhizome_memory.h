// Rhizome's memory (RHIZOME.md, 4b): the swarm's RAM and graphics card memory as one pool. The scene's pieces (the scene
// file's blocks, meshes, texture tiles), each named by its SHA-256, are placed on the nodes by their free memory,
// each kept on Replicas nodes so a machine that fails loses no piece; its pieces are then copied again, from a node
// that still holds them, to nodes with room (allocation as nodes come and go). A piece a graphics card reads (a mesh,
// a texture tile for the GPU) prefers card memory; the rest, and a card piece where no card has room, go to RAM.
//
// Placement is rendezvous hashing weighted by free memory: each node scores a piece by a hash of the two and its room,
// and the piece goes to the best scores that can hold it. Every node works out the same placement from the same
// members, so a node finds a piece's holders without asking; only what moved after a failure is told.
// The master keeps the authoritative pool; nodes keep a copy from its reports. Not thread-safe.
#pragma once
#include "rhizome_identity.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <vector>

namespace xpu2::interactive::rhizome {

enum class Tier : uint8_t { Ram = 0, Vram = 1 };

struct Piece {
    Hash hash{};
    uint64_t bytes = 0;
    Tier prefers = Tier::Ram;
};

// Where a copy of a piece lives.
struct Copy {
    Hash node{};
    Tier tier = Tier::Ram;
    bool operator<(const Copy& o) const { return node != o.node ? node < o.node : tier < o.tier; }
    bool operator==(const Copy& o) const { return node == o.node && tier == o.tier; }
};

// A move the pool asks for after a change: copy a piece from a holder to a node (the copy's tier).
struct Move {
    Hash piece{};
    Hash from{};  // a node that holds it (all zeros: none left; the master sends it from the scene)
    Copy to;
};

class Pool {
public:
    unsigned replicas = 2;

    // A node's memory for the pool: what it gives of its RAM and of its cards' memory.
    void addNode(const Hash& node, uint64_t ramBytes, uint64_t vramBytes) {
        auto& n = nodes_[node];
        n.capacity[0] = ramBytes, n.capacity[1] = vramBytes;
        n.alive = true;
    }
    // A piece to keep: placed now. False when the swarm has no room for even one copy.
    bool add(const Piece& p) {
        pieces_[p.hash] = p;
        return place(p.hash, nullptr) > 0;
    }
    // A node failed or left: its copies are gone; each piece it held is copied again to keep Replicas (from a node
    // that still holds it). Returns the copies to make.
    std::vector<Move> fail(const Hash& node) {
        std::vector<Move> moves;
        auto it = nodes_.find(node);
        if (it == nodes_.end()) return moves;
        it->second.alive = false;
        it->second.used[0] = it->second.used[1] = 0;
        std::vector<Hash> touched;
        for (auto& [h, copies] : copies_) {
            const size_t before = copies.size();
            copies.erase(std::remove_if(copies.begin(), copies.end(), [&](const Copy& c) { return c.node == node; }), copies.end());
            if (copies.size() != before) touched.push_back(h);
        }
        for (const Hash& h : touched) place(h, &moves);
        return moves;
    }
    // A node joined: pieces short of Replicas (the swarm was short of room) are placed again.
    std::vector<Move> joined() {
        std::vector<Move> moves;
        for (auto& [h, p] : pieces_)
            if (copies_[h].size() < replicas) place(h, &moves);
        return moves;
    }
    const std::vector<Copy>& holders(const Hash& piece) const {
        static const std::vector<Copy> none;
        auto it = copies_.find(piece);
        return it == copies_.end() ? none : it->second;
    }
    // Whether a piece has lost every copy (a failure the replicas did not cover).
    bool lostAll(const Hash& piece) const { return pieces_.count(piece) && holders(piece).empty(); }
    uint64_t used(const Hash& node, Tier t) const {
        auto it = nodes_.find(node);
        return it == nodes_.end() ? 0 : it->second.used[int(t)];
    }
    uint64_t capacity(const Hash& node, Tier t) const {
        auto it = nodes_.find(node);
        return it == nodes_.end() ? 0 : it->second.capacity[int(t)];
    }
    // The swarm's pool: what all working nodes give, and hold, in a tier.
    uint64_t total(Tier t) const {
        uint64_t s = 0;
        for (const auto& [h, n] : nodes_)
            if (n.alive) s += n.capacity[int(t)];
        return s;
    }
    // The pieces a node holds (what it must keep resident for the others).
    std::vector<Hash> heldBy(const Hash& node) const {
        std::vector<Hash> out;
        for (const auto& [h, copies] : copies_)
            for (const auto& c : copies)
                if (c.node == node) {
                    out.push_back(h);
                    break;
                }
        return out;
    }

private:
    struct NodeMemory {
        uint64_t capacity[2] = {0, 0}, used[2] = {0, 0};
        bool alive = false;
    };
    // The rendezvous score of a node for a piece, weighted by the room it has: -room / ln(u) with u from a hash of the
    // pair, the highest first (weighted rendezvous hashing).
    static double score(const Hash& piece, const Hash& node, double room) {
        unsigned char m[64];
        std::memcpy(m, piece.data(), 32);
        std::memcpy(m + 32, node.data(), 32);
        const Hash h = sha256(m, sizeof m);
        uint64_t v = 0;
        std::memcpy(&v, h.data(), 8);
        const double u = (double(v >> 11) + 0.5) * 0x1p-53;
        return -room / std::log(u);
    }
    // Places copies of a piece until it has Replicas (each on a different node, the preferred tier first, else RAM).
    // Moves, when asked for, say where each new copy comes from. Returns the copies it has.
    size_t place(const Hash& h, std::vector<Move>* moves) {
        const Piece& p = pieces_[h];
        auto& copies = copies_[h];
        const Hash source = copies.empty() ? Hash{} : copies.front().node;
        auto holds = [&](const Hash& n) {
            return std::any_of(copies.begin(), copies.end(), [&](const Copy& c) { return c.node == n; });
        };
        while (copies.size() < replicas) {
            std::optional<Copy> best;
            double bestScore = -1;
            const Tier tiers[2] = {p.prefers, Tier::Ram};
            for (int pass = 0; pass < (p.prefers == Tier::Ram ? 1 : 2) && !best; ++pass) {
                const Tier t = tiers[pass];
                for (const auto& [node, n] : nodes_) {
                    if (!n.alive || holds(node)) continue;
                    const uint64_t room = n.capacity[int(t)] - std::min(n.capacity[int(t)], n.used[int(t)]);
                    if (room < p.bytes) continue;
                    const double s = score(h, node, double(room));
                    if (s > bestScore) bestScore = s, best = Copy{node, t};
                }
            }
            if (!best) break;  // no node has room: short of Replicas until one joins
            copies.push_back(*best);
            nodes_[best->node].used[int(best->tier)] += p.bytes;
            if (moves) moves->push_back(Move{h, source, *best});
        }
        return copies.size();
    }
    std::map<Hash, NodeMemory> nodes_;
    std::map<Hash, Piece> pieces_;
    std::map<Hash, std::vector<Copy>> copies_;
};

}  // namespace xpu2::interactive::rhizome
