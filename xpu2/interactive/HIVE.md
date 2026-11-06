# Hive: XPU2 across machines

Hive renders one picture on many machines. It is a swarm: every machine that can contributes, finds the others,
takes work and passes it on by the same cellular rules XPU2 uses between a processor and a graphics card. One
machine, chosen by the user, is the **master integrator**: it holds the scene indexer, sends the cells out, adds what
comes back into the final picture and audits it.

Hive is the outer tier of the multi-device extension (`MULTI_DEVICE.md`). Inside a machine XPU2 already shares a
frame between processor groups, NUMA nodes and graphics cards. Hive adds machines above them. A machine is a node of
the swarm as a NUMA node is a node of the machine: the work it is given is near it, and the farther it is, the larger
and less urgent its share.

This note is the design and where it stands. The code is `hive_*.h`; `hive_test.cc` checks it:

```
g++ -std=c++20 -O2 -o hive_test.exe xpu2/interactive/hive_test.cc -lbcrypt -lws2_32
```

(the toolchain's `msys64/ucrt64` g++). `HIVE_TEST_LOG=1` prints what each node does.

## 1. Nodes and their hashes

Every node has a unique hash, its identity in the swarm. A node makes an ECDSA P-256 key pair the first time it runs
(Windows CNG, `hive_identity.h`) and keeps it in its folder. Its hash is the SHA-256 of its public key: 32 bytes,
written as 64 hex digits. Two nodes on one machine have two keys and two hashes, and a node keeps its hash across
restarts.

Every message a node sends carries its hash and is signed with its key (`hive_wire.h`). A receiver checks that the
signature verifies with the public key the node introduced itself with, and that this key hashes to the hash it
claims. So a node cannot speak for another, every sample in the picture is known by the node that made it, and the
master can drop a node by its hash.

A swarm also has a hive key, a secret every member is given. A node that cannot prove it holds the key (a keyed hash
of the other's challenge, in the handshake) is not let in. Hive is meant for a studio's own network; it is not built
to resist a hostile machine inside it beyond what the audit catches.

## 2. The swarm

A node starts with the address of any member (usually the master). It introduces itself (hash, public key, what it
has: processor threads, NUMA nodes, graphics cards, a measured rate), learns the members from the one it reached, and
connects to the others. Members are kept by heartbeats: a member not heard from for a few heartbeats is suspected,
then dropped (`hive_swarm.h`, `Members`). The master's list is the one that counts when lists differ.

A node joins and leaves at any time. One that joins mid-frame takes work from the next steals; one that leaves or
falls silent has its unreturned work given out again (section 4).

The distance between two nodes is their round-trip time, measured on the heartbeats. It is to Hive what a NUMA
node's distance is to a machine: near nodes take small chunks, far ones larger ones, so that a node spends far more
time rendering than waiting.

## 3. The master integrator and its indexer

The user names the master. It is the only node that reads the scene, so the only one with the scene indexer
(`INDEXER.md`). Before a frame it:

- probes the picture as the indexer's first-hit probe does (`device_suit.h`): what each cell sees, its families, and
  from them a cost and an importance;
- scores each cell's feature importance: edges and silhouettes, many materials or lights in view, detailed maps,
  volumes and glowing volumes, hair, subsurface; once the first samples are in, the picture so far (as the grid's
  important parts first, `async_grid.h`) replaces the probe's guess;
- sorts the nodes into tiers by rate, and gives the important cells to the fast tier and the rest to the slow tier,
  each node's cells kept together (`hive_schedule.h`, `Plan`).

The master holds the final film. Each node's contribution (its sums, odd sums and counts per pixel, as a graphics
card's film is added into the first card's, `GpuLane::addFollowers`) is added in as it arrives, after the audit
passes it.

The scene reaches the nodes through the swarm's memory (section 4b).

## 4. Cells, ownership and stealing

The picture is cut into hive cells (256 px by default, four of a machine's grid cells across). A cell's work is its
sample ranges. Each cell has exactly one owner at a time, and only the owner hands out the cell's ranges, so no
range is rendered twice: the swarm's version of the grid's single claim (CA-06).

A node works by the cellular rules: finish the cell it is in, then its own cells, then it asks a neighbour for work.
Its neighbours are the nodes owning cells next to its own in the picture, nearest by distance first. The owner gives
part of a cell's remaining ranges (half, at least one chunk) and tells the master in its next report. A fast node
may take a slow node's important work; a slow node may take a fast node's unimportant work but not its important
cells while the fast node is keeping up.

Ownership flips between frames towards the node that rendered most of a cell, as the grid's flips do (CA-10).

A node that leaves, falls silent, or fails an audit loses its cells: the master knows which ranges came back, and
gives the rest to the nodes nearest to them.

## 4b. Memory across the swarm

The swarm's RAM and graphics card memory are one pool (`hive_memory.h`). The master splits the scene into pieces
(the scene file's blocks, meshes, texture tiles), each named by its SHA-256, and places them on the nodes by the
memory each gives: a piece a graphics card reads (a mesh, a texture tile for the GPU) in a card's memory where one
has room, the rest in RAM. Each piece is kept on two nodes, never both copies on one, and no node holds more than it
gives. A scene too large for any one machine then fits in the swarm.

Placement is rendezvous hashing weighted by each node's free memory, so every node works out where a piece is from
the members alone, and a node joining or leaving moves only the pieces it takes or held. A node fetches a piece it
needs from a holder, and checks it against its hash.

When a machine fails, its copies are gone, and each piece it held is copied again, from the node that still holds
it, to a node with room: allocation follows the machines as they come and go. A swarm short of room keeps the copies
that fit and makes the rest when a node with room joins. Only a piece whose every holder fails at once is lost; the
master, which keeps the scene, then sends it again.

Within a machine the pieces it holds live as the multi-device extension keeps them: in the heap of the NUMA node that
uses them, and in each card's memory for that card.

## 5. The audit

Every XPU2 sample is defined by its identity (pixel, index, stream; `sample_stream.h`), so any machine renders a given
sample to the same value. The master audits each contribution by rendering again a few of its samples, chosen after
the contribution arrives by a seed the nodes do not know, and comparing them. One sample in a thousand by default,
and every sample of a node's first contribution's first pixel row.

A contribution that matches is added to the film. One that does not is thrown away whole, its node is quarantined
(its hash refused for the rest of the session) and its ranges given out again. A comparison allows for the
difference between a processor's and a graphics card's floating point (a relative tolerance), not for a different
picture.

## 6. Where it stands

| Part | State |
|---|---|
| Node hash, key pair kept in the node's folder, signed messages, hive key handshake (`hive_identity.h`, `hive_wire.h`) | built, tested |
| Members, heartbeats, suspicion, distance by round trip (`hive_swarm.h`) | built, tested |
| Tiers by rate, important cells to the fast tier, cells kept together, stealing by neighbours, reassignment (`hive_schedule.h`) | built, tested |
| Audit by rendering samples again, quarantine (`hive_audit.h`) | built, tested |
| Memory pool: RAM and card memory across the nodes, two copies a piece, copied again when a machine fails (`hive_memory.h`) | built, tested |
| Loopback swarm: nodes over TCP on one machine, a stand-in renderer, the master's film, pieces moved between nodes (`hive_node.h`, `hive_test.cc`) | built, tested (52 checks) |
| The XPU2 renderer as a node's renderer: a frame of a set of cells and sample ranges, its film back | todo |
| The indexer's probe as the master's importance (the first-hit probe per hive cell) | todo |
| The scene's own pieces (its file's blocks, meshes, texture tiles) in the pool, and a node's XPU2 reading them | todo |
| A cell given to a node that holds its pieces (memory affinity in the plan) | todo |
| Discovery on the local network (a broadcast beacon) besides a known member's address | todo |
| A machine's own processor, cards and NUMA nodes inside a node (its own grid, as now) | as now: a node is a whole XPU2 |
