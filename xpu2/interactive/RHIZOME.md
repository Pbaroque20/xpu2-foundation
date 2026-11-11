# Rhizome: XPU2 across machines

Rhizome renders one picture on many machines. It is a swarm: every machine that can contributes, finds the others,
takes work and passes it on by the same cellular rules XPU2 uses between a processor and a graphics card. One
machine, chosen by the user, is the **master integrator**: it holds the scene indexer, sends the cells out, adds what
comes back into the final picture and audits it.

Rhizome is the outer tier of the multi-device extension (`MULTI_DEVICE.md`). Inside a machine XPU2 already shares a
frame between processor groups, NUMA nodes and graphics cards. Rhizome adds machines above them. A machine is a node of
the swarm as a NUMA node is a node of the machine: the work it is given is near it, and the farther it is, the larger
and less urgent its share.

This note is the design and where it stands. How to drive Rhizome (the command line, a machine's folder and its
`status.json`, the job queue, the C++ class) is in [RHIZOME_API.md](RHIZOME_API.md). The code is `rhizome_*.h`;
`rhizome_test.cc` checks it:

```
g++ -std=c++20 -O2 -o rhizome_test.exe xpu2/interactive/rhizome_test.cc -lbcrypt -lws2_32 -liphlpapi
```

(the toolchain's `msys64/ucrt64` g++). `RHIZOME_TEST_LOG=1` prints what each node does.

## 1. Nodes and their hashes

Every node has a unique hash, its identity in the swarm. A node makes an ECDSA P-256 key pair the first time it runs
(Windows CNG, `rhizome_identity.h`) and keeps it in its folder. Its hash is the SHA-256 of its public key: 32 bytes,
written as 64 hex digits. Two nodes on one machine have two keys and two hashes, and a node keeps its hash across
restarts.

Every message a node sends carries its hash and is signed with its key (`rhizome_wire.h`). A receiver checks that the
signature verifies with the public key the node introduced itself with, and that this key hashes to the hash it
claims. So a node cannot speak for another, every sample in the picture is known by the node that made it, and the
master can drop a node by its hash.

A swarm also has a rhizome key, a secret every member is given. A node that cannot prove it holds the key (a keyed hash
of the other's challenge, in the handshake) is not let in. Rhizome is meant for a studio's own network; it is not built
to resist a hostile machine inside it beyond what the audit catches.

The traffic is encrypted (`rhizome_crypto.h`). Each connection makes a key exchange of its own: the two sides send a
one-time ECDH P-256 key in their signed Hello and Welcome, so no machine in between can put its own in, and from the
shared secret, the two nodes' hashes and the rhizome key come two AES-256 keys, one each way. Every frame after the
handshake is sealed with AES-GCM, its nonce the count of frames that way: a frame changed, cut short, played again or
out of order does not open, and the connection is closed. Scene files, pieces and films cross the network unread; the
handshake's two frames, which carry only keys, hashes and what a machine has, are the only ones in the open.

## 2. The swarm

A node starts with the address of any member (usually the master). It introduces itself (hash, public key, what it
has: processor threads, NUMA nodes, graphics cards, a measured rate), learns the members from the one it reached, and
connects to the others. Members are kept by heartbeats: a member not heard from for a few heartbeats is suspected,
then dropped (bytes of a message still arriving count as hearing it, so a large piece on a slow link is not silence;
a machine is suspected after 2 s of silence and dropped after 6 s). The swarm's threads run above the renderer's, so a
machine whose every core renders still answers on time: at normal priority its heartbeats waited seconds for a core
and a node took over from a live master. Log lines are written by a thread of their own, never under the node's lock (a write
that waited on the disk held a master silent 30 s); a loop held more than a second says so. A frame's end is taken from the
node's master alone (`rhizome_swarm.h`, `Members`). The master's list is the one that counts when lists differ.

A node joins and leaves at any time. One that joins mid-frame takes work from the next steals; one that leaves or
falls silent has its unreturned work given out again (section 4).

The distance between two nodes is their round-trip time, measured on the heartbeats. It is to Rhizome what a NUMA
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
  each node's cells kept together (`rhizome_schedule.h`, `Plan`).

The master holds the final film. Each node's contribution (its sums, odd sums and counts per pixel, as a graphics
card's film is added into the first card's, `GpuLane::addFollowers`) is added in as it arrives, after the audit
passes it.

The scene reaches the nodes through the swarm's memory (section 4b).

## 3b. When the master fails

The master names its successors as a frame begins: the frame's members by their measured rate, the fastest first,
three at most. It signs the list and sends it to every node. With approvals each successor is also endorsed as an
approver, so whichever takes over can approve and revoke in its turn.

The first successor is the deputy. Each film the master takes in (audited, the first of its chunk) goes to the deputy
too, which keeps the picture, the counts, the chunks back and the merged AOVs as the master does. A deputy named later
(the first one gone) is given all of it at once.

When the nodes find the master silent, the first successor still here takes over: its copy becomes the master's
picture and ledger, it makes the memory pool again from the scene it holds whole, and it tells the others, showing
the old master's signed list. A node takes the new master when that list verifies against the old master's key, the
new master is in it, and every successor before it is gone. The nodes then report the chunks they hold; two seconds
on, what no one holds and has not come back (the old master's own, films that were on their way to it) is given out
again. The new master finishes the frame and writes the picture and the scene's outputs (its folder's
`output\<scene>.pfm`). `rhizome status` shows the succession and the deputy's copy.

## 4. Cells, ownership and stealing

The picture is cut into rhizome cells (256 px by default, four of a machine's grid cells across). A cell's work is its
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

The swarm's RAM and graphics card memory are one pool (`rhizome_memory.h`). The master splits the scene into pieces
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
(its hash refused for the rest of the session) and its ranges given out again. The node is told so before its
connection closes; a refused node takes no more work and never takes over as master. A comparison allows for the
difference between a processor's and a graphics card's floating point, not for a different picture. A pixel's sums
differ by a few tenths of a percent between the two, and now and then by more: of a contribution's few samples, a path
through a volume can turn another way on a last bit. So a node is judged on all its pixels checked so far, not on
one contribution:
- **Exact, at once:** each pixel has exactly its samples, none of them NaN or infinite.
- **Outliers:** a pixel off by more than 5% is an outlier. A node is refused when its outliers pass 5% of its pixels
  checked by four standard deviations and one pixel more. Graphics cards on a volume scene give about 0.1-0.2%.
- **Bias:** the mean signed difference of the pixels within 5% stays within 0.2%, or four standard errors where they
  are few. A contribution made brighter or darker everywhere fails it; one changed by more than 5% is all outliers.

The master logs each node's audit after every frame: pixels checked, outliers, bias and the worst difference.

## 6. Where it stands

| Part | State |
|---|---|
| Node hash, key pair kept in the node's folder, signed messages, rhizome key handshake (`rhizome_identity.h`, `rhizome_wire.h`) | built, tested |
| Members, heartbeats, suspicion, distance by round trip (`rhizome_swarm.h`) | built, tested |
| Tiers by rate, important cells to the fast tier, cells kept together, stealing by neighbours, reassignment (`rhizome_schedule.h`) | built, tested |
| Audit by rendering samples again, quarantine (`rhizome_audit.h`) | built, tested |
| Memory pool: RAM and card memory across the nodes, two copies a piece, copied again when a machine fails (`rhizome_memory.h`) | built, tested |
| Loopback swarm: nodes over TCP on one machine, a stand-in renderer, the master's film, pieces moved between nodes (`rhizome_node.h`, `rhizome_test.cc`) | built, tested (96 checks) |
| The XPU2 renderer as a node's renderer: a share of the frame (a cell's pixels, a range of its sample indices), its film back; the film's rows alone cleared and downloaded (`async_render.h`, `async_gpu.h`, `rhizome_session.inc`) | built; tested on a Modo scene exported to MoonRay (beauty as one machine's) |
| AOVs and aux outputs: each node's accumulators merged at the master by each output's filter, resolved once, the scene's outputs written | built; Cryptomatte merged too (plan 11) |
| The scene's files (and the files they name, UDIM sets too) as pieces in the pool; a node fetches them and points the scene at its copies | built |
| A machine's own processor, cards and NUMA nodes inside a node (its own grid, as now) | as now: a node is a whole XPU2 |

### The plan

| # | Part | State |
|---|---|---|
| 1 | Command line by verbs: `moonray_xpu2 rhizome node \| master \| id \| trust \| approve \| revoke \| status`, each with `--help` (`--rhizome-node`, `--rhizome-master` kept) | built; tested end to end (help, id, trust, approve, revoke, status, node, master) |
| 2 | Headless nodes: no scene and no master given, a hash made on first start, listening, rendering each job and waiting for the next | built; tested: two nodes, two jobs one after another on them |
| 3 | Discovery on every adapter, a direct cable's own 169.254 address too, answered only by members (`rhizome_discovery.h`); a master given `--nodes host:port,...` where broadcast does not reach | built; tested: a master found two nodes on one machine through its network address; a direct cable not tried yet (19) |
| 4 | Signed approvals: endorsements and revocations, approver lists on every machine, `--approve`, `--trust-first`, `--trust` (`rhizome_trust.h`, 18 checks) | built; tested: held until approved, endorsed, a master the nodes do not trust refused, a revoked node refused; 19 + 9 checks |
| 5 | Status: this machine, the frame, each machine's role, state, tier, rate, progress and pool share, the deputy's copy and the succession, the memory pool; `--watch` | built (status.json, status.txt, `--watch`, the succession and the deputy's copy) |
| 6 | Master failover: a signed succession list, the deputy's copy of film, AOVs and ledger, a takeover by the first live successor, approvals carried by its approver endorsement | built; tested: the master killed a third of the way into a test scene, a node took over and finished it as one machine renders it; unit checks for a late and an early failure |
| 7 | Matching builds: the handshake compares the XPU2 build and shader hashes and refuses a machine whose samples would differ | built: the executable and its GPU programs hashed; tested in the unit checks |
| 8 | Encrypted traffic: a key exchange each connection (ECDH, Windows CNG) and AES-GCM, so scene files cross the network unread | built; tested: 15 checks (agreement, keys each way, sealing, a frame played again, out of order, changed, plain), and every swarm's traffic sealed after its handshake |
| 9 | The indexer's probe as the master's importance (the first-hit probe per cell) | built (`rhizome_importance.h`): a camera ray every 8 px on the master's processor, each cell scored by its edges, the objects it sees and their slow features (hair, skin, glitter, layers, maps, displacement, cut-outs, mesh lights, volumes, glow); 8 checks; a test scene's probe 0.01 s, 26 of 49 cells important |
| 10 | Adaptive sampling across machines: the nodes' errors pooled at the master, as the cards' are (`GpuLane::poolErrors`) | built: an adaptive scene's shares send their odd sums; the master works out each pixel's error from every node's films added, and a cell under the scene's target (one pixel in a hundred aside) has its chunks not back cancelled on every node (and on the deputy); 6 checks: the flat half of a frame stopped at its minimum, the noisy half took every sample, 56% of a uniform frame's samples |
| 11 | Cryptomatte: each node's fragments merged at the master | built: a share's fragments stay sums; nodes send them with each film, the master merges them by ID (as the lanes' are) and finalizes once; the deputy keeps a copy; a share's film is given its fragments anew each share (it had none); a test scene across three processes: every Cryptomatte ID as the single machine's, coverages 99.96% equal (the sums' order) |
| 12 | Workstations: a node stepping back while its machine is in use (paused, or fewer threads), hours it may render in | built: `--when-idle s`, `--hours 19-8`, `--busy-threads n`; stepped back, a node gives its chunks to the master, which gives them out and plans it no cells until it is back; status shows it stepped back; 3 checks |
| 13 | Animation and a job queue: frame ranges, jobs one after another, the scene kept between frames and its changes alone sent | built (`rhizome_jobs.h`): `--frames 1-24` (`1-24:2`, `1,5,9`); a path with a run of # is a file a frame (`-in shot.####.rdla`: the scene made each frame, its unchanged files not sent again, the pool's pieces being their own hashes); `--delta shot.####.rdla`: each frame's changes read on top of the frame before, the scene kept on every machine (a node joining late reads them all); `-out beauty.####.pfm`, the scene's outputs likewise; `--queue folder [--queue-watch]` and `rhizome submit`; a frame number on every plan and film (a late film of the frame before is left out) and readiness for each frame's manifest; 23 checks; tested on the VDB scene, three processes, CPU alone and with the graphics card: frames 1-3 by deltas and a queued job of two frames given whole, each frame as one machine renders it (block error 0) |
| 14 | Checkpoints: the master's film and ledger saved as it goes, a job taken up again after every machine has stopped | built: the master saves the job (the XPU2 build and the job, hashed) in its folder every 60 s (`--checkpoint s`) and after each frame: its frames done, and the frame under way (picture, counts, odd sums, the chunks back, the merged outputs, the frame's manifest hash); started again, it takes the job up where it was, the plan's `done` chunks given to no one (other machines may take the rest), a finished job skipped, the frames done of a delta job read so the next is right; a checkpoint is not used once the job's scene or delta files were written again, nor a partial frame of another scene (`--fresh`: anew); queue jobs left running put back; 5 checks; tested: every machine killed part way into frame 2, then into a queued job's frame 4, each taken up (16 and 19 of 80 chunks from the checkpoint, the rest rendered once), every frame as one machine renders it |
| 15 | Shaders on nodes: a node without compiled GPU programs offered to compile them on first start; status says which lack them | built: a machine (node or master) whose cards have no GPU programs compiled renders on its processor alone and, on its first start, is asked whether to compile them (`--shaders ask`, the answer kept in its folder; no answer in 30 s: asked again next start; `--shaders compile` or `skip` decide it beforehand); the compile runs as a process of its own at low priority (its lines in `shaders-compile.txt`), the cards rendering from the next job after, and a start while an earlier one's compile runs does not start another; each machine's GPU programs (no graphics card, compiled, missing and which cards, compiling and how far, declined) go in its heartbeats and show in every machine's status; 5 checks; tested: three processes, one node with none compiled that said yes (compiled, its card rendering), one that declined (the processor alone), the master's status showing each |
| 16 | Real distributed memory: texture tiles fetched from the pool as a node renders, in place of a whole copy of the scene each | built (`remote_files.h`): a node's textures (.tx) stay in the swarm's memory as pieces of 256 KB, an empty file of each name in its place; XPU2's texture readers read them through an OpenImageIO IOProxy, each piece fetched (from a holder, else the master) the first time a tile read touches it, so the tiles a render never reads never come (`--whole-textures`: copied whole first, as before); a render that waited for pieces leaves its time out of the machine's rate; the pool's placements sent once for a scene's pieces (a list a piece made a scene of n pieces n squared, and the nodes asked again for each piece each time); tested: a test scene's three textures (121 MB) read from the pool by a CPU node and a GPU node, the picture and every output as one machine renders them |
| 17 | A cell given to a node that holds its pieces (memory affinity in the plan) | built: the master counts each cell's films by machine; the next frame of the same cells gives each cell back to the machine that rendered most of it (the texture pieces its pixels read fetched there, its caches warm) while that machine is in the plan, of the cell's tier and short of its share by rate, then deals the rest along the Morton curve as before; 4 checks (rates moved; a node gone; a node joining: 91 of 128 cells kept against 70 without); tested: three frames of a test scene with the camera moving, 35 of 49 cells kept with their machine in frames 2 and 3 |
| 18 | Routed networks and VPNs beyond broadcast; a firewall rule its user adds once; macOS with the platform roadmap | built for Windows: a master remembers every machine it reached (`known.txt` in its folder) and probes each one by its address at the next start; `--scan 10.8.0.0/24,...` probes ranges address by address (at most /16 each, a few hundred a second), `--scan` alone each adapter's subnet; `--no-broadcast`; nodes on one machine each take a discovery port of their own (47101-47104), as a probe sent to one address reaches one socket; `moonray_xpu2 rhizome firewall` adds one inbound rule for this executable on private and domain networks (`--public` too), asking Windows for an administrator, `--check` (read from the firewall's own list), `--print`, `--remove`; a node or master on a network says at its start when the rule is missing; 3 checks; tested on one machine, broadcast off: two nodes found by a sweep, then by the remembered address alone; the rule itself not added in a test (it changes the machine's security settings); macOS waits for the platform roadmap |
| 19 | A test on two machines, a direct cable among them | todo |

The kit's side (a Rhizome render from Modo, its nodes in a panel built from status, a machine as a node while idle) is
MoonLightIPR's, not this branch's.
