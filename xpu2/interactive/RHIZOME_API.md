# Rhizome's API

What a program, a script or a person can rely on to drive Rhizome, and what each part does. How Rhizome works inside
is in [RHIZOME.md](RHIZOME.md).

There are four ways in:

| Way in | For | Stable |
|---|---|---|
| [The command line](#1-the-command-line) (`moonray_xpu2 rhizome <verb>`) | people, scripts, a host application's window | yes |
| [A machine's folder](#2-a-machines-folder) and its [`status.json`](#3-statusjson) | reading a running node or master | `status.json` format 1, `known.txt`, `approvers.txt`; the rest is Rhizome's own |
| [The job queue](#4-the-job-queue) (a folder of `.job` files) | sending jobs to a master | yes |
| [The C++ class](#6-the-c-class) (`rz::RhizomeNode`) | building Rhizome into another program | follows the source |

The network itself (section 5) is not an API: machines talk only to machines of the same XPU2 build.

## 1. The command line

```
moonray_xpu2 rhizome node     run a headless node: render for whichever master reaches it
moonray_xpu2 rhizome master   render a scene, a frame range or a queue of jobs with the machines found
moonray_xpu2 rhizome submit   add a job to a master's queue folder
moonray_xpu2 rhizome id       print this machine's hash
moonray_xpu2 rhizome trust    trust an approver
moonray_xpu2 rhizome approve  endorse a machine (an approver only)
moonray_xpu2 rhizome revoke   revoke a machine (an approver), or block it here (anyone else)
moonray_xpu2 rhizome status   the swarm as this machine sees it
moonray_xpu2 rhizome firewall the Windows Firewall rule that lets other machines reach this one
```

Each verb prints its own help with `--help`. `--rhizome-node` and `--rhizome-master` are older names for `rhizome node`
and `rhizome master`.

**Exit codes.** 0: done. 1: an error, its reason on standard error as one line starting `MoonRay XPU2 Rhizome:`.
Codes of particular verbs are given with them below. A wrong argument is an error (1), with the verb's help.

**Output.** Lines on standard output starting `MoonRay XPU2 Rhizome:` are a log for people. Programs read
`status.json`, not the log.

### Options of `node` and `master`

| Option | Default | What it does |
|---|---|---|
| `--key <text>` | (needed) | The rhizome key: only machines given the same key meet. It is never sent. |
| `--folder <f>` | `rhizome\node` or `rhizome\master` beside the executable | The machine's folder: its hash, approvals, status (section 2). One folder for each running node or master. |
| `--mode 0\|1\|2` | 2 | What renders here: 0 the processor alone, 1 the graphics cards alone, 2 both. |
| `-threads <n>` | every processor thread | Processor threads to render with. |
| `--bind <host>` | 0.0.0.0 | The address to listen on (0.0.0.0: every network; 127.0.0.1: this machine only). |
| `--port <p>` | 47100 (node), 47110 (master) | The TCP port. A default port that is taken is replaced by one the system picks; a port given is not. |
| `--advertise <host>` | the bound address | The address other machines are told to reach this one at. |
| `--trust <hash>` | | An approver this machine trusts (again for more). |
| `--no-discovery` | | No discovery at all: a node does not answer probes, a master sends none. |
| `--shaders ask\|compile\|skip` | ask | Graphics cards with no GPU programs compiled: ask on the first start (the answer kept in the folder), compile them in the background now, or not. |
| `--whole-textures` | | A node copies the scene's textures whole before it renders, instead of reading their tiles from the swarm's memory as it renders. |

A node also takes:

| Option | What it does |
|---|---|
| `--master <host:port>` | Reach that master, instead of waiting to be found. |
| `--when-idle <s>` | Render only once no one has used the keyboard or mouse for that many seconds. |
| `--hours <from-to>` | Render only in those hours, 19-8 across midnight too. |
| `--busy-threads <n>` | While the machine is in use, render on that many threads instead of stepping back. |

A master also takes:

| Option | Default | What it does |
|---|---|---|
| `-in <scene>` | | The scene's files (again for more): `.rdla` or `.rdlb`. A path with a run of `#` is a file a frame. Needed unless `--queue` is given. |
| `-out <picture>` | | Where the beauty goes, as PFM whatever the extension; a run of `#` takes the frame number. The scene's own render outputs are written as the scene names them. Needed with `-in`. |
| `--frames <list>` | the scene as it is | `12`, `1-24`, `1-24:2` (every second frame), `1,5,9-12`. |
| `--delta <file.####.rdla>` | | A frame's changes, read on top of the frame before (again for more). |
| `--queue <folder>` | | Then render the jobs in that folder, one after another (section 4). |
| `--queue-watch` | | Keep waiting for new jobs once the queue is empty. |
| `--nodes <n>` | 0 | Start the first frame once n nodes are ready. 0: once every node found is ready and none joins for 3 s. |
| `--node <host:port>` | | A node to reach by its address (again for more). It is remembered for the next start. |
| `--wait <s>` | 30 | The longest wait for the nodes before the first frame starts without them. |
| `--cell <px>` | 256 | The size of the cells the frame is cut into, a multiple of 64. |
| `--chunk <samples>` | 16 | Samples a chunk: the smallest piece of work given out. |
| `--approve` | | Only approved machines join; the others wait (`rhizome approve`). |
| `--trust-first` | | Endorse every machine met on this run. |
| `--checkpoint <s>` | 60 | Save the job in the folder every so many seconds and after each frame (0: never). Started again on the same job, the master takes it up where it was. |
| `--fresh` | | Begin the job anew, its checkpoint left unused. |
| `--scan [ranges]` | | Probe addresses one by one: `10.8.0.0/24,10.9.0.0/24` (at most /16 each), or each network adapter's subnet when no range is given. For VPNs and routed networks. |
| `--no-broadcast` | | Send no broadcast probes. |

A node runs until it is stopped. A master waits for each frame as long as it takes (machines that join meanwhile help),
and exits once its jobs are done: 0, or 1 on an error (a scene that does not read, a picture that cannot be written). A
queue's job that fails goes to `failed\` with why, and the master goes on to the next. A master stopped part way
takes its job up again when started on it again (`--checkpoint`).

### The other verbs

| Verb | What it does | Prints | Exit |
|---|---|---|---|
| `submit --queue <f> -in <scene> -out <picture> [--frames] [--delta] [--name]` | Writes a job file into the queue folder. Paths are made absolute here, so the master must reach them the same way. | `MoonRay XPU2 Rhizome: job <name> queued: <file>` | 0 |
| `id [--folder f]` | Makes the machine's hash the first time. | Its hash (64 hex digits) on the first line; a note in parentheses on the second. | 0 |
| `trust <hash> [--folder f]` | Trusts that approver: its endorsements and revocations count here. | One line | 0 |
| `approve <hash> [--approver] [--folder f]` | Endorses that machine with this machine's key. `--approver`: it may approve in its turn. A master running from the folder admits it within a second. | One line | 0; 2 when this machine is not an approver |
| `revoke <hash> [--folder f]` | An approver revokes the machine for the whole swarm; any other machine blocks it here only. | One line | 0 |
| `status [--folder f] [--watch] [--json]` | The table a running node or master writes each second; `--json`: its `status.json` (section 3). `--watch`: again each second. | The table, or the JSON | 0 while one runs from the folder; 3 when none does |
| `firewall [--check] [--print] [--remove] [--public]` | Adds one inbound rule letting this executable take connections on private and domain networks (`--public`: public ones too), asking Windows for an administrator. `--check`: whether the rule is there (nothing changed). `--print`: the commands to run by hand. `--remove`: the rule taken away. | One line | 0; 3 `--check` found none; 4 the administrator prompt was declined or did not open; 5 not an administrator |

When nothing runs, `status --json` prints `{"rhizome": 1, "running": false, "hash": "<the folder's hash, or empty>"}`.

### XPU2's own commands Rhizome uses

| Command | What it does | Exit |
|---|---|---|
| `moonray_xpu2 --compile-shaders` | Compiles the GPU programs for every graphics card in the machine (an hour the first time). A node with `--shaders compile` runs it in the background. | 0 done, 1 failed |
| `moonray_xpu2 --shaders-compiled` | Whether they are compiled for this driver and every card here. | 0 yes; 4 out of date (it says why) |

## 2. A machine's folder

Each running node or master has a folder (`--folder`). Two running from one folder would take each other's hash: give
each its own.

| File | Written by | What it holds | Others may |
|---|---|---|---|
| `node.key` | `id`, or the first start | The machine's key pair. Its hash is the SHA-256 of the public key. Keep it: a new key is a new machine, unapproved. | not read or copy |
| `status.json` | the running node or master, each second | The swarm as it sees it (section 3). | read |
| `status.txt` | the same | The same, as the table `status` prints. | read |
| `approvers.txt` | `trust` | The approvers trusted here, a hash a line. | write, one hash a line |
| `endorsed\`, `revoked\`, `blocked\` | `approve`, `revoke`, the swarm | Signed endorsements and revocations, one file each; blocks are this machine's alone. | read |
| `known.txt` | a master | The IPv4 addresses of the machines it reached, newest first, 256 at most, probed at each start. | write, an address a line |
| `checkpoints\<hash>.rzj` | a master | Each job's saved state (Rhizome's own format). | delete, to begin a job anew |
| `shaders.txt` | the first start | `compile` or `skip`: the answer to the GPU programs question. | write, or delete to be asked again |
| `shaders-compile.txt` | a background compile | Its output. While a compile runs it holds this file. | read |

A node also stages each job's scene in the system's temporary folder (`xpu2-rhizome-<process id>`).

## 3. status.json

Written whole each second, then named, so a reader never sees half a file. Format 1:

```json
{
  "rhizome": 1,
  "written": 1762800000,
  "hash": "<this machine, 64 hex digits>",
  "role": "master",
  "address": "192.168.1.64:47110",
  "approver": true,
  "master": "<the master's hash; empty before a node has one>",
  "job": "shot.rdla frame 12",
  "frame": {"running": true, "done": false, "chunks": 784, "back": 310, "held": 12, "audits": 91,
            "width": 1920, "height": 1080, "spp": 64, "cells": 40},
  "pool": {"ram": 34359738368, "vram": 25769803776, "pieces": 118, "held": 118},
  "members": [
    {"hash": "...", "self": true, "address": "192.168.1.64:47110", "role": "master", "state": "alive", "tier": "fast",
     "rate": 4.1e7, "rtt": 0, "chunks": 120, "ram": 0, "vram": 0, "ramGiven": 0, "vramGiven": 0, "gpuPrograms": "compiled"}
  ]
}
```

| Field | Meaning |
|---|---|
| `rhizome` | The format's number. A reader refuses formats it does not know; fields may be added within one. |
| `written` | When it was written, in Unix seconds. Older than 5 s: the node or master is not running. |
| `role` | `master` or `node`. A node that took over from a failed master says `master`. |
| `master` | The master's hash as this machine knows it (its own, for a master). |
| `job` | The job and frame under way, as the log names them. |
| `frame` | Present always; its size fields only while a frame has a plan. `chunks`: the frame's chunks in all; `back`: those whose films the master has (a master's alone); `held`: chunks this machine holds still; `audits`: films the master has checked. |
| `pool` | The swarm's memory as this machine knows it: bytes of RAM and graphics card memory given, pieces of the scene, pieces held here. |
| `members` | A row a machine: this one first (`self` true), then the others it knows, then those waiting for approval. |

A member's row:

| Field | Meaning |
|---|---|
| `role` | `master`, `node`, `quarantined` (its films failed the audit), `revoked`, `blocked` (here only) or `waiting` (for approval). |
| `state` | `alive`, `suspect` (silent a moment), `gone`, `stepped back` (its machine in use), `refused` or `held` (waiting). |
| `tier` | `fast` or `slow` in this frame's plan; empty when it is not in the plan. |
| `rate` | Samples a second, measured (0: not yet). |
| `rtt` | Seconds to this machine and back (0 for itself; below 0: not measured). |
| `chunks` | Chunks of this frame its films brought (a master's count). |
| `ram`, `vram` | Bytes of the pool it holds now. `ramGiven`, `vramGiven`: bytes it gives the pool. |
| `gpuPrograms` | Its GPU programs: `no graphics card`, `not used (the processor alone)`, `compiled`, `missing: <cards>`, `missing: <cards> (declined here)`, `compiling` or `compiling: <done> of <all> tasks, <s> s`. Empty: not heard yet. |

## 4. The job queue

A folder a master renders from (`rhizome master --queue <folder>`). A job is a text file ending in `.job`:

```
# a Rhizome job (moonray_xpu2 rhizome submit)
name shot 12
in C:\jobs\shot.####.rdla
frames 1-24
out C:\jobs\out\beauty.####.pfm
```

A line a field: `in` and `delta` (again for more), `frames`, `out`, `name`; `#` starts a comment. The fields mean what the
master's options of the same names mean. Any other field is an error.

| Where the file is | What it means |
|---|---|
| `<folder>\*.job` | Waiting. Jobs are rendered in the order of their names: `submit` names them by the time (`20261110-221503-127-000.job`). |
| `<folder>\running\` | The master is rendering it. A master started again puts these back to wait. |
| `<folder>\done\` | Finished, with a last `# done: ...` line. |
| `<folder>\failed\` | Not finished, with a last `# failed: <why>` line. |

Any program may add a job: write it as `<name>.job.part`, then rename it to `<name>.job`, so a master never reads half of
one.

## 5. The network

For firewalls and network people. Not an API: machines talk only with machines of the same XPU2 build.

| Port | Protocol | What it carries |
|---|---|---|
| 47100 (node), 47110 (master) | TCP | Everything between machines: signed with each machine's key, and sealed (ECDH, AES-GCM) after the handshake. Another port when `--port` says so, or when a default one is taken. |
| 47101-47104 | UDP | Discovery: a master's probe, and a node's answer with its hash and TCP port, keyed by the rhizome key. Nodes on one machine take one port each. |

A master probes every adapter's broadcast address each second, the machines in `known.txt` one by one, and the
`--scan` ranges at a few hundred addresses a second. Answers it cannot check against the rhizome key are ignored.

## 6. The C++ class

`rz::RhizomeNode` (`rhizome_node.h`, namespace `xpu2::interactive::rhizome`) is one machine of a swarm: a node, or a
master with `NodeOptions::master`. It does the networking, scheduling, audits and failover; the program gives it the
renderer.

```cpp
namespace rz = xpu2::interactive::rhizome;
rz::NodeOptions o;
o.rhizomeKey = "studio key";
o.master = true;
o.bind = "0.0.0.0";
o.render = [&](const uint32_t rect[4], uint32_t first, uint32_t count) { return myRenderer(rect, first, count); };
o.renderPixel = [&](uint32_t x, uint32_t y, uint32_t first, uint32_t count, float sum[4]) { myPixel(x, y, first, count, sum); };
o.statusPath = folder + "\\status.json";
rz::RhizomeNode node(o, rz::Identity::load(folder));
node.start();
node.dial("192.168.1.20:47100");                  // a node by its address (discovery is rhizome_discovery.h's)
if (node.beginFrame(shape, importance)) {}        // a frame of shape: width, height, cell size, samples
node.waitFrame(3600);                             // until every chunk's film is in
std::vector<double> picture = node.film();        // sums a pixel (RGBA), divided by counts()
node.stop();
```

**What the program gives it** (`NodeOptions`):

| Field | What it is |
|---|---|
| `rhizomeKey` | The rhizome key. |
| `master`, `masterAddress` | A master; or a node, and the master to reach (`host:port`; empty: wait to be found). |
| `render` | Renders samples [first, first + count) of a rectangle into a `Film`: the sums of each pixel's samples, their counts, and the AOVs. Every sample must be the one any machine gives it: that is what lets the master check films. |
| `renderPixel` | The master's: one pixel's sums for a sample range, for audits. |
| `onFilm` | The master's: each film as it is taken into the picture, for the program's own outputs. |
| `onManifest`, `readyAtJoin` | A node's: what the frame needs (the scene's files as pool pieces); and whether it takes work before `markReady`. |
| `mayWork` | A workstation's: false while it may not render (its machine in use). |
| `exportMerge`, `importMerge`, `onTakeover` | Failover: the program's own merged outputs as bytes for the deputy, taken in by it; what to do when this node takes over. |
| `trust`, `trustFirst`, `build` | Approvals (`rhizome_trust.h`; none: the key alone admits), endorse everyone met, and the build hash machines must share. |
| `caps`, `ramBytes`, `vramBytes` | What it has: processor threads, NUMA nodes, cards, its rate; what it gives the memory pool. |
| `bind`, `port`, `advertise`, `portFallback` | Where it listens and is reached. |
| `heartbeat`, `suspectAfter`, `goneAfter` | Seconds between heartbeats, and of silence before a member is suspect, then gone (a session uses 2 and 6). |
| `masterRenders` | Whether the master takes cells too. |
| `statusPath`, `job` | Where `status.json` goes (empty: nowhere), and the job's name in it. |
| `log` | Where its log lines go. Called under the node's lock: it must not wait (queue the line). |

**What it does** (its public methods):

| Method | What it does |
|---|---|
| `start()`, `stop()` | Listens and starts its threads; stops them and closes every connection. |
| `hash()`, `address()` | This machine's hash, and `host:port` it is reached at. |
| `dial(host:port)` | Reaches a machine; the handshake follows on a thread of its own. |
| `beginFrame(shape, importance, resume)` | Master: plans a frame with the machines ready now and gives out its cells. `importance`: a value a cell (empty: flat). `resume`: a `checkpoint()` of this frame to take up. Returns whether the checkpoint was used. |
| `waitFrame(seconds)` | Master: waits for every chunk's film (false: timed out). |
| `film()`, `counts()` | Master: the picture as sums a pixel (RGBA), and samples a pixel. |
| `checkpoint()`, `chunksBack()` | Master: the frame so far as bytes; chunks back and in all. |
| `publishManifest(bytes)`, `addPiece(piece, data)`, `announcePlacements()` | Master: what the frame needs, and the scene's files as pieces in the swarm's memory. |
| `fetch(pieces, seconds)`, `piece(hash)`, `holds(hash)` | Node: pieces fetched from the machines holding them; a piece's bytes; whether it is here. |
| `markReady(manifest)`, `setJob(text)` | Node: its scene for that manifest is made; what status says of the job. |
| `joined()`, `ready()`, `waitingForApproval()`, `isMember(hash)` | Master: the machines in the swarm, ready, and waiting for approval. |
| `isMaster()`, `masterOf()`, `succession()`, `refused()` | Whether this machine is the master now (it may have taken over), who it takes as master, the succession, and whether the master refused it. |
| `chunksByNode()`, `plan()`, `audits()`, `auditSummary()`, `quarantined(hash)` | Master: what each machine rendered, the frame's plan, the audit so far. |
| `setShaders(text)`, `shadersOf(hash)` | This machine's GPU programs state (said in heartbeats), and another's. |

The class is thread-safe: each method takes the node's lock. The callbacks are called from its own threads: `render`
from its work thread and `renderPixel` from its audit thread, neither under the lock; `onManifest`, `mayWork` and
`onTakeover` outside it too;
`log`, `onFilm`, `exportMerge` and `importMerge` under it, so they must not wait or call the node back.
