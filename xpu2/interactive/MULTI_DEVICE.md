# More than one processor group, more than one graphics card

XPU2's multi-device extension (`multi_device.h`). It switches on only where the machine has more than one processor
group or more than one graphics card. With one of each, XPU2 runs the code it ran before: no function in the extension
changes an answer, no follower lane exists, and the renderer's extra loop has nothing to feed.

None of it has run on such a machine. The simulation switches below drive it on one card and one processor group.

## Processor groups

Windows numbers more than 64 logical processors, and the sockets of many two-socket machines, in groups. On Windows 10
a process's threads stay in the group they started in unless placed, so XPU2 used one socket of such a machine.

- `multi::groups()` reads the active groups and their processors (`GetLogicalProcessorInformationEx`).
- The CPU lane's workers are placed over every group in proportion to its processors (`multi::placeWorker`, in
  `CpuLane`'s constructor). The feeder keeps its processor: in its own group the workers keep to every other one, as
  with one group.
- The session's thread count is every group's processors (`multi::logicalProcessors`), and so is the low-latency cap.
- The feeder's pin comes from its own group's mask, since `GetProcessAffinityMask` answers nothing once a process's
  threads sit in several groups.

Not done: memory placed by NUMA node. Scene data is read by every worker wherever it was first written.

## Graphics cards

Each card past the first is a lane of its own (`GpuLane::follow`): its own CUDA context, OptiX context, programs, copy
of the scene and film. Every lane is made and used on the one feeder thread, each call in its own context
(`GpuLane::use`).

- **Scene and edits.** `prepare`, `updateCamera`, `updatePlacements`, `dropScene` and a frame's `begin` reach the
  followers from the first lane (`forwardOrDrop`). A card that fails there is left out and the others go on.
- **Frames.** The renderer feeds each card as it feeds the first, from the same grid (`Renderer::render`, the cards
  loop): its own cellular workers, launch sizes by its own rate, its deferred samples to the CPU lane. Every chunk is
  claimed once, by one card. A pass's tables (`uploadPass`), the film downloads and Cryptomatte reach the followers
  within the frame (`forward`); a failure there fails the frame, which goes to the fallback as any GPU failure does.
- **Films.** As the first card's film lands, the followers' sums, counts, odd sums and render times are added into its
  host copy (`GpuLane::addFollowers`), and their AOVs merged by each output's filter (`Renderer::addCardAovs`). The
  previews, the picture, the sample audit and the outputs all read that copy.
- **Adaptive sampling.** With the errors, each card's film comes back (its sums, odd sums and counts); they add over the
  cards, and each pixel's error is worked out from the sums as one card works out its own (`GpuLane::poolErrors`), so
  it is the error one card with every sample would have found. The renderer then pools the CPU's as before. Errors are
  asked for only once no card holds parked samples and each has a launch slot free. (Pooling each card's own error by
  its count, the first way, was noisier: the full suite's three cards on Hydra came out past the noise.)
- **One card's alone.** The pass chain and partial shading stay with a single card: with followers, a frame takes the
  host's passes and parks samples on a missing tile, paths XPU2 already had.
- **Shaders.** Compile XPU2 Shaders compiles for each card used, since OptiX's cache holds the programs for each kind of
  card. A card whose programs are not in the cache when a scene opens stops the session with the usual message.
- `XPU2_CARDS=N` uses the first N cards; unset, all of them.

Known limits: the end-of-frame share between the GPUs and the CPU is worked out per card, so near the end the cards
together may take a little more than their share; the scene is prepared on one card after another, not at once.

## Denoising

Realtime's denoiser goes to whichever device can (`DenoiseEngine`, `xpu2_session.cc`): OptiX's denoiser on the first
card, else on each other card in turn, else Open Image Denoise on the processor (`CpuDenoiser`, loaded from MoonRay's
runtime beside XPU2's). A device that fails hands its pictures on to the next for the rest of the session, and the
Terminal says where they went. A scene the cards leave to the processor is still denoised on a card. Open Image Denoise
has no upscaling model: pictures from a quarter of their pixels are denoised at half size and enlarged.

## NUMA nodes

A machine with more than one NUMA node (two sockets, or a processor of several dies) reaches its own node's memory
quicker than another's. `multi_device.h` places the CPU lane's workers a share on each node (`places`, by node within
each chosen group) and each thread knows its node (`currentNode`). Each card has a nearest node (`gpu::cardNode`:
CUDA's host NUMA ID, else the node Windows gives the card's PCI device); its lane is made there, its scene's pinned
buffers are allocated there (`multi::OnNode` around `prepare` and the followers' calls), the tiles it misses are
decoded by threads there, and the feeder is pinned on the first card's node.

Texture tiles (`host_texture.h`) are kept in the heap of the node that decodes them. The pressure watch looks at
each node: a node short of memory (free below a tenth of its share of RAM) moves its oldest tiles to the node with
the most room, pointing their pages at the copies and freeing the old blocks once no reader holds them, as letting
go does. Read from the other node they cost a hop across the sockets, far less than decoding them from the file
again; only what no node has room for is let go. The machine-wide watch then works as before.

Not done: a copy of the coarse mip levels on each node. The CPU reads a texture through one image a binding
(`tex::Binding`), so copies would mean a lookup by node in the sampling path; worth it only if a two-socket machine
shows remote reads of the coarse levels costing time.

Simulation: `XPU2_SIMULATE_NODES=K` splits the chosen processors into K nodes, `XPU2_SIMULATE_CARD_NODES=a,b`
gives each lane its node, and `XPU2_SIMULATE_NODE_MB=a,b` holds each node's tiles to that many MB (0: no limit), so
the move to another node runs on one socket. Speed can only be measured on a machine with real nodes.

## Choosing devices

`moonray_xpu2 --list-devices` prints a line for each processor group (`group`, its number, its processors, the
processor's name) and each card (`card`, CUDA's number, its UUID, its memory in bytes, its name), simulated ones as
they render. `XPU2_CARD_LIST=0,2` and `XPU2_GROUP_LIST=0,1` use only those (a list that leaves nothing keeps
everything); `--compile-shaders` compiles for every card whatever the list says, so a card switched on later renders
at once. The kit's Preferences > MoonRay > XPU2 > Device Management (`devices.py`, `moonray_devices.py`) lists the devices,
keeps a switch for each (a card by its UUID, a group by its number) and starts XPU2 with the lists and the matching
`XPU2_SESSION_MODE`. One device at least stays on.

## Simulation

For a machine with one group and one card:

- `XPU2_SIMULATE_GROUPS=K` splits the first group's processors into K groups, so the workers are placed as on K groups.
- `XPU2_SIMULATE_CARDS=N` makes N lanes on the first card, each with its own context, scene and film, fed as N cards.
- `XPU2_SIMULATE_DENOISE_FAIL=N` makes the first N cards' OptiX denoisers fail, so realtime's pictures go to the next
  card, or to Open Image Denoise past the last.
- `XPU2_SIMULATE_CARD_THREADS=a,b,...` narrows each lane's launches (a, the first's; b, the second's...), so a lane works
  as a smaller card would beside a larger one.
- `XPU2_SIMULATE_CARD_FAIL=k` fails lane k (1 or more) as it takes the scene, so it is left out and the others go on.
- `XPU2_SIMULATE_SEED=n` renders the frame's other samples: the app's second one-card render, which measures the
  scene's noise (two renders with one seed are all but the same picture, a sample being its pixel, index and seed).

The simulation app, `xpu2/tools/simulate_devices.py`, runs a scene on these machines and judges each render against two
one-card renders, within their noise, with the Terminal lines each machine should give (`--suite quick` or `full`).

A simulated card shares the real card's memory and speed, so the timings say nothing about real cards. What it shows is
that the lanes claim, launch, merge and audit correctly: the picture should match a one-card render's within noise.

First runs (2026-11-03, RTX 3090, i7-4930K, session md1): 9 Spheres and Hydra on one card, then with 2 and 3 simulated
cards, 3 simulated processor groups, and 2 of each. None fell back. Against the one-card picture the mean ratio was
1.0000 on 9 Spheres (RMS difference 0.0001 to 0.0002 of a mean of 0.30) and 0.9998 to 0.9999 on Hydra (adaptive, RMS
0.0020 to 0.0026 of 0.038). Three lanes on one card took Hydra 2.7 times as long: the card time-slices between their
contexts, which separate cards do not. The denoiser's hand-off has not run yet.
