# XPU2 feature progress

Every MoonRay feature the Modo kit can produce, ported to XPU2 and checked against MoonRay (CPU XPU2 at 256 samples
a pixel against MoonRay vector mode, block error and mean ratio against MoonRay's own seed-to-seed noise), then on the
GPU in the final build. A scene with anything not listed as done still renders, by fallback to MoonRay.

## Lights

| Feature | State | Check |
|---|---|---|
| Rect, disk, sphere, spot, distant, env lights | done | earlier rounds |
| Light textures (all mappings, filters, mips, colour controls, transforms) | done | rect, disk, sphere, spot, distant, env cases within noise; circular-mapped textures 0.3% off vector mode at 256 spp, which converges to scalar mode (= XPU2) as samples grow (1.2% at 36 spp) |
| Spread (rect, disk) | done | within noise |
| Camera-visible lights, spot black level | done | |
| Cylinder light (with texture) | done | pending check |
| Mesh light (QuadMesh, TriMesh, parts) | done | quads within noise; triangles: MoonRay's in-place ear clipping reproduced (pending check) |
| Mesh light map shader | fallback | the kit never writes one |
| Clear radius, clear radius falloff | done | radius within noise; falloff 0.25% (mirror reflection of the light), under study |
| Min and max shadow distance | done | within noise |
| Two-sided sphere light | done | pending check |
| Portal light | done | pending check |
| More than 32 lights (light sets as bitsets) | done | pending check (48 lights) |
| Adaptive light sampling (light tree) | done | pending check (48 lights) |
| The scene's lights are those of the layer's light sets (Scene::updateLightList) | done | an environment light in no set is not seen (as MoonRay) |
| Light culling (Light::canIlluminate, computeActiveLights), culled lights left out of the tree's choice | done | visibility AOVs within noise |
| Light visibility flags, presence shadows | done | within noise |
| Light filters, ray termination, shadow linking, lobe light sets | done; on the CPU lane alone, except ray termination | a scene with light filters, shadow linking or lobe light sets renders on the CPU lane alone (`cpuOnly`, async_gpu.h); no check recorded here |

## Surfaces and shading

| Feature | State |
|---|---|
| MoonRay's TriMesh ear clipping, in place (its corrupted triangles) | done, pending check |
| Partial presence, presence continuation, presence shadows, cutouts | done, within noise |
| Quad meshes without UVs: MoonRay's per-quad default ST (getQuadST), st/uv primitive attributes | done, within noise |
| Geometry outside the layer is not rendered (MoonRay renders the layer's geometry) | done |
| A hit missing an attribute its shading requires: the fatal colour alone (no AOV, LPE or Cryptomatte) | done |
| Normalized diffusion and dipole subsurface | done; 0.3-0.6% bright on a DwaBase ball (SS.albedo 1.0 x albedo, MoonRay 0.997 x), open; the glass scene's dipole skin +0.55% against both MoonRay modes on two seeds, the rest of that scene exact |
| Modo switches: USD specular | done |
| Modo switches: opaque alpha, transmission roughness, toon terminator, fog alpha | done (INDEXER.md section 14) |

## Outputs

| Feature | State |
|---|---|
| Beauty, alpha | done |
| Files, parts, channel names and order (the scene's declaration order), compression, DWA level, metadata, windows | done, headers match MoonRay's |
| Depth, state variables (P, Ng, N, St, dPds, dPdt, dSdx, dSdy, dTdx, dTdy, Wp), wireframe, primitive attributes | done, within noise |
| Math filters (average, sum, min, max, closest, force consistent) | done, within noise |
| Material AOVs (albedo, color, emission, matte, roughness, normal, radius, factor, PBR validity, fresnel, state, labels, SS.*) | done, within noise (fresnel.factor: MoonRay writes uninitialized values) |
| Light AOVs: LPEs with aliases, labels (lights, materials, lobes, 'M:'), unoccluded prefix, subsurface <TD> events | done, within noise |
| Visibility AOVs (incl. culled lights and failed samples counted through every lobe, MoonRay's discarded insert) | done, within noise |
| Weight, heat map (time per pixel), beauty aux, alpha aux | done (aux and heat: values follow each renderer's own sampling) |
| Cryptomatte (categories, P0, extras, presence, multi presence, resume data; Modo count-once) | done on the CPU, within noise; on the GPU compiled into the program (build_async.py), no check recorded here |
| Display filters (all 18 of MoonRay's, run as MoonRay's own DSOs over XPU2's buffers) | done, within noise (deterministic ones exact) |
| AOV previews (view.txt) and held keys at a stop (beauty, denoiser guides) | done, to check in Modo |
| GPU AOV film (atomics, min/max, closest under a lock, heat) | written, compiled into the program (build_async.py), no check recorded here |
| Extra AOVs ('U:' from ExtraAovMap/ListMap, pre- and post-scatter, background U:bg_white) | done, within noise (7 outputs) |
| Motion vectors | done (the motionvec state AOV, aov_state.h), no check recorded here |
| Deep output, checkpoint and resume | checkpoints done (session 97: every checkpoint_interval, each file under its checkpoint_file_name, "Wrote:" lines as MoonRay's, the data file's weight, beauty_aux, alpha_aux and alpha); resume starts the render again; deep todo |

## MoonRay quirks found (vector mode)

| Quirk | XPU2 |
|---|---|
| Light-sample Russian roulette biases dim bounce light dark (0.1-4%, constant in spp) | not reproduced (tests use russian_roulette_threshold 0); XPU2 = scalar mode |
| A BSDF ray meeting two lights picks one from a non-uniform sequence: 3% dark there (mirror reflecting a light in front of the environment) | not reproduced; XPU2 = scalar mode = exact (gl1: the chrome's reflection of the key light +3.6% against vector mode, within noise against scalar) |
| Visibility misses counted once per lobe (insert's result discarded in accumVisibilityAovsOccluded) | reproduced |
| Modo's procedural textures through the object (ModoProceduralMap), by the patterns MoonRay and MoonLightIPR share | ported, world and texture spaces; not yet compared pattern by pattern |
| A camera ray seeing a light past every surface adds nothing to light AOVs (case 1 redeclares lpeStateId) | reproduced with the override off; MOONRAY_MODO_CAMERA_LIGHT_LPE keeps it |
| Subsurface light samples give unoccluded LPEs the attenuated value and count visibility at any depth | reproduced |
| Random walk rescales its light and visibility AOVs with its radiance (MOONRAY-3352) | reproduced |
| A max filter on a light AOV is unsupported (warned) | as MoonRay |

## Cameras, motion, geometry, volumes, sampling, sessions

| Feature | State |
|---|---|
| Perspective camera: film offsets, pixel aspect, stereo eyes (MoonRay's projection matrices, inverted in floats as MoonRay's) | done, within noise |
| Depth of field: disk, polygon and image bokeh, perspective and orthographic (createDOFRay, modifyForDofOrtho, LensDistribution) | done, within noise (lens samples on Sobol dimensions 14 and 15) |
| Orthographic camera (its rays' own origins as texture differentials) | done, within noise |
| Spherical camera (legacy, ranges, zoom offsets, inside out) | done, within noise |
| Fisheye camera (all four mappings, three formats; rays outside the field of view add nothing, as vector mode drops them) | done, within noise |
| DomeMaster3D camera (stereo eyes, zenith mode, flips, parallax; MoonRay never reads its interocular map) | done, within noise |
| Bake camera (position map from the mesh's UVs as MoonRay's mesh classes rasterize it, UDIM tiles, uv attribute, normal map, four modes) | done, within noise (7 cases, N, Wp and Z too); modes 0 and 2 twice MoonRay's time, open |
| Aperture and region windows (overscan), sub viewport (render region), resolution divisor, texture blur | done, within noise |
| Depth AOV: MoonRay takes the least of the camera depth and -volumeT * ray z (volumeT the largest float without a volume), so rays toward the camera's +z give huge negative depths that sum to -inf | reproduced (volume surface depth itself: with volumes) |
| Camera pixel sample map | done, weights exact |
| Camera medium material and geometry | todo (with volumes) |
| Motion blur: transforms, vertex steps, velocities and accelerations, instances, curves and points in motion, shutter bias | done on the CPU lane, within noise (pm1 all, mi1 most); curves turning 0.9981 (block 0.0101) and moving curves in screen-space radius 0.0077 open; on the GPU: done for meshes, subdivision, instances, curves, points, meshes bounding a volume and lights (OptiX motion structures, sessions 211-213: every case with no sample deferred, as close to MoonRay as the CPU lane or within noise); any number of moving lights (16 records a launch thread, worked out as a sample reaches them); moving lights read about 0.17% and moving fog about 0.08% over MoonRay on every lane, open |
| Subdivision surfaces: MoonRay's own OpenSubdiv tessellation (xpu2_geom.dll), Catmull-Clark and bilinear, creases, corners, boundaries, face-varying UVs, uniform and camera-adaptive, parts, reversed | done, within noise (13 cases; N, St, dPds exact); an open grid twice MoonRay's time, open |
| Geometry sets: only geometry in a GeometrySet is rendered (as MoonRay) | done |
| Subdivision with displacement, primitive attributes, dicing camera | displacement done (vdp1: Z exact); P and Wp behind a volume's surface fixed, retest pending; attributes in displacement done (INDEXER.md section 15); polygon maps in displacement not verified |
| VDB volumes (nearest, linear, quadratic; transforms, shells, alpha; bound maps: gain, colour, anisotropy, velocity and its rate) | done on the CPU lane, within noise (vs1, vs2); quadratic edge residual 0.0027 open; VDB in motion falls back (animated node_xform); on the GPU: deferred, matching MoonRay (vdb-sphere 0.0007, floor 0.0006; session 96, after the deferred samples' ring was made for volumes too: CUDA error 700 before) |
| Emissive and heterogeneous volumes (bound attributes), volume light path expressions | emissive done (em1); heterogeneous 5 of 8 (hv1: albedo 0.9989, anisotropy 1.0158, density on subdivision 1.0131 open); volume LPEs done (MoonRay's miss double count reproduced); on the GPU: heterogeneous deferred; a scene with emissive volumes renders on the CPU lane alone (they light every surface, and the GPU programs do not sample them yet: emit-thin 0.0017, floor 0.0017) |
| Cutout volumes | done (CutoutVolume: its indirect volume, held out on camera rays; cpu_scene.h), no check recorded here |
| Adaptive sampling: XPU2 takes 27% more samples than MoonRay at target error 10 (aov-adaptive scene) | open |
| Cellular-generation adaptive sampling as default | done (host passes; matches the error trees' images); on the GPU pass chain: todo |
| Checkpoint save and resume | saves done in the session (checked with Safe Denoising in the test kit of the time: three whole saves of 27 files on the Modo scene; the 0.5 kit greys Safe Denoising under XPU2 and asks for no saves); resume starts again (MoonRay cannot take up XPU2's saves: no progressCheckpointTileSamples; XPU2 does not yet know which samples a save holds) |

## Devices (the scene indexer, INDEXER.md)

| Feature | State |
|---|---|
| Classification of every RDL object (support, GPU families), shown in the session's log | done |
| Deferral: what the GPU programs lack (VDB, heterogeneous and emissive volumes) is a stand-in box on the GPU; a sample meeting one is rendered whole by the CPU lane from its identity | done: the GPU round (session 94-96) matches for points, motion, VDB, cutouts; the volumes defer through the same ring |
| Cells by suitability: a first-hit probe gives each cell a probability per device of being its efficient owner (families, deferral, the lanes' measured rates); the cells start by comparative advantage before any stealing; deferred samples count as the CPU's in the ownership flips | built (session 76), GPU test pending |
| One OptiX program for uniform and adaptive frames (half the compile's CPU and memory) | done (build 76) |
| The GPU program in nine modules, compiled in parallel (manual section 12) | done (build_async.py) |
| Volumes and subsurface as continuation callables | done (async_volumes.cu) |
| Modules linked by the scene's families | todo |
| Pixel filters: Gaussian, Triangle, Blackman-Harris, width to 64 | done (session 201; means and images beside MoonRay's as the B-splines') |
| Material Samples of any count (rays thinned to the room a path has) | done (session 201; compared up to 8) |
| Subsurface Samples above 4 | render at 4 x 4 a hit, noisier than MoonRay's asked count; more than 4 x 4 a hit needs the GPU programs: todo |
| Texture overflow mode, the fine-detail cache refit to a filling card | done (session 202; overflow forced by a full card and a ceiling; the refit's trigger not yet seen firing) |
| Realtime coarse first pass, a camera followed on it | done (session 199; heavy tessellation, first preparation) |
| Scene edits as change sets to both devices (partial rebuilds) | done for moves, lights and one-mesh edits (placements, reuse of unchanged meshes); a change of shape rebuilds that mesh |
| The kit's Denoiser with XPU2 previews (guides: albedo and normal; drain on Stop) | done (session 97: the kit's denoising steps on XPU2's beauty and guides, a drain mid-render denoised); not in realtime |
| Smart Render and Safe Denoising with XPU2 previews | greyed in the 0.5 kit: they are MoonRay's. The session's drains and checkpoints remain (session 97: Safe saves promoted); Continue starts again |
