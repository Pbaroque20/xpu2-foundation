# XPU2: MoonRay's path tracer, then its shader library

The asynchronous renderer's integrator (`async_integrator.h`) began as the prototype's: a camera hit, its emission,
and one cosine-sampled ray to a constant environment. MoonRay's shader library needs MoonRay's path tracer under it:
its lights, light and BSDF sampling combined by multiple importance sampling, bounces with per-lobe depth limits and
Russian roulette. This is the plan and the reference for that port. The source is the MoonRay the reference runtime
was built from (`upstream/openmoonray/moonray`, with its Modo changes); file names below are relative to
`moonray/lib/rendering`.

## MoonRay's estimator (pbr/integrator)

XPU mode runs the bundled ISPC integrator, with the same structure as the scalar one
(`PathIntegratorMultiSampler.cc`, `PathIntegratorUtil.cc`). At every surface hit:

- **Self-emission**: `radiance += throughput × bsdf.selfEmission` at every depth.
- **Which lobes continue** (`PathIntegrator.cc`, recurse): if `ray depth < max_depth`, a lobe's indirect ray is
  allowed when its type's counter is below `max_diffuse_depth`, `max_glossy_depth` or `max_mirror_depth` (hair:
  `max_hair_depth`). A child vertex adds one to the counter of its lobe's type, one to `nonMirrorDepth` unless the lobe
  is a mirror, and one to the ray depth.
- **Sample counts**: at `nonMirrorDepth == 0` each lobe takes `bsdf_samples²` samples and each light
  `light_samples²` (the scene variables are square roots: `mLightSamples = lightSamplesSqrt²`, PathIntegrator.cc);
  deeper, one each. A first hit with several samples splits the path. MoonRay's defaults (2 and 2) are four of each.
- **BSDF samples** (`drawBsdfSamples`, `integrateBsdfSample`), for lobe i with n_i samples: draw wi, f, pdf;
  `pt = T f / (n_i pdf)`. The ray is tested against every light of the light set (not only the nearest): one hit
  light is chosen uniformly by reservoir and its radiance multiplied by the number hit (`LightSet::intersect`). Its
  direct contribution is `Li pt w`, with `w = powerHeuristic(n_i pdf, n_l pdf_light)`, or 1 for a mirror lobe; it
  counts only if a shadow ray to the light's distance is unoccluded. `tIndirect = pt` when the lobe continues.
- **Light samples** (`drawLightSamples`, `integrateLightSetSample`), per light with n_l samples: the light draws a
  point from 3 random numbers (`Light::sample`), `eval` gives Li and its solid-angle pdf; `factor = T Li / (n_l pdf)`;
  for every lobe k that matches the side of wi: `t_k = factor f_k powerHeuristic(n_l pdf, n_k pdf_k)`; the sample
  adds `Σ t_k` if a shadow ray to the light is unoccluded. MIS weights are per lobe, not over the mixture.
- **Russian roulette**, only past `nonMirrorDepth > 0` with `russian_roulette_threshold > 0`: a sample whose
  luminance (BSDF: the larger of direct and indirect) is below the threshold survives with probability
  `lum / threshold` and is scaled by its inverse.
- **Sample clamping** (`smartClamp`) past `sample_clamping_depth`, with `sample_clamping_value` divided by the
  BSDF sample count (`bsdf_samples` squared), as `PathIntegrator::initPrimaryRay` sets the subpixel's clamp.
- **Misses** add nothing: lights, the environment too, are reached through BSDF samples that hit them. Camera rays see
  lights only with `lights_visible_in_camera`.
- **Shadow rays** start at the hit offset by the ray epsilon (`modoShadowOrigin`) and end just short of the light
  (`distance × sHitEpsilonEnd`). Lights do not occlude each other.

## Lights (pbr/light)

Radiance is `color × intensity × 2^exposure`, divided by the area when `normalized` (`LightUtil.h`,
`computeLightRadiance`). Every light can `sample(P, N, r[3])` (direction, distance, light normal), `eval` (Li and
the solid-angle pdf) and `intersect(P, wi)` (for BSDF samples). Area pdfs become solid-angle pdfs by `d² / cos`
(`areaToSolidAngleScale`). Phase 1 admitted `EnvLight` (constant colour), `RectLight`, `DiskLight`, `SphereLight`,
`SpotLight` and `DistantLight`, each without image maps, light filters or clear radius; `CylinderLight`, `MeshLight`,
`PortalLight`, textured lights, clear radius and light filters have followed (`FEATURE_PROGRESS.md`, Lights). Anything
not admitted renders by fallback.

## XPU2's design

- **Shared code.** Lights (`light_core.h`) and the integrator are `X2_HD`: the CPU lane and the GPU lane run the
  same estimator, operation for operation, so the device split, the parity of identities and the audits hold.
- **Scene preparation** (`cpu_scene.h`) reads the lights of the layer's light sets into flat records in render space
  (transforms, radiance, sizes, sidedness, spread, visibility flags), gives each mesh its light set, a bitset over the
  lights, and admits `max_depth`, `max_diffuse_depth`, `max_glossy_depth`, `max_mirror_depth`, `bsdf_samples`,
  `light_samples`, `russian_roulette_threshold` and `lights_visible_in_camera`.
- **Random numbers.** The camera (dimensions 0-1) and, at the first hit, the BSDF samples (16-17) and light samples
  (18-19) are stratified across the pixel's samples (Sobol, digitally shifted); so are, at the first hit, subsurface
  scattering's draws (20-27, Sobol dimensions 6 to 13, policy bit 8: the profile samples, the forward-scattering samples,
  and the exits' light and Lambertian samples), as MoonRay draws a pixel's subsurface samples from one sequence across
  its samples. Every other dimension comes from the per-bounce Philox stream (`sample::next`) in a fixed order: per
  vertex, the BSDF samples, their light choices, the light samples (3 each), then Russian roulette. Without the
  stratified subsurface draws XPU2's dipole and normalized diffusion were 1.7 times as noisy as XPU at 64 samples.
- **Splitting.** A first hit with several samples continues each as its own branch, depth first, on a small stack.
- **Shadow rays** get a finite end (`visible(origin, direction, distance)`).

## Render space

MoonRay shades in render space, the active camera's space at its first motion step (`Scene::updateActiveCamera`);
XPU2 shades in world space. A map's colour keeps MoonRay's numbers (a render-space position or vector comes out in
render space), and a normal map's result is a world normal. Where MoonRay reads a value as a render-space quantity,
XPU2 turns it into world space: RgbToNormalMap's colour (a normal: by world-to-render's transpose), a toon specular's
input vectors (by its inverse), the render-space normals and vectors the newer maps take and give. ProjectCameraNormalMap,
on a face turned from its projector, decodes the shading normal itself as if it were a texel: its render-space
components are decoded. Without these the fixtures' cameras (turned 12 degrees) moved RgbToNormal's highlight (2%
darker) and drew a line along a ball's rim.

## Materials

The DwaBase family (`shade_dwabase.h`: DwaBaseMaterial, DwaMetal, DwaRefractive, DwaSolidDielectric, DwaFabric,
DwaVelvet_v2, DwaEmissive, DwaSkin, DwaToonMaterial, HairMaterial_v3, HairDiffuseMaterial, HairToonMaterial),
UsdPreviewSurface, and the combinators as a tree of nodes (`shade_dwalayer.h`): DwaLayer, DwaMix, DwaTwoSided,
DwaSwitch, DwaColorCorrect, DwaAdjust, and the hair ones, HairLayerMaterial (DWABASE_blendHairParameters: the emission,
hair lobes, hair diffuse and hair toon speculars blended, an input alone with its own castsCaustics; two hairs with
different Fresnel types give no lobes) and HairColorCorrectMaterial (the hair colours corrected and clamped, the
emission not above; its presence a bool, as MoonRay's). A hair material combines only through the hair combinators:
MoonRay's DwaBase combinators drop hair.

MoonRay's own SwitchMaterial and RaySwitchMaterial are records holding their materials' records
(`shade_material.h`, `SwitchRecord`). A SwitchMaterial shades its chosen material. A RaySwitchMaterial is replaced
once, as MoonRay's integrator does, by the material for the type of the lobe whose sample the ray is
(`lobeTypeToRayType`: none for a camera ray, then diffuse, mirror and glossy; a subsurface exit's ray is diffuse);
shaded itself it shades its default material, and cutting out camera rays it ends a primary path (early
termination) after its default material's presence. A material whose maps require a primitive attribute the hit
lacks shades as the fatal colour and ends the path (`hasAllRequiredAttributes`).

## Curves and hair

- **Curves** (`curve_core.h`, `curve_embree.h`): `RdlCurveGeometry` (linear, Bézier and B-spline; ray-facing or
  round) and the plugin's `ModoFurGeometry` become plan meshes whose spans take the place of triangles, numbered as
  MoonRay's `LineSegments`, `BezierSpanChains` and `BSpline` number them. Both lanes run MoonRay XPU's own curve
  programs (`OptixGPUPrograms.cu`'s `__intersection__` programs and the Embree 4 code MoonRay ported for round
  curves, vendored by `vendor_curves.py`): the GPU from a custom-primitive hit group, the CPU from an Embree user
  geometry. A hit reports only its span; each lane works the hit out again from it, as for triangles. As in XPU,
  Bézier curves are drawn flat whatever their subtype; normal-oriented curves and Bézier radii per span end are
  refused.
- **The hit** (`curveState`): `LineSegments::postIntersect` and `CubicSpline::postIntersect`: St from the "st" or
  "uv" primitive attribute, else the span's share of its curve; dPds the tangent; N faces the ray (a round B-spline:
  its geometric normal); a linear curve's dNdt is its dPdt. Primitive attributes interpolate per curve, per point
  (linear ends or the cubic basis) or per span end.
- **Rays from a curve** start at the hit and skip MoonRay's epsilon hint, 1.5 × the radius there: shadow rays from
  it (and reach at least twice it), bounces from it over the cosine to Ng, presence continuations from it.
- **Hair** (`shade_hair.h`): `HairMaterial_v3` and `HairDiffuseMaterial` through DwaBase (`DWABASE_addHairLobes`):
  the one-sampler lobe or the separate R, TT, TRT and TRRT lobes (the latter three under R's hair attenuator, as
  MoonRay places them), glints from each strand's `scatter_tag`, and hair diffuse, which also replaces the hair on a
  caustic path the material does not cast. Subsurface scattering on curves and hair under a DwaBase combinator are
  refused; hair combines through `HairLayerMaterial` and `HairColorCorrectMaterial` (see Materials).
- **Hair toon** (`HairToonMaterial`): hair diffuse at its weight and up to three toon specular sets, each MoonRay's
  surface toon specular or its hair toon specular (`HairToonSpecularBsdfLobe`: a hair R lobe's value through the
  ramp by its luminance, sampled by a second R lobe at the indirect roughness when indirect reflections are on).

## Glitter

`shade_glitter.h` is moonshine's `Glitter.ispc` (the vector path XPU runs) for DwaBase's glitter: the shading point
in object space or in reference space (`ref_P`, its st derivatives and `ref_N`, with the deformation compensation),
the Worley flake search over glitter's static noise tables (built on the host with MoonRay's own generator,
`glitterTables`), up to four macro flakes as glitter GGX lobes, the rest of the footprint's flakes as one stochastic
flakes lobe (`shade_flakes.h`, `BsdfStochasticFlakes.ispc`) blended toward a GGX lobe of their average colour, and
the debug modes. The flakes found are kept as ids in a per-thread scratch, so the lobe reads them back rather than
caching up to 2,000 normals and colours. A mesh without `ref_P` under a reference-space glitter shades as MoonRay's
fatal colour. Flake textures and glitter on both inputs of a layer (the layer's own Glitter object) are refused.

MoonRay's vector mode (and so XPU) cannot render glitter on Windows: `StochasticFlakesBsdfLobe_eval` keeps two
reflection caches of 2,000 varying entries on the stack (1.4 MB at eight lanes) and overflows a worker thread's stack
(exit 0xC00000FD). Scalar mode renders it; glitter's fixtures are measured against scalar MoonRay. The two differ
where one macro flake covers the whole footprint: the ISPC `createMacroFlakes` (which XPU runs, and which XPU2
follows) weighs that flake's lobe by its coverage weight, which can exceed one, while the C++ one keeps one. Those
flakes' glints are brighter in XPU2 than in scalar MoonRay (one glint 1.85 times in `dwa-glitter`); the means agree.

## Volumes

`volume_core.h` and the volume part of `async_integrator.h` are MoonRay's `PathIntegratorVolume.cc` for BaseVolumes
bound to closed meshes, with or without a surface material (a mesh with no material bounds a volume alone):

- A ray's crossings of the volume meshes along its whole length (`collectVolumeIntervals`: entry or exit by the
  triangle's geometric normal, the outermost entry and exit of each volume kept counting back from the far end), so
  the regions it starts in are known without tracking them along the path. The GPU keeps volume meshes in the same
  instance tree under visibility 2 (surfaces 1); the CPU keeps them in an Embree scene of their own.
- Homogeneous volumes (no attribute bound), as this section was first written: one step a stretch of the ray
  (`decoupledRayMarching`), with BaseVolume's ramps over the stretch's length; overlapping volumes summed, the thickest
  taken, or one picked by its extinction (`volume_overlap_mode`).
- Single scattering per light (`integrateVolumeScatteringDirect`): equi-angular samples about a point on bounded
  lights and distance samples, with MIS; the light sample's shadow ray goes through the volumes' transmittance. With
  `max_volume_depth` above one, Wrenninge's multiple scattering approximation with the attenuation, contribution and
  phase attenuation factors. Surface light samples and lights met by BSDF samples go through the transmittance too.
- The volume depth goes up by one at every BSDF bounce as well as after a ray's own scattering (MoonRay's
  multi-sampler sets a child's to its parent's plus one), so with the default `max_volume_depth` of one only camera
  rays gather in-scattered light; later rays only take the transmittance.
- Refused: volume shaders other than BaseVolume, VdbVolume and CutoutVolume, `volume_indirect_samples` above zero,
  more than 32 volumes. Heterogeneous and emissive volumes, VDB and cutouts, refused when this was written, have been
  ported since (`volume_grid.h`, `volume_emission.h`; state and checks in `FEATURE_PROGRESS.md`).

## Displacement

`mesh_displacement.h` is MoonRay's `PolyMesh::tessellate` and `displaceMesh` for triangle meshes, done on the host
when the scene is prepared; the renderers get an ordinary mesh:

- Uniform tessellation to `mesh_resolution` (each edge cut into that many pieces; 1 or less displaces the mesh as it
  is): `TriMesh::generateIndexBufferAndSurfaceSamples` with `MeshTessellationUtil`'s vertex lookups and ring
  stitching, in MoonRay's order, so each vertex is displaced from the same face as MoonRay's. Each quad becomes the
  two triangles Embree splits it into, with its corners' texture coordinates and primitive attributes taken from
  its base face.
- Each vertex displaced once, by the state of the first face that has it (its geometric normal, its texture
  coordinates' partials, half its corners' differences as the texture footprint, the primitive attributes at its
  place), through a NormalDisplacement or a tangent-space VectorDisplacement whose maps run on the host.
- Smooth shading normals over the displaced faces (`PolyMeshCalcNv`); the mesh's own normals are discarded. A mesh
  with `smooth_normal` and no normals of its own gets the same smoothing undisplaced.
- CombineDisplacement (its inputs' displacements scaled, added or the longer or shorter kept) and
  SwitchDisplacement (its choice, a float read as an int and cycled through 64, at preparation).
- Not taken here: adaptive tessellation (`adaptive_error` above zero), meshes with quads, n-gons or parts,
  `mesh_resolution` above 64, maps that read the polygon, and subdivision meshes. These, refused when this was written,
  now go through MoonRay's own tessellation in `xpu2_geom.dll` (`cpu_scene.h`, `polyGeometry`), which also hands a
  displacement the mesh's attributes (`INDEXER.md` sections 15 and 16). Object-space VectorDisplacement is now taken
  here (`objectSpace`).

## Maps

`shade_mapnodes.h` runs a material's maps as a program (`map_admission.h` builds it): ImageMap (with UDIM sets),
UsdUVTexture, UsdTransform2d and UsdPrimvarReader, AttributeMap, the projections (ProjectPlanarMap,
ProjectSphericalMap, ProjectCylindricalMap, ProjectTriplanarMap_v2 with its random per-face transforms,
ProjectCameraMap_v2 from a perspective or orthographic camera), UVTransformMap, NoiseMap_v2, NoiseWorleyMap_v2 and
_v3, GradientMap, RampMap, the colour corrects (ColorCorrect, Contrast, GainOffset, Gamma, Hsv, HueShift, Saturation,
TMI and Legacy), BlendMap, ClampMap, RemapMap, OpMap, the conversions, Switch maps, TwoSidedMap and LODMap; the normal
maps ImageNormalMap, ProjectPlanarNormalMap, ProjectTriplanarNormalMap_v2, ProjectCameraNormalMap, CombineNormalMap,
SwitchNormalMap and RgbToNormalMap; and the kit's ModoTextureMap and ModoNormalMap. Projections read P and N
(reference space, ref_P and ref_N, is refused); textures must be complete stored mip pyramids.

Texture lookups follow XPU's: MoonRay's vector shading reads textures through `CPP_oiioTexture` and
`CPP_oiioUdimTexture` (`BasicTexture.cc`, `UdimTexture.cc`), which pass OIIO `dsdx, dsdy, dtdx, dtdy` where it takes
`dsdx, dtdx, dsdy, dtdy`: the footprint transposed, its anisotropic filter turned. XPU2's path tracer swaps the two
the same way (`tex::Binding::swapCross`), except in displacement, which MoonRay samples through its scalar path, in
order. Without it texture edges (checker squares, triplanar stripes) differed by up to 0.2 at single pixels.

`shade_mapnodes_more.h` has the rest, as XPU runs them (their ISPC): AxisAngleMap, TransformNormalMap and
TransformSpaceMap (render, camera, world, screen through the map's camera and window, object, tangent; the instance
spaces leave a value as it is without instancing), RgbToLabMap, RandomMap and RandomNormalMap (MoonRay's static
tables, a negative index wrapped where MoonRay reads before its table), MultiChannelToFloatMap, LayerMap and LayerMap_v2,
HairMap, HairColumnMap (scatter_tag required), HairColorPresetsMap, ToonMap, CurvatureMap and WireframeMap (a mesh hit a
triangle, MoonRay's TriMesh; the normal's St partials from the vertex normals, not flipped with N as MoonRay's are not),
DirectionalMap, DeformationMap (ref_P required; the scene's fatal colour where a pair of partials is degenerate),
ProjectTriplanarUdimMap, ExtraAovMap (its colour) and ListMap (white), each with the extra AOVs it lists (`FEATURE_PROGRESS.md`), DebugMap and
DistortNormalMap (MoonRay passes it no transform: its noise reads the render-space position unless St, input
coordinates or ref_P; the hair surface coordinates are refused). Not ported: OpenVdbMap.

## Gates

Each light type and depth setting gets a fixture: the gallery geometry with the light added through an override, as
the adaptive benchmark does. Against MoonRay XPU at 1024 samples: maximum pixel 0.08, maximum 8×8 block mean 0.015,
alpha 0.015. At 64 samples, relative MSE at or below XPU's own. The existing uniform and adaptive benchmarks keep
passing.

## Phases

The plan as first written. All five have been carried out; `FEATURE_PROGRESS.md` lists what each came to.

1. Path tracer: lights, MIS, bounces, Russian roulette, splitting. Lambert only.
2. BSDF lobes and `BsdfBuilder` (shading/bsdf): Cook-Torrance/GGX with energy compensation, transmission, Oren-Nayar,
   Ward, iridescence, fabric and velvet, clearcoat layering, flakes, Fresnel; ray differentials through lobes
   (`scatterAndScale`) and roughness clamping.
3. Materials: DwaBase and its family, the combinators, UsdPreviewSurface.
4. Maps and normal maps.
5. Subsurface, hair, glitter, toon, volumes, displacement.

## Results

Build 60, the 50 fixtures of `make_material_scenes.py` at 320×240 (`run_material_benchmark.py`, tabled by
`summarize_material_benchmark.py`). MoonRay's time is XPU's (scalar mode for glitter) at 64 samples a pixel, XPU2's
the same; "Error vs MoonRay" is XPU2's 64-sample relative MSE over XPU's own (1.00 is XPU's noise); at 1024 samples
the largest pixel difference (with XPU2's own noise between two seeds) and the largest 8×8 block mean.

| Case | MoonRay (s) | XPU2 (s) | Error vs MoonRay | 1024 max pixel (XPU2 noise) | 1024 8x8 block | Gates |
|---|---:|---:|---:|---:|---:|---|
| usd-dielectric | 9.13 | 0.710 | 0.91 | 0.012 (0.010) | 0.0014 | all |
| usd-metal | 9.22 | 0.696 | 0.98 | 0.026 (0.031) | 0.0023 | all |
| usd-mirror | 9.24 | 0.753 | 1.02 | 0.019 (0.019) | 0.0019 | 64 spp error missed |
| usd-glass | 10.54 | 1.373 | 1.00 | 0.058 (0.055) | 0.0026 | all |
| usd-clearcoat | 9.36 | 0.805 | 0.93 | 0.018 (0.014) | 0.0025 | all |
| usd-mixed | 9.16 | 0.730 | 0.93 | 0.010 (0.009) | 0.0025 | all |
| dwa-base | 8.97 | 0.713 | 0.93 | 0.012 (0.012) | 0.0019 | all |
| dwa-family | 9.38 | 0.920 | 0.96 | 0.048 (0.019) | 0.0019 | all |
| dwa-fabric | 9.01 | 0.690 | 0.89 | 0.019 (0.021) | 0.0009 | all |
| dwa-iridescence | 8.78 | 0.667 | 0.93 | 0.017 (0.012) | 0.0024 | all |
| dwa-layer | 9.21 | 0.784 | 0.89 | 0.012 (0.010) | 0.0018 | all |
| dwa-adjust | 9.02 | 0.683 | 0.85 | 0.014 (0.014) | 0.0015 | all |
| map-procedural | 9.11 | 0.737 | 0.84 | 0.012 (0.020) | 0.0009 | all |
| map-modo | 9.06 | 0.745 | 0.86 | 0.051 (0.048) | 0.0014 | all |
| dwa-toon | 9.03 | 0.706 | 0.80 | 0.022 (0.017) | 0.0011 | all |
| map-noise | 9.33 | 0.781 | 0.87 | 0.012 (0.009) | 0.0011 | all |
| map-cc-legacy | 9.29 | 0.736 | 0.86 | 0.007 (0.008) | 0.0009 | all |
| dwa-sss | 9.99 | 0.932 | 1.76 | 0.055 (0.061) | 0.0142 | 64 spp error missed |
| dwa-sss-rw | 10.32 | 1.445 | 1.03 | 0.150 (0.160) | 0.0129 | 64 spp error missed |
| dwa-twosided | 9.39 | 1.024 | 0.94 | 0.009 (0.008) | 0.0014 | all |
| map-image | 4.49 | 0.178 | 1.02 | 0.390 (0.258) | 0.0036 | 64 spp error, convergence missed |
| map-image-bounce | 8.99 | 0.711 | 1.08 | 0.390 (0.257) | 0.0036 | 64 spp error, convergence missed |
| map-projection | 9.14 | 0.740 | 0.87 | 0.014 (0.013) | 0.0011 | all |
| map-triplanar | 9.29 | 0.753 | 1.18 | 0.174 (0.008) | 0.0047 | 64 spp error, convergence missed |
| map-camera | 9.06 | 0.709 | 1.26 | 0.170 (0.009) | 0.0490 | 64 spp error, convergence missed |
| map-normals | 9.08 | 0.724 | 2.21 | 0.151 (0.024) | 0.0629 | 64 spp error, convergence missed |
| map-triplanar-normal | 9.29 | 0.738 | 0.99 | 0.138 (0.035) | 0.0084 | convergence missed |
| map-uvtransform | 9.11 | 0.742 | 0.85 | 0.013 (0.014) | 0.0014 | all |
| curve-linear | 9.13 | 0.719 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-linear-round | 9.13 | 0.721 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-bezier | 9.11 | 0.742 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-bspline | 9.29 | 0.771 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-bspline-round | 9.13 | 0.804 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-hair | 9.12 | 0.719 | 0.89 | 0.010 (0.012) | 0.0015 | all |
| curve-hair-separate | 9.08 | 0.725 | 0.89 | 0.010 (0.012) | 0.0015 | all |
| curve-hair-round | 9.06 | 0.737 | 0.89 | 0.010 (0.012) | 0.0015 | all |
| curve-hair-bspline | 9.12 | 0.772 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-hair-diffuse | 9.16 | 0.722 | 0.89 | 0.009 (0.012) | 0.0015 | all |
| curve-hair-glint | 10.19 | 0.739 | 0.89 | 0.010 (0.012) | 0.0015 | all |
| curve-hair-toon | 10.36 | 1.058 | 1.16 | 0.028 (0.022) | 0.0046 | 64 spp error missed |
| curve-hair-toon-hair | 9.54 | 0.847 | 1.19 | 0.024 (0.031) | 0.0033 | 64 spp error missed |
| volume-absorption | 10.49 | 1.512 | 0.91 | 0.008 (0.007) | 0.0012 | all |
| volume-fog | 18.65 | 2.839 | 1.06 | 0.008 (0.009) | 0.0014 | 64 spp error missed |
| volume-fog-multi | 26.08 | 7.399 | 1.04 | 0.102 (0.018) | 0.0106 | 64 spp error, convergence missed |
| volume-ramp | 10.61 | 1.613 | 0.91 | 0.007 (0.008) | 0.0011 | all |
| disp-noise | 9.54 | 0.793 | 0.85 | 0.043 (0.038) | 0.0051 | all |
| disp-image | 4.75 | 0.221 | 0.55 | 0.028 (0.021) | 0.0036 | all |
| dwa-glitter (scalar MoonRay) | 11.48 | 0.926 | 4.11 | 5.228 (1.294) | 0.1397 | 64 spp error, convergence missed |
| dwa-glitter-micro (scalar MoonRay) | 12.44 | 1.382 | 1.07 | 0.388 (0.104) | 0.0086 | 64 spp error, convergence missed |
| dwa-glitter-ref (scalar MoonRay) | 12.34 | 1.180 | 1.62 | 0.695 (0.522) | 0.0384 | 64 spp error, convergence missed |

XPU2 takes 91.9% less time at the median (71.6% to 96.0%). 34 cases pass every gate. Six converge to MoonRay's
image but are noisier at 64 samples: `usd-mirror`, `dwa-sss-rw` and `volume-fog` within a few per cent, the HairToon
cases by 16–19% (cause not found), `dwa-sss` (dipole) by 76%. The rest:

- `map-image`, `map-image-bounce`: specular fireflies, MoonRay's and XPU2's alike; the blocks agree.
- `map-triplanar`: the edges of the red stripes, where the per-face random transforms meet; the blocks agree.
- `map-camera`: a thin line on the projector's silhouette in ProjectCameraNormalMap.
- `map-normals`: RgbToNormalMap's ball 2% darker where its specular lobe is.
- `map-triplanar-normal`: single edge pixels; the blocks agree.
- `volume-fog-multi`: differences the size of MoonRay's own 1024-sample noise there.
- Glitter, against scalar MoonRay (XPU cannot render it): the image means agree within 0.05%, but the glints of
  macro flakes that cover their footprint are brighter, as ISPC weighs them (see Glitter); `dwa-glitter-micro`'s
  64-sample error is within 7% of scalar MoonRay's.
