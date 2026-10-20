// GPU miss proof for random-walk subsurface waves (xpu2/sss_gpu.cc).
// XPU2_SSS_GPU=1 turns it on; XPU2_SSS_GPU_MIN (default 128) is the smallest wave
// sent to the GPU; XPU2_SSS_GPU_VERIFY=1 still traces every ray with Embree and
// counts proven misses that Embree hit (must stay zero). XPU2_SSS_GPU_PTX names the
// kernel (xpu2/sss_missproof.cu compiled to PTX); XPU2_SSS_GPU_REPORT a CSV
// written at exit.
#pragma once
#include <cstdint>

namespace xpu2sssgpu {

bool enabled();
int threshold();
bool verify();

// rays: n records of 8 floats {org.xyz, tnear, dir.xyz, tfar}, with each ray's
// ext.materialID and the subsurface trace set it is traced with. On success
// candidate[i] is 0 when no face Embree's subsurface filter accepts for that ray
// can be hit on its interval (Embree would miss too) and 1 otherwise. Returns
// false when the GPU path is not available for this scene; the caller then
// traces every ray with Embree.
bool prove(const void* embreeAccelerator, const float* rays, const int* materialIDs, const void* const* traceSets,
           int n, unsigned char* candidate);

// Verify mode: rays proven to miss, and how many of those Embree hit.
void noteVerify(uint64_t checked, uint64_t embreeHits);

} // namespace xpu2sssgpu
