// Private XPU2: emissive volumes as light sources (MoonRay's emissive regions: geom::internal::EmissiveRegion over a
// DenseEmissionDistribution, a 3D histogram of the volume's emission luminance laid on a box of voxels). A region is
// sampled by a voxel picked through the histogram's marginals (Distribution3D, its 1D distributions as PBRT's), a
// direction toward a place in it, and that direction's pdf by a DDA along the ray through the box (Simon's line
// integral of the 3D pdf, EGSR 2017). Shared by the lanes (the CUDA lane's regions uploaded by async_gpu.h's
// uploadRegions). Derived from MoonRay Apache-2.0, Copyright 2023-2024 DreamWorks Animation LLC (MOONRAY_LICENSE.txt).
#pragma once
#include "surface_core.h"

namespace xpu2::interactive::volume {
using surface::V3;
using sample::U32;

// A 1D step function's distribution (Distribution3D::Distribution1D): n values, its cdf (n + 1) and its integral.
struct Dist1D {
    const float* func;
    const float* cdf;
    float funcInt;
    U32 n;
};
// geom::findInterval: the last index whose cdf is at or below u, within [0, size - 2].
X2_HD inline int findInterval(const float* cdf, int size, float u) {
    int first = 0, len = size;
    while (len > 0) {
        const int half = len >> 1, middle = first + half;
        if (cdf[middle] <= u) {
            first = middle + 1;
            len -= half + 1;
        } else {
            len = half;
        }
    }
    const int i = first - 1;
    return i < 0 ? 0 : i > size - 2 ? size - 2 : i;
}
X2_HD inline int sampleDiscrete1D(const Dist1D& d, float u, float& pdf, float& uRemapped) {
    const int offset = findInterval(d.cdf, int(d.n) + 1, u);
    pdf = d.funcInt > 0.0f ? d.func[offset] / (d.funcInt * float(d.n)) : 0.0f;
    uRemapped = (u - d.cdf[offset]) / (d.cdf[offset + 1] - d.cdf[offset]);
    return offset;
}
X2_HD inline float pdfDiscrete1D(const Dist1D& d, int index) { return d.funcInt > 0.0f ? d.func[index] / (d.funcInt * float(d.n)) : 0.0f; }

// An emissive region: the volume it is, its geometry's visibility (DIFFUSE_REFLECTION ... PHASE_TRANSMISSION bits as
// MoonRay's), its box's resolution, its distribution to world transform and back (row-vector affine, 12 floats: the
// rows' x, y, z then the translation), the inverse of a voxel's volume, and its distribution: the conditional x
// distributions (nz * ny of nx), the marginal y (nz of ny) and z (one of nz), their values and cdfs in one pool.
struct Region {
    U32 volume, visibility;
    int res[3];
    float distToWorld[12], worldToDist[12];
    float invUnitVolume;
    U32 conditionalX, marginalY, marginalZ;  // offsets in the pool of each set's first distribution's record
};
// A distribution's record in the pool: funcInt, n, then n values, then n + 1 cdf entries.
X2_HD inline Dist1D dist1D(const float* pool, U32 offset) {
    Dist1D d;
    d.funcInt = pool[offset];
    d.n = U32(pool[offset + 1]);
    d.func = pool + offset + 2;
    d.cdf = pool + offset + 2 + d.n;
    return d;
}
X2_HD inline U32 dist1DFloats(U32 n) { return 2 + n + n + 1; }
struct Regions {
    const Region* regions = nullptr;
    U32 count = 0;
    const float* pool = nullptr;
};

X2_HD inline V3 affinePoint(const float* m, V3 p) {
    return V3{p.x * m[0] + p.y * m[3] + p.z * m[6] + m[9], p.x * m[1] + p.y * m[4] + p.z * m[7] + m[10],
              p.x * m[2] + p.y * m[5] + p.z * m[8] + m[11]};
}
X2_HD inline V3 affineVector(const float* m, V3 v) {
    return V3{v.x * m[0] + v.y * m[3] + v.z * m[6], v.x * m[1] + v.y * m[4] + v.z * m[7], v.x * m[2] + v.y * m[5] + v.z * m[8]};
}

// Distribution3D::sampleDiscrete and pdfDiscrete.
X2_HD inline void sampleDiscrete3D(const Region& r, const float* pool, float u1, float u2, float u3, int ijk[3], float& pdf, float rem[3]) {
    float pdfX, pdfY, pdfZ;
    const Dist1D dz = dist1D(pool, r.marginalZ);
    const int z = sampleDiscrete1D(dz, u1, pdfZ, rem[2]);
    const Dist1D dy = dist1D(pool, r.marginalY + U32(z) * dist1DFloats(U32(r.res[1])));
    const int y = sampleDiscrete1D(dy, u2, pdfY, rem[1]);
    const Dist1D dx = dist1D(pool, r.conditionalX + U32(z * r.res[1] + y) * dist1DFloats(U32(r.res[0])));
    const int x = sampleDiscrete1D(dx, u3, pdfX, rem[0]);
    ijk[0] = x;
    ijk[1] = y;
    ijk[2] = z;
    pdf = pdfX * pdfY * pdfZ;
}
X2_HD inline float pdfDiscrete3D(const Region& r, const float* pool, const int ijk[3]) {
    if (ijk[0] < 0 || ijk[0] >= r.res[0] || ijk[1] < 0 || ijk[1] >= r.res[1] || ijk[2] < 0 || ijk[2] >= r.res[2]) return 0.0f;
    const float pdfZ = pdfDiscrete1D(dist1D(pool, r.marginalZ), ijk[2]);
    const float pdfY = pdfDiscrete1D(dist1D(pool, r.marginalY + U32(ijk[2]) * dist1DFloats(U32(r.res[1]))), ijk[1]);
    const float pdfX = pdfDiscrete1D(dist1D(pool, r.conditionalX + U32(ijk[2] * r.res[1] + ijk[1]) * dist1DFloats(U32(r.res[0]))), ijk[0]);
    return pdfX * pdfY * pdfZ;
}

// EmissionDistribution::pdf: the direction's pdf from p (world), and where the ray leaves the box (world distance).
X2_HD inline float regionPdf(const Region& r, const float* pool, V3 pWorld, V3 wiWorld, float& tEnd) {
    V3 p = affinePoint(r.worldToDist, pWorld), wi = affineVector(r.worldToDist, wiWorld);
    const float lengthScale = 1.0f / sqrtf(wi.x * wi.x + wi.y * wi.y + wi.z * wi.z);
    wi = surface::mul(wi, lengthScale);
    float tStart = 0.0f;
    tEnd = 3.402823466e+38f;
    const float pa[3] = {p.x, p.y, p.z}, wa[3] = {wi.x, wi.y, wi.z};
    for (int axis = 0; axis < 3; ++axis) {
        const float invDir = 1.0f / wa[axis];
        float tnear = (0.0f - pa[axis]) * invDir, tfar = (float(r.res[axis]) - pa[axis]) * invDir;
        if (tnear > tfar) {
            const float s = tnear;
            tnear = tfar;
            tfar = s;
        }
        tStart = tnear > tStart ? tnear : tStart;
        tEnd = tfar < tEnd ? tfar : tEnd;
        if (tStart > tEnd) return 0.0f;
    }
    tEnd *= lengthScale;
    const float q[3] = {pa[0] + tStart * wa[0], pa[1] + tStart * wa[1], pa[2] + tStart * wa[2]};
    float nextT[3], deltaT[3];
    int step[3], out[3], pos[3];
    for (int axis = 0; axis < 3; ++axis) {
        int c = int(q[axis]);
        pos[axis] = c < 0 ? 0 : c > r.res[axis] - 1 ? r.res[axis] - 1 : c;
        if (wa[axis] >= 0) {
            nextT[axis] = tStart + (float(pos[axis]) + 1.0f - q[axis]) / wa[axis];
            deltaT[axis] = 1.0f / wa[axis];
            step[axis] = 1;
            out[axis] = r.res[axis];
        } else {
            nextT[axis] = tStart + (float(pos[axis]) - q[axis]) / wa[axis];
            deltaT[axis] = -1.0f / wa[axis];
            step[axis] = -1;
            out[axis] = -1;
        }
    }
    float tCurrent = tStart, accumulated = 0.0f;
    for (int guard = 0; guard < (r.res[0] + r.res[1] + r.res[2]) * 2 + 8; ++guard) {
        const float pdfD = pdfDiscrete3D(r, pool, pos);
        const int stepAxis = ((nextT[1] < nextT[0]) | (nextT[2] < nextT[0])) << (nextT[2] < nextT[1]);
        if (pdfD > 0.0f) {
            const float t0 = tCurrent * lengthScale, t1 = nextT[stepAxis] * lengthScale;
            accumulated += pdfD * r.invUnitVolume * (t1 * t1 * t1 - t0 * t0 * t0) / 3.0f;
        }
        tCurrent = nextT[stepAxis];
        pos[stepAxis] += step[stepAxis];
        if (pos[stepAxis] == out[stepAxis]) break;
        nextT[stepAxis] += deltaT[stepAxis];
    }
    return accumulated;
}
// DenseEmissionDistribution::sample (a place in the region, world) and sample from p (its direction and pdf).
X2_HD inline V3 regionPoint(const Region& r, const float* pool, float u1, float u2, float u3) {
    int ijk[3];
    float pdf, rem[3];
    sampleDiscrete3D(r, pool, u1, u2, u3, ijk, pdf, rem);
    return affinePoint(r.distToWorld, V3{float(ijk[0]) + rem[0], float(ijk[1]) + rem[1], float(ijk[2]) + rem[2]});
}
X2_HD inline V3 regionSample(const Region& r, const float* pool, V3 p, float u1, float u2, float u3, float& pdf, float& tEnd) {
    const V3 pt = regionPoint(r, pool, u1, u2, u3);
    V3 wi = surface::add(pt, surface::mul(p, -1.0f));
    const float l = sqrtf(wi.x * wi.x + wi.y * wi.y + wi.z * wi.z);
    wi = surface::mul(wi, 1.0f / l);
    pdf = regionPdf(r, pool, p, wi, tEnd);
    return wi;
}

}  // namespace xpu2::interactive::volume
