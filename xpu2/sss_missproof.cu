// Conservative any-hit test for random-walk subsurface rays (xpu2/sss_gpu.cc).
// out[i] = 0 only when no triangle of the structure can be hit on the ray's
// interval, so Embree is guaranteed to miss as well; otherwise 1 (candidate).
// Each ray names the structure for its walk key (sss_gpu.cc); a null structure
// holds no face the key's filter accepts. Node: leaf when b < 0 (a = first
// triangle, -b = count); triangles are 9 floats.
struct Node { float lo[3]; int a; float hi[3]; int b; };
struct Rec { float r[8]; const Node* nodes; const float* tris; };   // {org.xyz, tnear, dir.xyz, tfar}

__device__ double dot3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Squared distance from p to triangle (a, b, c) in double precision (Ericson, RTCD 5.1.5).
__device__ double pointTriangleDist2(const double* p, const double* a, const double* b, const double* c)
{
    double ab[3], ac[3], ap[3], q[3];
    for (int i = 0; i < 3; ++i) { ab[i] = b[i]-a[i]; ac[i] = c[i]-a[i]; ap[i] = p[i]-a[i]; }
    const double d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0) { for (int i = 0; i < 3; ++i) q[i] = a[i]; }
    else {
        double bp[3]; for (int i = 0; i < 3; ++i) bp[i] = p[i]-b[i];
        const double d3 = dot3(ab, bp), d4 = dot3(ac, bp);
        double cp[3]; for (int i = 0; i < 3; ++i) cp[i] = p[i]-c[i];
        const double d5 = dot3(ab, cp), d6 = dot3(ac, cp);
        const double vc = d1*d4 - d3*d2, vb = d5*d2 - d1*d6, va = d3*d6 - d5*d4;
        if (d3 >= 0 && d4 <= d3) { for (int i = 0; i < 3; ++i) q[i] = b[i]; }
        else if (vc <= 0 && d1 >= 0 && d3 <= 0) { const double v = d1/(d1-d3); for (int i = 0; i < 3; ++i) q[i] = a[i]+v*ab[i]; }
        else if (d6 >= 0 && d5 <= d6) { for (int i = 0; i < 3; ++i) q[i] = c[i]; }
        else if (vb <= 0 && d2 >= 0 && d6 <= 0) { const double w = d2/(d2-d6); for (int i = 0; i < 3; ++i) q[i] = a[i]+w*ac[i]; }
        else if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
            const double w = (d4-d3)/((d4-d3)+(d5-d6)); for (int i = 0; i < 3; ++i) q[i] = b[i]+w*(c[i]-b[i]); }
        else { const double denom = 1.0/(va+vb+vc), v = vb*denom, w = vc*denom;
            for (int i = 0; i < 3; ++i) q[i] = a[i]+ab[i]*v+ac[i]*w; }
    }
    double dd[3]; for (int i = 0; i < 3; ++i) dd[i] = p[i]-q[i];
    return dot3(dd, dd);
}

extern "C" __global__ void sss_missproof(const Rec* rays, unsigned char* out, unsigned n, float relMargin)
{
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const float* rr = rays[id].r;
    const Node* nodes = rays[id].nodes;
    const float* tris = rays[id].tris;
    if (!nodes) { out[id] = 0; return; }
    const float org[3] = {rr[0], rr[1], rr[2]}, dir[3] = {rr[4], rr[5], rr[6]};
    const float tnear = rr[3], tfar = rr[7];
    bool finite = isfinite(tnear) && isfinite(tfar);
    for (int i = 0; i < 3; ++i) finite = finite && isfinite(org[i]) && isfinite(dir[i]);
    if (!finite) { out[id] = 1; return; }          // let Embree handle it
    float scale = 1.f;
    for (int i = 0; i < 3; ++i) scale = fmaxf(scale, fabsf(org[i]));
    const float m = relMargin * (scale + tfar);     // side margin
    const float mt = 4e-7f * (scale + tfar);        // interval margin (below the walk's 1e-5 start offset)
    const float t0 = tnear - mt, t1 = tfar + mt;
    float inv[3];
    for (int i = 0; i < 3; ++i) inv[i] = 1.f / (fabsf(dir[i]) > 1e-20f ? dir[i] : copysignf(1e-20f, dir[i]));
    int stack[64]; int sp = 0; stack[sp++] = 0;
    unsigned char found = 0;
    int visits = 0;
    while (sp && !found) {
        if (++visits > 4096) { found = 1; break; }  // bounded work: give up to Embree
        const Node nd = nodes[stack[--sp]];
        float tmin = t0, tmax = t1;
        bool cull = false;
        for (int i = 0; i < 3 && !cull; ++i) {
            float a = (nd.lo[i] - m - org[i]) * inv[i], b = (nd.hi[i] + m - org[i]) * inv[i];
            if (a > b) { const float t = a; a = b; b = t; }
            a -= 1e-5f * fabsf(a) + m; b += 1e-5f * fabsf(b) + m;
            tmin = fmaxf(tmin, a); tmax = fminf(tmax, b);
            cull = tmin > tmax;
        }
        if (cull) continue;
        if (nd.b < 0) {
            for (int k = nd.a; k < nd.a - nd.b && !found; ++k) {
                const float* tv = tris + 9 * k;
                // Float Moller-Trumbore: clear misses rejected, clear hits accepted, the rest in double.
                const float tau = 1e-4f;
                float fe1[3], fe2[3], fs[3], P[3], Q[3];
                for (int i = 0; i < 3; ++i) { fe1[i] = tv[3+i]-tv[i]; fe2[i] = tv[6+i]-tv[i]; fs[i] = org[i]-tv[i]; }
                P[0] = dir[1]*fe2[2]-dir[2]*fe2[1]; P[1] = dir[2]*fe2[0]-dir[0]*fe2[2]; P[2] = dir[0]*fe2[1]-dir[1]*fe2[0];
                const float det = fe1[0]*P[0]+fe1[1]*P[1]+fe1[2]*P[2];
                const float e1n = sqrtf(fe1[0]*fe1[0]+fe1[1]*fe1[1]+fe1[2]*fe1[2]);
                const float e2n = sqrtf(fe2[0]*fe2[0]+fe2[1]*fe2[1]+fe2[2]*fe2[2]);
                if (fabsf(det) > 1e-3f * e1n * e2n) {
                    const float invDet = 1.f / det;
                    const float u = (fs[0]*P[0]+fs[1]*P[1]+fs[2]*P[2]) * invDet;
                    Q[0] = fs[1]*fe1[2]-fs[2]*fe1[1]; Q[1] = fs[2]*fe1[0]-fs[0]*fe1[2]; Q[2] = fs[0]*fe1[1]-fs[1]*fe1[0];
                    const float v = (dir[0]*Q[0]+dir[1]*Q[1]+dir[2]*Q[2]) * invDet;
                    const float t = (fe2[0]*Q[0]+fe2[1]*Q[1]+fe2[2]*Q[2]) * invDet;
                    const float bm = tau + m / fmaxf(1e-30f, fminf(e1n, e2n));
                    const float tm = tau * fabsf(t) + mt;
                    if (u < -bm || v < -bm || u + v > 1.f + bm || t < t0 - tm || t > t1 + tm) continue;
                    if (u > bm && v > bm && u + v < 1.f - bm && t > t0 + tm && t < t1 - tm) { found = 1; break; }
                }
                double A[3] = {tv[0], tv[1], tv[2]}, B[3] = {tv[3], tv[4], tv[5]}, C[3] = {tv[6], tv[7], tv[8]};
                double e1[3], e2[3], nrm[3], o[3] = {org[0], org[1], org[2]}, d[3] = {dir[0], dir[1], dir[2]};
                for (int i = 0; i < 3; ++i) { e1[i] = B[i]-A[i]; e2[i] = C[i]-A[i]; }
                nrm[0] = e1[1]*e2[2]-e1[2]*e2[1]; nrm[1] = e1[2]*e2[0]-e1[0]*e2[2]; nrm[2] = e1[0]*e2[1]-e1[1]*e2[0];
                const double nlen = sqrt(dot3(nrm, nrm)), dlen = sqrt(dot3(d, d));
                const double denom = dot3(nrm, d);
                if (nlen == 0.0 || fabs(denom) < 1e-6 * nlen * dlen) { found = 1; break; }   // grazing
                double ao[3]; for (int i = 0; i < 3; ++i) ao[i] = A[i]-o[i];
                const double t = dot3(nrm, ao) / denom;
                const double te = t < t0 ? t0 : (t > t1 ? t1 : t);
                double pe[3]; for (int i = 0; i < 3; ++i) pe[i] = o[i] + te*d[i];
                if (pointTriangleDist2(pe, A, B, C) <= double(m)*m) found = 1;
            }
        } else {
            if (sp < 62) { stack[sp++] = nd.b; stack[sp++] = nd.a; }
            else found = 1;
        }
    }
    out[id] = found;
}

// Keeps the GPU from idling between waves (sss_gpu.cc, XPU2_SSS_GPU_KEEPALIVE_US).
// spinNs > 0 keeps one warp busy that long, enough load for the driver to raise clocks.
extern "C" __global__ void sss_keepalive(unsigned long long spinNs)
{
    if (!spinNs) return;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < spinNs);
}
