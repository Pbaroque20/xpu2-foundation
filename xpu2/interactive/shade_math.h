// Private XPU2 shading library, shared CPU/CUDA: the vector and colour arithmetic,
// frames and sampling functions MoonRay's BSDF lobes use. Derived from MoonRay
// Apache-2.0 scene_rdl2 math (Constants, Col3, Vec3, ReferenceFrame, Math) and
// shading/ispc/Util.isph. Copyright 2023-2024 DreamWorks Animation LLC. See
// MOONRAY_LICENSE.txt. Unlike the parity path (surface_core.h), these functions
// use each device's own libm: the asynchronous renderer's devices agree in
// distribution, not bit for bit.
#pragma once
#include "surface_core.h"
#if !defined(__CUDACC_RTC__)
#include <math.h>
#endif
namespace xpu2 { namespace interactive { namespace surface {
// Operators for the shared vector type, found by argument lookup everywhere.
X2_HD inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
X2_HD inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
X2_HD inline V3 operator-(V3 a) { return {-a.x, -a.y, -a.z}; }
X2_HD inline V3 operator*(V3 a, float b) { return {a.x * b, a.y * b, a.z * b}; }
X2_HD inline V3 operator*(float b, V3 a) { return {a.x * b, a.y * b, a.z * b}; }
X2_HD inline V3 operator*(V3 a, V3 b) { return {a.x * b.x, a.y * b.y, a.z * b.z}; }
X2_HD inline V3 operator/(V3 a, float b) { return {a.x / b, a.y / b, a.z / b}; }
X2_HD inline V3 operator/(V3 a, V3 b) { return {a.x / b.x, a.y / b.y, a.z / b.z}; }
X2_HD inline V3& operator+=(V3& a, V3 b) { a = a + b; return a; }
X2_HD inline V3& operator-=(V3& a, V3 b) { a = a - b; return a; }
X2_HD inline V3& operator*=(V3& a, V3 b) { a = a * b; return a; }
X2_HD inline V3& operator*=(V3& a, float b) { a = a * b; return a; }
}}}

// The shading library's large functions are called from several places in the
// path tracer; inlined at each, they multiply the GPU program (and OptiX's
// compile time) many times over. On the GPU they stay single calls.
#ifndef X2_CALL
#ifndef X2_CALL
#if defined(__CUDACC__)
#define X2_CALL __host__ __device__ __noinline__ inline
#else
#define X2_CALL inline
#endif
#endif
#endif
// A lambda kept out of line: a large block of the path loop, which OptiX and GCC then optimize apart from the rest.
#define X2_OUT_OF_LINE __attribute__((noinline))
// A local array kept in memory on the card, not in registers, where it must live across the calls into the program's
// other modules (direct callables): an empty statement that takes its address and may touch memory. No code on the host.
#if defined(__CUDA_ARCH__)
#define X2_IN_MEMORY(array) asm volatile("" : : "l"(&(array)[0]) : "memory")
#else
#define X2_IN_MEMORY(array) ((void)0)
#endif
namespace xpu2 { namespace interactive { namespace shade {
using surface::V3;
using sample::U32;
using sample::U64;
using Color = V3;  // r, g, b are x, y, z

constexpr float Epsilon = 1e-6f, OneMinusEpsilon = 1.0f - 1e-6f;
constexpr float Pi = 3.14159265358979323846f, TwoPi = 6.28318530717958647692f, FourPi = 12.5663706143591729539f;
constexpr float HalfPi = 1.57079632679489661923f, OneOverPi = 0.318309886183790671538f;
constexpr float OneOverTwoPi = 0.159154943091895335769f, OneOverFourPi = 0.0795774715459476678845f;
constexpr float SqrtTwo = 1.4142135f, SqrtTwoPi = 2.5066282f, SqrtPi = 1.77245385090551602730f;
constexpr float MaxValue = 3.402823466e38f;

X2_HD inline Color rgb(float v) { return {v, v, v}; }
X2_HD inline Color rgb(float r, float g, float b) { return {r, g, b}; }
using surface::cross;  // the shared vector type's own (argument lookup finds them anyway)
using surface::dot;
X2_HD inline float lengthSqr(V3 a) { return dot(a, a); }
X2_HD inline float length(V3 a) { return sqrtf(dot(a, a)); }
X2_HD inline V3 normalize(V3 a) { return a * (1.0f / sqrtf(dot(a, a))); }
X2_HD inline float rsqrt(float x) { return 1.0f / sqrtf(x); }
X2_HD inline float sqr(float x) { return x * x; }
X2_HD inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
X2_HD inline float saturate(float x) { return clampf(x, 0.0f, 1.0f); }
X2_HD inline float lerpf(float a, float b, float t) { return (1.0f - t) * a + t * b; }
// scene_rdl2 Transcendental dw_acos: a minimax acos within 2 ULP.
X2_HD inline float dwAcos(float x) {
    if (x >= 0.0f) {
        const float p = 1.5707963039207744f +
                        x * (-0.2145986966117427f +
                             x * (0.08897731218991513f +
                                  x * (-0.0501641f +
                                       x * (0.030862751448600963f +
                                            x * (-0.017045102007693682f + x * (0.006638618338665405f + x * -0.0012534570550072417f))))));
        return p * sqrtf(1.0f - x);
    }
    const float p = 1.5707964f -
                    x * (-0.2145986966117427f -
                         x * (0.08897731218991513f -
                              x * (-0.05016418844436275f -
                                   x * (0.030862751448600963f -
                                        x * (-0.017045102007693682f - x * (0.006638618338665405f - x * -0.0012534570550072417f))))));
    return 3.14159265f - p * sqrtf(1.0f + x);
}
// scene_rdl2 Math bias(): the classic PDI shading bias.
X2_HD inline float biasf(float val, float b) {
    if (b == 0.5f || val <= 0.0f || val >= 1.0f) return val;
    if (b <= 0.0f) return 0.0f;
    return powf(val, logf(b) * (1.0f / logf(0.5f)));
}
X2_HD inline Color lerp(Color a, Color b, float t) { return a * (1.0f - t) + b * t; }
X2_HD inline Color lerp(Color a, Color b, Color t) {
    return {lerpf(a.x, b.x, t.x), lerpf(a.y, b.y, t.y), lerpf(a.z, b.z, t.z)};
}
X2_HD inline Color clamp(Color c, float lo, float hi) { return {clampf(c.x, lo, hi), clampf(c.y, lo, hi), clampf(c.z, lo, hi)}; }
X2_HD inline Color cmax(Color a, Color b) { return {fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }
X2_HD inline Color cmin(Color a, Color b) { return {fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
X2_HD inline float maxComponent(Color c) { return fmaxf(c.x, fmaxf(c.y, c.z)); }
X2_HD inline float minComponent(Color c) { return fminf(c.x, fminf(c.y, c.z)); }
X2_HD inline Color sqrt3(Color c) { return {sqrtf(c.x), sqrtf(c.y), sqrtf(c.z)}; }
X2_HD inline Color pow3(Color c, float e) { return {powf(c.x, e), powf(c.y, e), powf(c.z, e)}; }
X2_HD inline Color exp3(Color c) { return {expf(c.x), expf(c.y), expf(c.z)}; }
X2_HD inline Color log3(Color c) { return {logf(c.x), logf(c.y), logf(c.z)}; }
// scene_rdl2 luminance and isBlack (|c| within 1e-6 of zero), isExactlyZero.
X2_HD inline float luminance(Color c) { return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z; }
X2_HD inline bool isZero(float x, float tol = Epsilon) { return fabsf(x) <= tol; }
X2_HD inline bool isOne(float x, float tol = Epsilon) { return fabsf(1.0f - x) <= tol; }
X2_HD inline bool isEqual(float x, float y, float tol = Epsilon) { return fabsf(x - y) <= fmaxf(fabsf(x), 1.0f) * tol; }
X2_HD inline bool isBlack(Color c) { return isEqual(c.x, 0) && isEqual(c.y, 0) && isEqual(c.z, 0); }
X2_HD inline bool isExactlyZero(Color c) { return c.x == 0 && c.y == 0 && c.z == 0; }
X2_HD inline bool finite3(V3 v) { return surface::finite(v); }
X2_HD inline bool isNormalized(V3 v) { const float l = lengthSqr(v); return l > 0.998001f && l < 1.002001f; }
X2_HD inline float deg2rad(float d) { return d * (Pi / 180.0f); }
X2_HD inline void sincosf2(float a, float* s, float* c) {
#if defined(__CUDACC__)
    sincosf(a, s, c);
#else
    *s = sinf(a);
    *c = cosf(a);
#endif
}
X2_HD inline float safeAcos(float x) { return acosf(clampf(x, -1.0f, 1.0f)); }

// ReferenceFrame: right-handed, Z the normal (Duff et al. from N alone, or from N
// and a tangent).
struct Frame {
    V3 x, y, z;
};
X2_HD inline Frame frame(V3 N) {
    const float sign = copysignf(1.0f, N.z), a = -1.0f / (sign + N.z), b = N.x * N.y * a;
    return {{1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x}, {b, sign + N.y * N.y * a, -N.y}, N};
}
X2_HD inline Frame frame(V3 N, V3 T) {
    Frame f;
    f.z = N;
    f.y = normalize(cross(N, T));
    f.x = cross(f.y, f.z);
    return f;
}
X2_HD inline V3 localToGlobal(const Frame& f, V3 v) { return f.x * v.x + f.y * v.y + f.z * v.z; }
X2_HD inline V3 globalToLocal(const Frame& f, V3 v) { return {dot(f.x, v), dot(f.y, v), dot(f.z, v)}; }

// shading/ispc/Util.isph
X2_HD inline float roughness2Exponent(float r) { return 2.0f / (r * r); }
X2_HD inline float exponent2Roughness(float e) { return SqrtTwo * rsqrt(e); }
X2_HD inline float roughness2Width(float roughness) {
    return (55.0f * Pi / 180.0f) * powf(roughness2Exponent(roughness), -0.8f);
}
X2_HD inline float width2Roughness(float width) {
    return SqrtTwo * powf((180.0f / (Pi * 55.0f)) * width, 1.0f / (2.0f * 0.8f));
}
X2_HD inline void computeSpectralIOR(float eta, float abbe, float& etaR, float& etaG, float& etaB) {
    etaG = eta;
    etaB = (etaG - 1.0f) / (2 * abbe) + etaG;
    etaR = 2 * etaG - etaB;
}
X2_HD inline void sampleSpectralIOR(float& r, float etaR, float etaG, float etaB, float& sampleEta, float& pdf,
                                    Color& dispersion) {
    if (r < 0.35f) {
        r /= 0.35f;
        dispersion = {1, 0, 0};
        sampleEta = etaR;
        pdf = 0.35f;
    } else if (r < 0.7f) {
        r = (r - 0.35f) / 0.35f;
        dispersion = {0, 1, 0};
        sampleEta = etaG;
        pdf = 0.35f;
    } else {
        r = (r - 0.7f) / 0.3f;
        dispersion = {0, 0, 1};
        sampleEta = etaB;
        pdf = 0.3f;
    }
}
X2_HD inline void squarifyRectangle(V3& e1, V3& e2) {
    const float l1 = length(e1), l2 = length(e2), l = sqrtf(l1 * l2);
    e1 = e1 * (l / l1);
    e2 = e2 * (l / l2);
}
X2_HD inline V3 localSpherical(float cosTheta, float sinTheta, float phi) {
    float s, c;
    sincosf2(phi, &s, &c);
    return {sinTheta * c, sinTheta * s, cosTheta};
}
X2_HD inline V3 localSpherical(float cosTheta, float sinTheta, float cosPhi, float sinPhi) {
    return {sinTheta * cosPhi, sinTheta * sinPhi, cosTheta};
}
X2_HD inline float computeReflectionDirection(V3 N, V3 wo, V3& wi) {
    const float NdotWo = dot(N, wo);
    wi = -1.0f * wo + 2.0f * NdotWo * N;
    return NdotWo;
}
X2_HD inline void reflectionDifferential(V3 wo, V3 wi, V3 H, V3 dNdx, V3 dNdy, V3& dDdx, V3& dDdy) {
    const float Cx = dot(dDdx, H) - dot(wo, dNdx), Cy = dot(dDdy, H) - dot(wo, dNdy), dDotN = -dot(wo, H);
    dDdx = dDdx - (2.0f * (dDotN * dNdx + Cx * H));
    dDdy = dDdy - (2.0f * (dDotN * dNdy + Cy * H));
}
X2_HD inline void reflectionDifferentialNoCurvature(V3 H, V3& dDdx, V3& dDdy) {
    dDdx = dDdx - 2.0f * (dot(dDdx, H) * H);
    dDdy = dDdy - 2.0f * (dot(dDdy, H) * H);
}
X2_HD inline bool computeRefractionDirection(V3 N, V3 wo, float neta, V3& wi, float& NdotWo, float& NdotWi) {
    NdotWo = dot(N, wo);
    const V3 Nf = NdotWo > 0.0f ? N : -N;
    NdotWo = fabsf(NdotWo);
    NdotWi = 1.0f - neta * neta * (1.0f - NdotWo * NdotWo);
    if (NdotWi < 0.0f) return false;
    NdotWi = sqrtf(NdotWi);
    wi = (neta * NdotWo - NdotWi) * Nf - neta * wo;
    return true;
}
X2_HD inline void refractionDifferential(float neta, V3 wo, V3 wi, V3 H, V3 dNdx, V3 dNdy, V3& dDdx, V3& dDdy) {
    float HdotWo = dot(H, wo);
    const V3 Hf = HdotWo > 0.0f ? H : -H;
    HdotWo = fabsf(HdotWo);
    const float Cx = dot(dDdx, Hf) - dot(wo, dNdx), Cy = dot(dDdy, Hf) - dot(wo, dNdy);
    const float HdotWi = fminf(dot(Hf, wi), -Epsilon);
    const float dmu = neta + neta * neta * HdotWo / HdotWi, mu = -neta * HdotWo - HdotWi;
    dDdx = neta * dDdx - (mu * dNdx + (dmu * Cx) * Hf);
    dDdy = neta * dDdy - (mu * dNdy + (dmu * Cy) * Hf);
}
X2_HD inline bool computeNormalizedHalfVector(V3 wo, V3 wi, V3& H) {
    H = wo + wi;
    const float l = length(H);
    if (l > 0.001f) {
        H = H * (1.0f / l);
        return true;
    }
    return false;
}
X2_HD inline bool computeNormalizedRefractionHalfVector(float iorWo, V3 wo, float iorWi, V3 wi, V3& H) {
    H = -(iorWo * wo + iorWi * wi);
    const float l2 = lengthSqr(H);
    if (l2 > 0.001f * 0.001f) {
        H = H * rsqrt(l2);
        return true;
    }
    return false;
}
X2_HD inline V3 sampleLocalHemisphereCosine(float r1, float r2) {
    const float cosTheta = r1 > 1.0f ? 1.0f : sqrtf(r1), sinTheta = sqrtf(1.0f - r1);
    return localSpherical(cosTheta, sinTheta, r2 * TwoPi);
}
// shading/Util.h sampleLocalSphericalEllipticCapCosine: a cosine-weighted cap whose half-widths along x and y are
// asin(sinThetaMax1) and asin(sinThetaMax2) (pdf cosTheta / (pi sinThetaMax1 sinThetaMax2)).
X2_HD inline V3 sampleLocalSphericalEllipticCapCosine(float r1, float r2, float sinThetaMax1, float sinThetaMax2) {
    float sinPhi, cosPhi;
    sincosf2(r2 * TwoPi, &sinPhi, &cosPhi);
    const float s = sqrtf(1.0f - r1);
    V3 v;
    v.x = s * sinThetaMax1 * cosPhi;
    v.y = s * sinThetaMax2 * sinPhi;
    v.z = sqrtf(1.0f - v.x * v.x - v.y * v.y);
    return v;
}
// Intersection::computePrincipalCurvatures: from the first and second fundamental forms of the surface's St partials
// (M the mean of its two estimates, as MoonRay takes it), the largest and least normal curvatures k1 and k2 and the
// direction v1 of k1.
X2_HD inline void principalCurvatures(V3 dPds, V3 dPdt, V3 dNds, V3 dNdt, float& k1, float& k2, V3& v1) {
    const float E = dot(dPds, dPds), F = dot(dPds, dPdt), G = dot(dPdt, dPdt);
    const float L = dot(dPds, dNds), M = 0.5f * (dot(dPds, dNdt) + dot(dPdt, dNds)), N = dot(dPdt, dNdt);
    const float recip = 1.0f / (E * G - F * F);
    const float K = recip * (L * N - M * M), H = recip * (0.5f * (E * N + G * L) - F * M);
    const float S = sqrtf(H * H - K);
    k1 = H + S;
    k2 = H - S;
    v1 = normalize(dPds * (N - k1 * G) - dPdt * (M - k1 * F));
}
X2_HD inline void localHemisphereCosineDifferentials(float r1, float r2, V3& dDdr1, V3& dDdr2) {
    const float c85 = cosf(deg2rad(85.0f)), minR1 = c85 * c85, maxR1 = 1.0f - minR1;
    r1 = clampf(r1, minR1, maxR1);
    const float cosTheta = r1 > 1.0f ? 1.0f : sqrtf(r1), sinTheta = sqrtf(1.0f - r1);
    float sinPhi, cosPhi;
    sincosf2(r2 * TwoPi, &sinPhi, &cosPhi);
    const float scale = 1.0f / 8.0f, halfScale = 0.5f * scale, twoPiScale = TwoPi * scale;
    dDdr1 = {-halfScale * cosPhi / sinTheta, -halfScale * sinPhi / sinTheta, halfScale / cosTheta};
    dDdr2 = {-twoPiScale * sinPhi * sinTheta, twoPiScale * cosPhi * sinTheta, 0.0f};
}
X2_HD inline V3 sampleLocalHemisphereUniform(float r1, float r2) {
    const float cosTheta = r1 > 1.0f ? 1.0f : r1, sinTheta = sqrtf(1.0f - cosTheta * cosTheta);
    return localSpherical(cosTheta, sinTheta, r2 * TwoPi);
}
X2_HD inline V3 sampleSphereUniform(float r1, float r2) {
    const float cosTheta = 1.0f - 2.0f * r1, sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    return localSpherical(cosTheta, sinTheta, r2 * TwoPi);
}
// pbr/core/Util.h: a sample is valid when its pdf is non-zero and its value not exactly zero.
X2_HD inline bool isSampleValid(Color f, float pdf) { return pdf != 0.0f && !isExactlyZero(f); }
X2_HD inline bool isSampleInvalid(Color f, float pdf) { return pdf == 0.0f || isExactlyZero(f); }
// MoonRay's erf replacement where it is used (Transcendental): libm on each device.
X2_HD inline float erfx(float x) { return erff(x); }
}}}
