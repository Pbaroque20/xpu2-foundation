// Private XPU2 shading library, shared CPU/CUDA: MoonRay's diffusion Bssrdfs
// (shading/bssrdf NormalizedDiffusion and Dipole, their ISPC initialisation and
// C++ evaluation and sampling, which MoonRay's vector mode hands its scalar
// subsurface integrator) and that integrator's sampling helpers
// (pbr/integrator/PathIntegratorUtil.cc). Derived from MoonRay Apache-2.0,
// Copyright 2023-2024 DreamWorks Animation LLC. See MOONRAY_LICENSE.txt.
#pragma once
#include "shade_fresnel.h"
namespace xpu2 { namespace interactive { namespace shade {

enum BssrdfType : U32 { BssrdfNormalizedDiffusion = 0, BssrdfDipole = 1, BssrdfRandomWalk = 2, BssrdfNone = 0xffffffffu };
// A Bsdf's one Bssrdf: its profile's parameters, scale and transmission Fresnel.
struct BssrdfRecord {
    U32 type = BssrdfNone;
    U32 label = 0;  // its AOV label ids (shade_lobes.h encodedLabel)
    U32 lightSet = 0xffffffffu;  // its light set (the exits' lobes inherit it), ~0u none
    V3 N;
    Color scale;
    U32 fresnel;  // the multiple-transmission Fresnel of the layers above (NoIndex: none)
    float maxRadius;
    Color channelCdf;
    // normalized diffusion
    Color albedo, dmfp, d, maxR;
    // dipole
    float A;
    Color sigmaTP, sigmaTr, alphaP, zr, zv, ratio, pdfNormFactor, diffuseReflectance;
    // random walk (VolumeSubsurface): the volume's albedo and extinction, Dwivedi sampling's v0 and pdf
    // normalisation, the zero-scatter extinction, and whether exits must leave the surface
    Color rwAlbedo, sigmaT, dwivediV0, dwivediNormPdf, zeroScatterSigmaT;
    Color rwColor, rwRadius;  // VolumeSubsurface's mScatteringColor and mScatteringRadius (its material AOV properties)
    U32 resolveSelfIntersections;
    // VolumeSubsurface's mCreaseAttenuation: over 0, the walk's first direction narrowed where the surface curves
    // outward (PathIntegratorSubsurface, its zero-scatter rays out of creases).
    float creaseAttenuation = 0.0f;
};
// VolumeSubsurface_init (Chiang et al., "Practical and Controllable Subsurface Scattering for Production Path
// Tracing", with MoonRay's own albedo fit; the radius is taken in thirds).
X2_HD inline void initRandomWalk(BssrdfRecord& b, V3 N, Color trlColor, Color trlRadius, bool resolveSelfIntersections) {
    b.type = BssrdfRandomWalk;
    b.N = N;
    const Color A{fmaxf(0.001f, trlColor.x), fmaxf(0.001f, trlColor.y), fmaxf(0.001f, trlColor.z)};
    const Color d{fmaxf(0.001f, trlRadius.x * 0.3333f), fmaxf(0.001f, trlRadius.y * 0.3333f), fmaxf(0.001f, trlRadius.z * 0.3333f)};
    b.rwColor = A;
    b.rwRadius = d;
    b.resolveSelfIntersections = resolveSelfIntersections ? 1u : 0u;
    const Color e = -3.25172f * A * A * A + 1.2166f * A * A - 4.75914f * A;
    b.rwAlbedo = rgb(1) - Color{expf(e.x), expf(e.y), expf(e.z)};
    const Color s = rgb(1.9f) - A + 3.5f * (A - rgb(0.8f)) * (A - rgb(0.8f));
    b.sigmaT = Color{1.0f / (d.x * s.x), 1.0f / (d.y * s.y), 1.0f / (d.z * s.z)};
    const Color omA = rgb(1) - b.rwAlbedo;
    const Color kappa0 = Color{sqrtf(3.0f * omA.x), sqrtf(3.0f * omA.y), sqrtf(3.0f * omA.z)} *
                         (rgb(1) - omA * (rgb(2.0f / 5.0f) + omA * (rgb(12.0f / 175.0f) + omA * (rgb(2.f / 125.f) + omA * (166.f / 67375.f)))));
    b.dwivediV0 = Color{1.0f / kappa0.x, 1.0f / kappa0.y, 1.0f / kappa0.z};
    const Color ratio = (b.dwivediV0 + rgb(1)) / (b.dwivediV0 - rgb(1));
    b.dwivediNormPdf = Color{1.0f / logf(ratio.x), 1.0f / logf(ratio.y), 1.0f / logf(ratio.z)};
    const float large = 1e5f;
    b.zeroScatterSigmaT = Color{isZero(d.x) ? large : -logf(A.x - Epsilon) / d.x, isZero(d.y) ? large : -logf(A.y - Epsilon) / d.y,
                                isZero(d.z) ? large : -logf(A.z - Epsilon) / d.z};
    b.maxRadius = 0.0f;
}

//----------------------------------------------------------------------------
// Normalized diffusion ("Approximate Reflectance Profiles for Efficient Subsurface Scattering",
// Christensen & Burley 2015).
X2_HD inline float ndInverseDiffusionCDF(float y, float d) {
    const float z = (1.f - y) * 2.0f;
    const float sqrtTerm = sqrtf(1.0f + z * z);
    const float eCubeA = z + sqrtTerm;
    const float eA = cbrtf(eCubeA);
    const float x = (eA * eA - 1.0f) / eA;
    return -logf(x) * 3.f * d;
}
X2_HD inline float ndDiffusion(float r, float d) {
    if (r < .001f) r = .001f;
    const float e1 = expf(r / (-3.0f * d));
    return (e1 * e1 * e1 + e1) / ((Pi * 8.0f) * d * r);
}
// NormalizedDiffusionBssrdf_init
X2_HD inline void initNormalizedDiffusion(BssrdfRecord& b, V3 N, Color albedo, Color radius) {
    b.type = BssrdfNormalizedDiffusion;
    b.N = N;
    b.albedo = Color{fmaxf(albedo.x, 0.001f), fmaxf(albedo.y, 0.001f), fmaxf(albedo.z, 0.001f)};
    b.dmfp = Color{fmaxf(radius.x, 0.001f), fmaxf(radius.y, 0.001f), fmaxf(radius.z, 0.001f)};
    const Color tmp = b.albedo - rgb(0.33f);
    const Color tmp2 = tmp * tmp;
    const Color s = rgb(3.5f) + 100.f * tmp2 * tmp2;
    b.d = Color{b.dmfp.x / s.x, b.dmfp.y / s.y, b.dmfp.z / s.z};
    b.channelCdf = Color{fmaxf(b.albedo.x, Epsilon), fmaxf(b.albedo.y, Epsilon), fmaxf(b.albedo.z, Epsilon)};
    b.maxR = Color{ndInverseDiffusionCDF(.995f, b.d.x), ndInverseDiffusionCDF(.995f, b.d.y), ndInverseDiffusionCDF(.995f, b.d.z)};
    const float sum = b.channelCdf.x + b.channelCdf.y + b.channelCdf.z;
    if (!isZero(sum)) b.channelCdf = b.channelCdf * (1.0f / sum);
    b.channelCdf.y += b.channelCdf.x;
    b.channelCdf.z += b.channelCdf.y;
    b.maxRadius = fmaxf(fmaxf(b.maxR.x, b.maxR.y), b.maxR.z);
}

//----------------------------------------------------------------------------
// Dipole ("A Practical Model for Subsurface Light Transport", Jensen et al. 2001), from the
// artist's colour and radius ("A Rapid Hierarchical Rendering Technique...", Jensen & Buhler).
// (diffuseFresnelReflectance: shade_fresnel.h)
X2_HD inline float dipoleTotalDiffuseReflectance(float alphaP, float A) {
    const float e = sqrtf(3.0f * (1.0f - alphaP));
    return 0.5f * alphaP * (1.0f + expf(-4.0f / 3.0f * A * e)) * expf(-e);
}
X2_HD inline float dipoleFindAlphaP(float reflectance, float A) {
    float low = 0.0f, high = 1.0f;
    for (int i = 0; i < 16; i++) {
        const float mid = (low + high) * 0.5f;
        if (dipoleTotalDiffuseReflectance(mid, A) < reflectance) low = mid;
        else high = mid;
    }
    return (low + high) * 0.5f;
}
X2_HD inline float dipoleRd(float r2, float zr, float zv, float alphaP, float sigmaTr) {
    const float dr = sqrtf(r2 + zr * zr), dv = sqrtf(r2 + zv * zv);
    return (alphaP * OneOverFourPi) * ((zr * (dr * sigmaTr + 1.0f) * expf(-sigmaTr * dr)) / (dr * dr * dr) +
                                       (zv * (dv * sigmaTr + 1.0f) * expf(-sigmaTr * dv)) / (dv * dv * dv));
}
// DipoleBssrdf_init from a translucent colour and a radius (eta 1).
X2_HD inline void initDipole(BssrdfRecord& b, V3 N, Color translucentColor, Color radius) {
    b.type = BssrdfDipole;
    b.N = N;
    b.albedo = translucentColor;  // mAlbedo, its PBR validity's
    const float fdr = diffuseFresnelReflectance(1.0f);
    b.A = (1.0f + fdr) / (1.0f - fdr);
    const Color r{fmaxf(radius.x, 0.001f), fmaxf(radius.y, 0.001f), fmaxf(radius.z, 0.001f)};
    b.sigmaTr = Color{1.0f / r.x, 1.0f / r.y, 1.0f / r.z};
    const float tc[3] = {translucentColor.x, translucentColor.y, translucentColor.z};
    float alphaP[3];
    for (int c = 0; c < 3; ++c) alphaP[c] = dipoleFindAlphaP(fminf(fmaxf(tc[c], 0.0f), 1.0f), b.A);
    b.alphaP = Color{alphaP[0], alphaP[1], alphaP[2]};
    b.diffuseReflectance = Color{dipoleTotalDiffuseReflectance(alphaP[0], b.A), dipoleTotalDiffuseReflectance(alphaP[1], b.A),
                                 dipoleTotalDiffuseReflectance(alphaP[2], b.A)};
    b.sigmaTP = Color{b.sigmaTr.x / sqrtf(3.0f * (1.0f - alphaP[0])), b.sigmaTr.y / sqrtf(3.0f * (1.0f - alphaP[1])),
                      b.sigmaTr.z / sqrtf(3.0f * (1.0f - alphaP[2]))};
    // finishInit
    b.zr = Color{1.0f / b.sigmaTP.x, 1.0f / b.sigmaTP.y, 1.0f / b.sigmaTP.z};
    b.zv = b.zr * (1.0f + (4.0f / 3.0f) * b.A);
    const float maxMfp = fmaxf(fmaxf(1.0f / b.sigmaTr.x, 1.0f / b.sigmaTr.y), 1.0f / b.sigmaTr.z);
    b.maxRadius = 3.0f * maxMfp;
    float normalize = 0.0f, ratio[3], norm[3], cdf[3];
    const float st[3] = {b.sigmaTr.x, b.sigmaTr.y, b.sigmaTr.z}, zr[3] = {b.zr.x, b.zr.y, b.zr.z}, zv[3] = {b.zv.x, b.zv.y, b.zv.z};
    for (int c = 0; c < 3; c++) {
        const float expTermR = expf(-st[c] * zr[c]), expTermV = expf(-st[c] * zv[c]);
        const float sumExpTerms = expTermR + expTermV;
        const float reflectance = alphaP[c] * 0.5f * sumExpTerms;
        ratio[c] = expTermR / sumExpTerms;
        norm[c] = 1.0f / reflectance;
        cdf[c] = fmaxf(reflectance, Epsilon);
        normalize += cdf[c];
    }
    normalize = 1.0f / normalize;
    float acc = 0.0f;
    for (int c = 0; c < 3; c++) {
        cdf[c] *= normalize;
        cdf[c] += acc;
        acc = cdf[c];
    }
    b.ratio = Color{ratio[0], ratio[1], ratio[2]};
    b.pdfNormFactor = Color{norm[0], norm[1], norm[2]};
    b.channelCdf = Color{cdf[0], cdf[1], cdf[2]};
}

//----------------------------------------------------------------------------
// Evaluation and sampling (the C++ classes).
X2_HD inline float channelOf(Color c, int i) { return i == 0 ? c.x : i == 1 ? c.y : c.z; }
X2_HD inline Color bssrdfEval(const BssrdfRecord& b, float r) {
    if (b.type == BssrdfNormalizedDiffusion)
        return Color{b.albedo.x * ndDiffusion(r, b.d.x), b.albedo.y * ndDiffusion(r, b.d.y), b.albedo.z * ndDiffusion(r, b.d.z)};
    const float r2 = r * r;
    return Color{dipoleRd(r2, b.zr.x, b.zv.x, b.alphaP.x, b.sigmaTr.x), dipoleRd(r2, b.zr.y, b.zv.y, b.alphaP.y, b.sigmaTr.y),
                 dipoleRd(r2, b.zr.z, b.zv.z, b.alphaP.z, b.sigmaTr.z)};
}
X2_HD inline float bssrdfPdfLocal(const BssrdfRecord& b, float r) {
    float pdf = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const float ci = i == 0 ? channelOf(b.channelCdf, 0) : channelOf(b.channelCdf, i) - channelOf(b.channelCdf, i - 1);
        if (b.type == BssrdfNormalizedDiffusion) {
            pdf += ci * ndDiffusion(r, channelOf(b.d, i));
        } else {
            pdf += ci < Epsilon ? 0.0f
                                : ci * channelOf(b.pdfNormFactor, i) *
                                      dipoleRd(r * r, channelOf(b.zr, i), channelOf(b.zv, i), channelOf(b.alphaP, i),
                                               channelOf(b.sigmaTr, i));
        }
    }
    return pdf;
}
X2_HD inline float dipoleSolveU(float r1, float sigmaZ) {
    const float logfR1Term = logf(1.0f - r1);
    float u = 1.0f;
    int step = 0;
    while (step < 10) {
        const float f = sigmaZ * (u - 1) + logf(u) + logfR1Term;
        step = fabsf(f) < 0.001f ? 10 : step + 1;
        u -= f / (sigmaZ + 1.0f / u);
    }
    return u;
}
// sampleLocal: an offset in the tangent plane (dPi.z 0), its distance, and its pdf.
X2_HD inline float bssrdfSampleLocal(const BssrdfRecord& b, float r1, float r2, V3& dPi, float& r) {
    const int c = r2 < b.channelCdf.x ? 0 : (r2 < b.channelCdf.y ? 1 : 2);
    r2 = c == 0 ? r2 / b.channelCdf.x : (r2 - channelOf(b.channelCdf, c - 1)) / (channelOf(b.channelCdf, c) - channelOf(b.channelCdf, c - 1));
    if (b.type == BssrdfNormalizedDiffusion) {
        r1 = float(double(r1) * .99 + .005);
        r = ndInverseDiffusionCDF(r1, channelOf(b.d, c));
    } else {
        float z;
        const float ratio = channelOf(b.ratio, c);
        if (r1 < ratio) {
            z = channelOf(b.zr, c);
            r1 /= ratio;
        } else {
            z = channelOf(b.zv, c);
            r1 = (r1 - ratio) / (1.0f - ratio);
        }
        const float u = dipoleSolveU(r1, channelOf(b.sigmaTr, c) * z);
        r = z * sqrtf(u * u - 1);
    }
    const float phi = r2 * TwoPi;
    dPi = V3{r * cosf(phi), r * sinf(phi), 0.0f};
    return bssrdfPdfLocal(b, r);
}
X2_HD inline Color bssrdfDiffuseReflectance(const BssrdfRecord& b) {
    return b.type == BssrdfNormalizedDiffusion ? b.albedo : b.diffuseReflectance;
}
// Bssrdf::sampleGlobal: a direction in the spherical cap, uniformly.
X2_HD inline float bssrdfSampleGlobal(float r1, float r2, float cosThetaMax, V3& ws) {
    float cosTheta = 1.0f - r1 * (1.0f - cosThetaMax);
    cosTheta = cosTheta > 1.0f ? 1.0f : cosTheta;
    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = r2 * TwoPi;
    ws = V3{sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta};
    return 1.0f / (TwoPi * (1.0f - cosThetaMax));
}

//----------------------------------------------------------------------------
// The subsurface integrator's projection-axis helpers (PathIntegratorUtil.cc).
// sBssrdfAxisProbabilities: N 0.75, X 0.125, Y 0.125 (X and Y halved between their signs).
X2_HD inline int bssrdfSelectAxis(const Frame& f, float& rnd, V3& directionProj) {
    const float p[5] = {0.75f, 0.0625f, 0.0625f, 0.0625f, 0.0625f};
    float cdf[5];
    cdf[0] = p[0];
    for (int i = 1; i < 5; i++) cdf[i] = p[i] + cdf[i - 1];
    if (rnd < cdf[0]) {
        directionProj = -f.z;
        rnd /= p[0];
        return 0;
    }
    if (rnd < cdf[1]) {
        directionProj = -f.x;
        rnd = (rnd - cdf[0]) / p[1];
        return 1;
    }
    if (rnd < cdf[2]) {
        directionProj = f.x;
        rnd = (rnd - cdf[1]) / p[2];
        return 2;
    }
    if (rnd < cdf[3]) {
        directionProj = -f.y;
        rnd = (rnd - cdf[2]) / p[3];
        return 3;
    }
    directionProj = f.y;
    rnd = (rnd - cdf[3]) / p[4];
    return 4;
}
X2_HD inline V3 bssrdfOffsetLocalToGlobal(const Frame& f, int axis, V3 o) {
    if (axis == 0) return o.x * f.x + o.y * f.y;
    if (axis == 1 || axis == 2) return o.x * f.y + o.y * f.z;
    return o.x * f.z + o.y * f.x;
}
X2_HD inline float bssrdfMISAxisWeight(const Frame& f, V3 sampleNormal, V3 sampleOffset, const BssrdfRecord& b) {
    const V3 d{dot(sampleOffset, f.x), dot(sampleOffset, f.y), dot(sampleOffset, f.z)};
    const float rN = sqrtf(d.x * d.x + d.y * d.y), rX = sqrtf(d.y * d.y + d.z * d.z), rY = sqrtf(d.z * d.z + d.x * d.x);
    const float nN = fabsf(dot(f.z, sampleNormal)), nX = fabsf(dot(f.x, sampleNormal)), nY = fabsf(dot(f.y, sampleNormal));
    float w = 0.75f * bssrdfPdfLocal(b, rN) * nN;
    w += 0.125f * bssrdfPdfLocal(b, rX) * nX;
    w += 0.125f * bssrdfPdfLocal(b, rY) * nY;
    return !isZero(w) ? 1.0f / w : 1.0f;
}
X2_HD inline Color bssrdfAreaCompensation(Color measured, Color analytic) {
    return Color{(measured.x - 0.01f) > 0 ? analytic.x / measured.x : 1.0f, (measured.y - 0.01f) > 0 ? analytic.y / measured.y : 1.0f,
                 (measured.z - 0.01f) > 0 ? analytic.z / measured.z : 1.0f};
}
}}}
