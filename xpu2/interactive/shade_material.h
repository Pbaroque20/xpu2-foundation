// Private XPU2 shading library, shared CPU/CUDA: the hit's shading state from a
// mesh's attributes, and the materials (MoonRay's material DSOs' shade()
// functions over the BsdfBuilder). Derived from MoonRay Apache-2.0
// (geom/prim/TriMesh.cc postIntersect, dso/material/UsdPreviewSurface),
// Copyright 2023-2024 DreamWorks Animation LLC. See MOONRAY_LICENSE.txt.
#pragma once
#include "shade_dwalayer.h"
#include "shade_mapnodes.h"
#include "curve_core.h"
namespace xpu2 { namespace interactive { namespace shade {

// A mesh's shading attributes: normals and texture coordinates, per vertex
// (indexed like the positions) or per face-vertex (three per triangle).
enum AttributeRate : U32 { RateNone = 0, RateVertex = 1, RateFaceVarying = 2 };
struct MeshAttributes {
    const V3* normals = nullptr;
    const float* uvs = nullptr;  // two per entry
    U32 normalRate = RateNone, uvRate = RateNone;
    float worldToObject[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // the mesh's node transform's inverse, 3x4 rows
    const PrimAttr* prims = nullptr;  // its primitive attributes (UserData)
    const float* primData = nullptr;
    U32 primCount = 0;
    curves::Set curve;  // a curve set in place of triangles (curve_core.h)
    // MoonRay's polyvertices of each triangle (four vertex indices, the fourth ~0u for a TriMesh's triangle) and the
    // positions they index; null where nothing reads them.
    const U32* polyVertices = nullptr;
    const V3* positions = nullptr;
    // A subdivision surface's limit derivatives at its tessellated vertices (OpenSubdivMesh's mSurfaceDpds and
    // mSurfaceDpdt), interpolated at a hit in place of the triangle's own; null for any other mesh.
    const V3* dpds = nullptr;
    const V3* dpdt = nullptr;
    // Motion blur: the vertices at the shutter's open and at its next time step (MoonRay's mVertices(id, 0) and (id, 1),
    // which its derivatives and motion read whatever the steps), the shading normals' second time sample (none: still)
    // and the ray's time. positions1 null: a still mesh.
    const V3* positions0 = nullptr;
    const V3* positions1 = nullptr;
    const V3* normals1 = nullptr;
    float time = 0.0f;
    // A moving instance (baked at time steps, which its instance transform moves it between): positions0 and
    // positions1 the steps about the ray's time, time the place between them, and the motion scaled to the shutter
    // by the steps' count less one.
    float motionScale = 1.0f;
    // A subdivision surface under motion (OpenSubdivMesh::postIntersect): its normals and limit derivatives at the
    // two time steps about the ray's time (normals, normals1, dpds, dpds1, dpdt, dpdt1) and the place between them
    // (sampleTime; below zero: the ray's time, between the first two steps, as a polygon mesh's).
    const V3* dpds1 = nullptr;
    const V3* dpdt1 = nullptr;
    float sampleTime = -1.0f;
    // A toon material's mesh under MOONRAY_MODO_TOON_TERMINATOR (not an instance): its hits record the smooth surface's
    // offset (State::modoSmoothOffset).
    bool toonOffset = false;
    // A static instance of shared geometry (MoonRay's instance hits: IntersectionInit takes the reference's normal,
    // derivatives and shading normals through the instance's transform). Its arrays are the reference's, in the
    // reference's own space, its vertices placed as they are fetched (placed). objectToWorld is the transform, four rows
    // of three (row vectors, MoonRay's: x' = x r0 + y r1 + z r2 + r3), normalToWorld its inverse's transpose (three rows,
    // n' = n.x r0 + n.y r1 + n.z r2). instance 0: not one; 1: placed; 2: placed and reflected (a negative determinant),
    // whose geometric normal MoonRay keeps as the reference's faces have it, where the placed vertices' order turns it.
    U32 instance = 0;
    float objectToWorld[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    float normalToWorld[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // The instance's own primitive attributes (its instancer's, one value an instance), looked up after the reference's.
    const PrimAttr* instancePrims = nullptr;
    const float* instancePrimData = nullptr;
    U32 instancePrimCount = 0;
};
// An instance's point, vector and normal in world space (MeshAttributes::objectToWorld, normalToWorld).
X2_HD inline V3 placedPoint(const MeshAttributes& m, V3 p) {
    const float* x = m.objectToWorld;
    return V3{p.x * x[0] + p.y * x[3] + p.z * x[6] + x[9], p.x * x[1] + p.y * x[4] + p.z * x[7] + x[10],
              p.x * x[2] + p.y * x[5] + p.z * x[8] + x[11]};
}
X2_HD inline V3 placedVector(const MeshAttributes& m, V3 v) {
    const float* x = m.objectToWorld;
    return V3{v.x * x[0] + v.y * x[3] + v.z * x[6], v.x * x[1] + v.y * x[4] + v.z * x[7], v.x * x[2] + v.y * x[5] + v.z * x[8]};
}
X2_HD inline V3 placedNormal(const MeshAttributes& m, V3 n) {
    const float* x = m.normalToWorld;
    return V3{n.x * x[0] + n.y * x[3] + n.z * x[6], n.x * x[1] + n.y * x[4] + n.z * x[7], n.x * x[2] + n.y * x[5] + n.z * x[8]};
}
// A vertex of the mesh where it is: an instance's placed, any other's as it is.
X2_HD inline V3 placed(const MeshAttributes& m, V3 p) { return m.instance ? placedPoint(m, p) : p; }
// A shading normal, an st derivative or a polyvertex of the mesh in world space (an instance's placed).
X2_HD inline V3 meshNormal(const MeshAttributes& m, const V3* normals, U32 i) {
    return m.instance ? placedNormal(m, normals[i]) : normals[i];
}
X2_HD inline V3 meshVector(const MeshAttributes& m, const V3* vectors, U32 i) {
    return m.instance ? placedVector(m, vectors[i]) : vectors[i];
}
// Intersection::setModoSmoothOffset's lift: P moved up to a corner's tangent plane, by the corner's weight.
X2_HD inline void liftToSmooth(V3& offset, V3 P, V3 corner, V3 n, float weight) {
    const V3 unit = normalize(n);
    const float below = dot(P - corner, unit);
    if (below < 0.0f) offset = offset - unit * (weight * below);
}

// computeTrianglePartialDerivatives (scene_rdl2 MathUtil.h).
X2_HD inline float differenceOfProducts(float a, float b, float c, float d) {
    const float cd = c * d;
    const float err = fmaf(-c, d, cd);
    const float dop = fmaf(a, b, -cd);
    return dop + err;
}
X2_HD inline bool trianglePartials(V3 x0, V3 x1, V3 x2, const float uv0[2], const float uv1[2], const float uv2[2], V3& dPds,
                                   V3& dPdt) {
    const float A[4] = {uv1[0] - uv0[0], uv1[1] - uv0[1], uv2[0] - uv0[0], uv2[1] - uv0[1]};
    const V3 b0 = x1 - x0, b1 = x2 - x0;
    const float det = differenceOfProducts(A[0], A[3], A[1], A[2]);
    if (!(fabsf(det) >= 1.17549435e-38f) || !(fabsf(det) <= 3.402823466e38f)) return false;  // std::isnormal
    const float inv = 1.0f / det;
    dPds = V3{differenceOfProducts(b0.x, A[3], b1.x, A[1]), differenceOfProducts(b0.y, A[3], b1.y, A[1]),
              differenceOfProducts(b0.z, A[3], b1.z, A[1])} * inv;
    dPdt = V3{differenceOfProducts(b1.x, A[0], b0.x, A[2]), differenceOfProducts(b1.y, A[0], b0.y, A[2]),
              differenceOfProducts(b1.z, A[0], b0.z, A[2])} * inv;
    return finite3(dPds) && finite3(dPdt);
}

// TriMesh::postIntersect and initIntersectionPhase1: St, Ng, N, dPds and dPdt
// at barycentrics (u, v) of triangle (a, b, c) of the given mesh, and the
// flip toward the viewer. Ng is (b - a) x (c - a), as MoonRay XPU's OptiX
// program computes it.
X2_CALL bool hitState(const MeshAttributes& m, U32 prim, U32 ia, U32 ib, U32 ic, V3 a, V3 b, V3 c, float u, float v,
                           V3 P, V3 rayDir, State& st) {
    const float w = 1.0f - u - v;
    V3 ng = cross(b - a, c - a);
    // A reflected instance: its reference's face normal through the transform (transformNormal), which is the placed
    // triangle's turned over.
    if (m.instance == 2) ng = -ng;
    const float l = length(ng);
    if (!(l > 0)) return false;
    st.Ng = ng / l;
    float st1[2] = {0, 0}, st2[2] = {1, 0}, st3[2] = {1, 1};
    if (m.uvRate != RateNone && m.uvs) {
        const U32 i1 = m.uvRate == RateVertex ? ia : prim * 3, i2 = m.uvRate == RateVertex ? ib : prim * 3 + 1,
                  i3 = m.uvRate == RateVertex ? ic : prim * 3 + 2;
        st1[0] = m.uvs[2 * i1];
        st1[1] = m.uvs[2 * i1 + 1];
        st2[0] = m.uvs[2 * i2];
        st2[1] = m.uvs[2 * i2 + 1];
        st3[0] = m.uvs[2 * i3];
        st3[1] = m.uvs[2 * i3 + 1];
    }
    st.s = w * st1[0] + u * st2[0] + v * st3[0];
    st.t = w * st1[1] + u * st2[1] + v * st3[1];
    {
        const float A0 = st2[0] - st1[0], A1 = st2[1] - st1[1], A2 = st3[0] - st1[0], A3 = st3[1] - st1[1];
        const float detA = A0 * A3 - A1 * A2;
        if (fabsf(detA) >= 1.17549435e-38f && fabsf(detA) <= 3.402823466e38f) {  // std::isnormal
            const float invDet = 1.0f / detA;
            st.stInv[0] = invDet * A3;
            st.stInv[1] = invDet * -A1;
            st.stInv[2] = invDet * -A2;
            st.stInv[3] = invDet * A0;
        } else {
            st.stInv[0] = st.stInv[3] = 1.0f;
            st.stInv[1] = st.stInv[2] = 0.0f;
        }
    }
    V3 N = st.Ng;
    st.dNds = st.dNdt = V3{0, 0, 0};
    if (m.normalRate != RateNone && m.normals) {
        const U32 i1 = m.normalRate == RateVertex ? ia : prim * 3, i2 = m.normalRate == RateVertex ? ib : prim * 3 + 1,
                  i3 = m.normalRate == RateVertex ? ic : prim * 3 + 2;
        V3 n1 = meshNormal(m, m.normals, i1), n2 = meshNormal(m, m.normals, i2), n3 = meshNormal(m, m.normals, i3);
        if (m.normals1) {
            // Two time samples (motion blur): each vertex's normal at the ray's time.
            const float nt = m.sampleTime >= 0.0f ? m.sampleTime : m.time, s0 = 1.0f - nt;
            n1 = s0 * n1 + nt * m.normals1[i1];
            n2 = s0 * n2 + nt * m.normals1[i2];
            n3 = s0 * n3 + nt * m.normals1[i3];
        }
        const V3 n = w * n1 + u * n2 + v * n3;
        const float ln = length(n);
        if (ln > 0) N = n / ln;
        if (!trianglePartials(n1, n2, n3, st1, st2, st3, st.dNds, st.dNdt)) st.dNds = st.dNdt = V3{0, 0, 0};
    }
    // MOONRAY_MODO_TOON_TERMINATOR (TriMesh, QuadMesh and OpenSubdivMesh postIntersect on a DwaToonMaterial): how far
    // the smooth surface the corner normals describe lies above the flat face at P.
    st.modoSmoothOffset = V3{0, 0, 0};
    if (m.toonOffset && m.normalRate != RateNone && m.normals) {
        const U32 j1 = m.normalRate == RateVertex ? ia : prim * 3, j2 = m.normalRate == RateVertex ? ib : prim * 3 + 1,
                  j3 = m.normalRate == RateVertex ? ic : prim * 3 + 2;
        V3 c1 = m.normals[j1], c2 = m.normals[j2], c3 = m.normals[j3];
        if (m.normals1) {
            const float nt = m.sampleTime >= 0.0f ? m.sampleTime : m.time, s0 = 1.0f - nt;
            c1 = s0 * c1 + nt * m.normals1[j1];
            c2 = s0 * c2 + nt * m.normals1[j2];
            c3 = s0 * c3 + nt * m.normals1[j3];
        }
        V3 offset{0, 0, 0};
        liftToSmooth(offset, P, a, c1, w);
        liftToSmooth(offset, P, b, c2, u);
        liftToSmooth(offset, P, c, c3, v);
        if (finite3(offset)) st.modoSmoothOffset = offset;
    }
    // The polyvertices are the mesh's first time step (mVertices(id)).
    const bool moving = m.positions0 && m.positions1;
    st.pa = moving ? m.positions0[ia] : a;
    st.pb = moving ? m.positions0[ib] : b;
    st.pc = moving ? m.positions0[ic] : c;
    st.pd = st.pc;
    st.polyCount = 3;
    if (m.polyVertices && m.positions) {
        const U32* q = m.polyVertices + 4 * size_t(prim);
        st.pa = placed(m, m.positions[q[0]]);
        st.pb = placed(m, m.positions[q[1]]);
        st.pc = placed(m, m.positions[q[2]]);
        if (q[3] != ~0u) {
            st.pd = placed(m, m.positions[q[3]]);
            st.polyCount = 4;
        }
    }
    st.motion = V3{0, 0, 0};
    st.subdivision = m.dpds && m.dpdt;
    if (m.dpds && m.dpdt) {
        // OpenSubdivMesh::postIntersect: the limit surface's derivatives at the triangle's vertices.
        st.dPds = w * meshVector(m, m.dpds, ia) + u * meshVector(m, m.dpds, ib) + v * meshVector(m, m.dpds, ic);
        st.dPdt = w * meshVector(m, m.dpdt, ia) + u * meshVector(m, m.dpdt, ib) + v * meshVector(m, m.dpdt, ic);
        if (m.dpds1 && m.dpdt1) {
            // Under motion: each vertex's derivatives lerped between the time steps about the ray's time.
            const float nt = m.sampleTime >= 0.0f ? m.sampleTime : m.time, s0 = 1.0f - nt;
            st.dPds = w * (s0 * m.dpds[ia] + nt * m.dpds1[ia]) + u * (s0 * m.dpds[ib] + nt * m.dpds1[ib]) +
                      v * (s0 * m.dpds[ic] + nt * m.dpds1[ic]);
            st.dPdt = w * (s0 * m.dpdt[ia] + nt * m.dpdt1[ia]) + u * (s0 * m.dpdt[ib] + nt * m.dpdt1[ib]) +
                      v * (s0 * m.dpdt[ic] + nt * m.dpdt1[ic]);
        }
    } else if (moving) {
        // TriMesh::postIntersect under motion blur: the derivatives at the first two time steps, lerped to the time.
        V3 d0[2], d1[2];
        const V3 a0 = m.positions0[ia], b0 = m.positions0[ib], c0 = m.positions0[ic];
        const V3 a1 = m.positions1[ia], b1 = m.positions1[ib], c1 = m.positions1[ic];
        if (!trianglePartials(a0, b0, c0, st1, st2, st3, d0[0], d0[1])) {
            const Frame f = frame(N);
            d0[0] = f.x;
            d0[1] = f.y;
        }
        if (!trianglePartials(a1, b1, c1, st1, st2, st3, d1[0], d1[1])) {
            const Frame f = frame(N);
            d1[0] = f.x;
            d1[1] = f.y;
        }
        st.dPds = (1.0f - m.time) * d0[0] + m.time * d1[0];
        st.dPdt = (1.0f - m.time) * d0[1] + m.time * d1[1];
    } else if (!trianglePartials(a, b, c, st1, st2, st3, st.dPds, st.dPdt)) {
        const Frame f = frame(N);
        st.dPds = f.x;
        st.dPdt = f.y;
    }
    if (moving) {
        // Mesh::computeMotion: the hit's place at the time less and plus sHalfDt (0.01), between its first two time
        // steps, as a velocity a shutter interval.
        const V3 open = w * m.positions0[ia] + u * m.positions0[ib] + v * m.positions0[ic];
        const V3 close = w * m.positions1[ia] + u * m.positions1[ib] + v * m.positions1[ic];
        const float t0 = m.time - 0.01f, t1 = m.time + 0.01f;
        const V3 p0 = (1.0f - t0) * open + t0 * close, p1 = (1.0f - t1) * open + t1 * close;
        st.motion = (p1 - p0) / (2.0f * 0.01f) * m.motionScale;
    }
    st.P = P;
    st.wo = -rayDir;
    st.entering = dot(st.Ng, rayDir) < 0.0f;
    if (!st.entering) {
        N = -N;
        st.Ng = -st.Ng;
    }
    st.N = N;
    st.prim = prim;
    st.u = u;
    st.v = v;
    st.ia = ia;
    st.ib = ib;
    st.ic = ic;
    st.prims = m.prims;
    st.primData = m.primData;
    st.primCount = m.primCount;
    st.instancePrims = m.instancePrims;
    st.instancePrimData = m.instancePrimData;
    st.instancePrimCount = m.instancePrimCount;
    return finite3(st.N) && finite3(st.Ng);
}

//----------------------------------------------------------------------------
// Materials. A material is a record and its parameters in a pool of floats.
enum MaterialKind : U32 { MatUsdPreviewSurface = 1, MatDwaBase = 2, MatSwitch = 3 };
// The kit's MoonRay source switches each material record carries (cpu_scene.h, from the environment):
// MOONRAY_MODO_TOON_TERMINATOR, MOONRAY_MODO_TRANSMISSION_ROUGHNESS, and a DwaToonMaterial under the first (its
// smooth shadow origins).
enum MaterialFlags : U32 {
    MatModoUsdSpecular = 1,
    MatPolyVertices = 2,
    MatModoToonTerminator = 4,
    MatModoTransmissionRoughness = 8,
    MatModoToonMaterial = 16
};  // MatPolyVertices: its maps read the hit polygon
// maps: the material's map program (shade_maps.h), NoMapProgram for none; bindings: its own bound
// attributes (UsdPreviewSurface; a DwaBase tree's are on its nodes).
struct MaterialRecord {
    U32 kind = 0, flags = 0, offset = 0, count = 0;
    // labels: its label table's pool offset (Builder::labels), NoLabels when no AOV asks for labels.
    U32 maps = NoMapProgram, bindings = 0, bindingCount = 0, labels = 0xffffffffu;
    // Its extra AOVs' pool offset (none: NoExtraAovs): their map program, their count, then each one's label, whether
    // post scatter, the register of the map its colour is bound to (MapNoRegister: none) and its colour.
    U32 extras = 0xffffffffu;
};
constexpr U32 NoExtraAovs = 0xffffffffu, ExtraAovFloats = 6;

// UsdPreviewSurface's parameters, in its pool order.
struct UsdPreviewSurfaceParams {
    Color diffuseColor, emissiveColor, specularColor;
    float useSpecularWorkflow, metallic, roughness, clearcoat, clearcoatRoughness, opacity, opacityThreshold, ior;
    V3 normal;
};
constexpr U32 UsdPreviewSurfaceFloats = sizeof(UsdPreviewSurfaceParams) / sizeof(float);
// UsdPreviewSurface.json's labels, as the build numbers them (labels.h).
enum UsdLabels : U32 { AovDiffuse = 1, AovSpecular, AovSpecularTransmission, AovOuterSpecular };

X2_HD inline void usdMirrorDielectric(Builder& b, V3 N, float ior, float weight, float opacity, U32 reflLabel, U32 transLabel,
                                      U32 behavior) {
    MirrorBSDF m{N, ior, rgb(1), 0.0f, ior, weight, 1.0f - opacity, nullptr};
    addMirrorBSDF(b, m, 1.0f, behavior, reflLabel, transLabel);
}
X2_HD inline void usdMicrofacetDielectric(Builder& b, V3 N, float ior, float roughness, float weight, float opacity,
                                          U32 reflLabel, U32 transLabel, U32 behavior) {
    MicrofacetIsotropicBSDF m{N, ior, roughness, roughness, false, rgb(1), 0.0f, ior, weight, 1.0f - opacity, GGXDistribution,
                              nullptr};
    addMicrofacetIsotropicBSDF(b, m, 1.0f, behavior, reflLabel, transLabel);
}
// UsdPreviewSurface.ispc shade().
X2_CALL void shadeUsdPreviewSurface(const UsdPreviewSurfaceParams& p, U32 flags, const State& st, Builder& bb) {
    V3 N;
    if (isZero(length(st.dPds))) N = st.N;
    else N = normalize(localToGlobal(frame(st.N, normalize(st.dPds)), p.normal));
    const Color albedo = p.diffuseColor;
    const float ior = p.ior, roughness = saturate(p.roughness), opacityThreshold = saturate(p.opacityThreshold);
    const float opacity = opacityThreshold > 0.0f ? 1.0f : saturate(p.opacity);
    const Color emissive = p.emissiveColor;
    if (!isBlack(emissive)) addEmission(bb, emissive);
    if (!isZero(p.clearcoat)) {
        if (isZero(p.clearcoatRoughness))
            usdMirrorDielectric(bb, N, ior, p.clearcoat, 1.0f, AovOuterSpecular, AovSpecularTransmission, Additive);
        else
            usdMicrofacetDielectric(bb, N, ior, p.clearcoatRoughness, p.clearcoat, 1.0f, AovOuterSpecular, AovSpecularTransmission,
                                    Additive);
    }
    const float weight = 1.0f;
    if (p.useSpecularWorkflow != 0.0f) {
        if (flags & MatModoUsdSpecular) {
            const Color spec = {saturate(p.specularColor.x), saturate(p.specularColor.y), saturate(p.specularColor.z)};
            if (!isBlack(spec)) addSchlickReflection(bb, N, spec, roughness, isZero(roughness), AovSpecular);
        }
    } else {
        const float metallic = p.metallic;
        if (isZero(metallic)) {
            if (isZero(roughness)) usdMirrorDielectric(bb, N, ior, weight, opacity, AovSpecular, AovSpecularTransmission, Physical);
            else usdMicrofacetDielectric(bb, N, ior, roughness, weight, opacity, AovSpecular, AovSpecularTransmission, Physical);
        } else if (isOne(metallic)) {
            if (isZero(roughness)) addMirrorBRDF(bb, mirrorConductor(N, albedo, albedo), weight, Physical, AovSpecular);
            else addMicrofacetIsotropicBRDF(bb, microfacetConductor(N, albedo, albedo, roughness, GGXDistribution), weight, Physical,
                                            AovSpecular);
            return;
        } else {
            if (isZero(roughness)) {
                addMirrorBRDF(bb, mirrorConductor(N, albedo, albedo), metallic, Physical, AovSpecular);
                usdMirrorDielectric(bb, N, ior, weight, opacity, AovSpecular, AovSpecularTransmission, Physical);
            } else {
                addMicrofacetIsotropicBRDF(bb, microfacetConductor(N, albedo, albedo, roughness, GGXDistribution), metallic, Physical,
                                           AovSpecular);
                usdMicrofacetDielectric(bb, N, ior, roughness, weight, opacity, AovSpecular, AovSpecularTransmission, Physical);
            }
        }
    }
    if (!isBlack(albedo)) addLambertianBRDF(bb, LambertianBRDF{N, albedo, nullptr}, 1.0f, Physical, AovDiffuse);
}
// UsdPreviewSurface::presence: 1 where the opacity reaches the threshold, else 0.
X2_HD inline float presenceUsdPreviewSurface(const UsdPreviewSurfaceParams& p) {
    return p.opacity >= p.opacityThreshold ? 1.0f : 0.0f;
}

// The texture coordinates' derivatives at the hit (State dsdx...), from the ray's differentials
// (camera rays; the path tracer's other rays carry none, so their lookups use the finest level):
// each differential ray meets the triangle's plane, and its barycentric offset maps through the
// triangle's texture coordinates. The differentials are scaled for the samples per pixel as the
// validated ImageMap subset scales them (tex::footprint).
X2_HD inline void hitTexDerivatives(const MeshAttributes& m, U32 prim, U32 ia, U32 ib, U32 ic, V3 a, V3 b, V3 c,
                                    const surface::Ray& ray, U32 spp, State& st) {
    st.dsdx = st.dsdy = st.dtdx = st.dtdy = 0.0f;
    if (isExactlyZero(ray.dx) || isExactlyZero(ray.dy) || !spp) return;
    float st1[2] = {0, 0}, st2[2] = {1, 0}, st3[2] = {1, 1};
    if (m.uvRate != RateNone && m.uvs) {
        const U32 i1 = m.uvRate == RateVertex ? ia : prim * 3, i2 = m.uvRate == RateVertex ? ib : prim * 3 + 1,
                  i3 = m.uvRate == RateVertex ? ic : prim * 3 + 2;
        st1[0] = m.uvs[2 * i1];
        st1[1] = m.uvs[2 * i1 + 1];
        st2[0] = m.uvs[2 * i2];
        st2[1] = m.uvs[2 * i2 + 1];
        st3[0] = m.uvs[2 * i3];
        st3[1] = m.uvs[2 * i3 + 1];
    }
    const V3 e1 = b - a, e2 = c - a, n = cross(e1, e2);
    U32 axis = fabsf(n.x) > fabsf(n.y) ? 0 : 1;
    if (fabsf(n.z) > (axis == 0 ? fabsf(n.x) : fabsf(n.y))) axis = 2;
    auto pa = [&](V3 v) { return axis == 0 ? v.y : axis == 1 ? v.z : v.x; };
    auto pb = [&](V3 v) { return axis == 0 ? v.z : axis == 1 ? v.x : v.y; };
    const float det = pa(e1) * pb(e2) - pb(e1) * pa(e2), den = dot(n, ray.direction);
    if (!(fabsf(det) > 0) || !(fabsf(den) > 0)) return;
    const float plane = dot(n, a - ray.origin);
    const V3 primary = ray.direction * (plane / den);
    const float scale = 1.f / fminf(sqrtf(float(spp)), 2.f);
    const V3 directions[2] = {ray.dx, ray.dy}, offsets[2] = {ray.ox, ray.oy};
    float ds[2], dt[2];
    for (int i = 0; i < 2; ++i) {
        V3 d = directions[i];
        if (scale < 1) d = normalize(ray.direction + (d - ray.direction) * scale);
        const float dd = dot(n, d);
        if (!(fabsf(dd) > 0)) return;
        const V3 o = offsets[i] * scale;
        const V3 delta = o + d * (dot(n, a - ray.origin - o) / dd) - primary;
        const float x = (pa(delta) * pb(e2) - pb(delta) * pa(e2)) / det, y = (pa(e1) * pb(delta) - pb(e1) * pa(delta)) / det;
        ds[i] = (st2[0] - st1[0]) * x + (st3[0] - st1[0]) * y;
        dt[i] = (st2[1] - st1[1]) * x + (st3[1] - st1[1]) * y;
    }
    if (!(isfinite(ds[0]) && isfinite(ds[1]) && isfinite(dt[0]) && isfinite(dt[1]))) return;
    st.dsdx = ds[0];
    st.dtdx = dt[0];
    st.dsdy = ds[1];
    st.dtdy = dt[1];
}

// Ray::transfer: the incoming ray's auxiliary rays meet the hit's plane; dPdx and dPdy their offsets from P.
X2_HD inline void transferDifferentials(const RayDiff& d, V3 origin, V3 dir, State& st) {
    st.dPdx = st.dPdy = V3{0, 0, 0};
    st.diffValid = false;
    if (!d.valid) return;
    const V3 oX = origin + d.dOx, oY = origin + d.dOy, dX = dir + d.dDx, dY = dir + d.dDy;
    const float denomX = dot(st.Ng, dX), denomY = dot(st.Ng, dY);
    if (denomX == 0.0f || denomY == 0.0f) return;
    const float planeD = -dot(st.Ng, st.P);
    const float distX = -(dot(oX, st.Ng) + planeD) / denomX, distY = -(dot(oY, st.Ng) + planeD) / denomY;
    st.dPdx = (oX + dX * distX) - st.P;
    st.dPdy = (oY + dY * distY) - st.P;
    st.diffValid = finite3(st.dPdx) && finite3(st.dPdy);
    if (!st.diffValid) st.dPdx = st.dPdy = V3{0, 0, 0};
}
// Intersection::transferAndComputeDerivatives: the texture derivatives from dPdx and dPdy over dPds and dPdt;
// without differentials, the coarsest level (1024).
X2_HD inline void differentialTexDerivatives(State& st) {
    st.dsdx = st.dtdx = st.dsdy = st.dtdy = 0.0f;
    if (!st.diffValid) {
        st.dsdx = 1024.0f;
        st.dtdy = 1024.0f;
        return;
    }
    const float e = 1e-5f;  // sIntersectionEpsilon
    if (isZero(dot(st.dPds, st.dPdt), e)) {
        const float du = dot(st.dPds, st.dPds), dv = dot(st.dPdt, st.dPdt);
        if (du > e && dv > e) {
            const float iu = 1.0f / dot(st.dPds, st.dPds), iv = 1.0f / dot(st.dPdt, st.dPdt);
            st.dsdx = dot(st.dPdx, st.dPds) * iu;
            st.dtdx = dot(st.dPdx, st.dPdt) * iv;
            st.dsdy = dot(st.dPdy, st.dPds) * iu;
            st.dtdy = dot(st.dPdy, st.dPdt) * iv;
        }
        return;
    }
    // Drop the normal's largest axis and solve the 2x2 system (pbrt 2nd ed. p. 508).
    const V3 n2 = st.Ng * st.Ng;
    float A[4], bx[2], by[2];
    if (n2.x > n2.y && n2.x > n2.z) {
        A[0] = st.dPds.y; A[1] = st.dPdt.y; A[2] = st.dPds.z; A[3] = st.dPdt.z;
        bx[0] = st.dPdx.y; bx[1] = st.dPdx.z; by[0] = st.dPdy.y; by[1] = st.dPdy.z;
    } else if (n2.y > n2.z) {
        A[0] = st.dPds.x; A[1] = st.dPdt.x; A[2] = st.dPds.z; A[3] = st.dPdt.z;
        bx[0] = st.dPdx.x; bx[1] = st.dPdx.z; by[0] = st.dPdy.x; by[1] = st.dPdy.z;
    } else {
        A[0] = st.dPds.x; A[1] = st.dPdt.x; A[2] = st.dPds.y; A[3] = st.dPdt.y;
        bx[0] = st.dPdx.x; bx[1] = st.dPdx.y; by[0] = st.dPdy.x; by[1] = st.dPdy.y;
    }
    const float det = A[0] * A[3] - A[1] * A[2];
    if (isZero(det, e)) return;
    // solve2x2LinearSystem
    const float detA = differenceOfProducts(A[0], A[3], A[1], A[2]);
    if (!(fabsf(detA) >= 1.17549435e-38f) || !(fabsf(detA) <= 3.402823466e38f)) return;
    const float inv = 1.0f / detA;
    st.dsdx = differenceOfProducts(bx[0], A[3], bx[1], A[1]) * inv;
    st.dtdx = differenceOfProducts(bx[1], A[0], bx[0], A[2]) * inv;
    st.dsdy = differenceOfProducts(by[0], A[3], by[1], A[1]) * inv;
    st.dtdy = differenceOfProducts(by[1], A[0], by[0], A[2]) * inv;
}
// getdNdx and getdNdy: the shading normal across the footprint, from the vertex normals' st partials
// (TriMesh::computeIntersectCurvature); zero without vertex normals.
X2_HD inline void normalDifferentials(const MeshAttributes& m, U32 prim, U32 ia, U32 ib, U32 ic, State& st) {
    st.dNdx = st.dNdy = V3{0, 0, 0};
    if (m.normalRate == RateNone || !m.normals) return;
    float st1[2] = {0, 0}, st2[2] = {1, 0}, st3[2] = {1, 1};
    if (m.uvRate != RateNone && m.uvs) {
        const U32 i1 = m.uvRate == RateVertex ? ia : prim * 3, i2 = m.uvRate == RateVertex ? ib : prim * 3 + 1,
                  i3 = m.uvRate == RateVertex ? ic : prim * 3 + 2;
        st1[0] = m.uvs[2 * i1];
        st1[1] = m.uvs[2 * i1 + 1];
        st2[0] = m.uvs[2 * i2];
        st2[1] = m.uvs[2 * i2 + 1];
        st3[0] = m.uvs[2 * i3];
        st3[1] = m.uvs[2 * i3 + 1];
    }
    const U32 n1 = m.normalRate == RateVertex ? ia : prim * 3, n2 = m.normalRate == RateVertex ? ib : prim * 3 + 1,
              n3 = m.normalRate == RateVertex ? ic : prim * 3 + 2;
    V3 dNds, dNdt;
    if (!trianglePartials(meshNormal(m, m.normals, n1), meshNormal(m, m.normals, n2), meshNormal(m, m.normals, n3), st1, st2, st3, dNds,
                          dNdt))
        return;
    const float sign = st.entering ? 1.0f : -1.0f;
    const V3 auxX = normalize(st.N * sign + dNds * st.dsdx + dNdt * st.dtdx);
    const V3 auxY = normalize(st.N * sign + dNds * st.dsdy + dNdt * st.dtdy);
    st.dNdx = auxX * sign - st.N;
    st.dNdy = auxY * sign - st.N;
    if (!finite3(st.dNdx) || !finite3(st.dNdy)) st.dNdx = st.dNdy = V3{0, 0, 0};
}
// The texture derivatives times textureDiffScale (transferAndComputeDerivatives' last step), before dNdx and dNdy.
X2_HD inline void scaleTexDerivatives(State& st) {
    if (st.textureDiffScale == 1.0f) return;
    st.dsdx *= st.textureDiffScale;
    st.dsdy *= st.textureDiffScale;
    st.dtdx *= st.textureDiffScale;
    st.dtdy *= st.textureDiffScale;
}
// The hit's differentials: a camera ray's texture derivatives as the validated subset computes them, and
// its auxiliary rays scaled for the samples per pixel (PathIntegrator::initPrimaryRay) as the incoming
// differentials; any other ray's transferred (MoonRay's texture derivatives); then dNdx and dNdy.
X2_CALL void hitDifferentials(const MeshAttributes& m, U32 prim, U32 ia, U32 ib, U32 ic, V3 a, V3 b, V3 c,
                                   const surface::Ray& ray, U32 spp, State& st) {
    if (!isExactlyZero(ray.dx) && !isExactlyZero(ray.dy) && spp) {
        if (!(m.dpds && m.dpdt)) hitTexDerivatives(m, prim, ia, ib, ic, a, b, c, ray, spp, st);
        const float scale = 1.f / fminf(sqrtf(float(spp)), 2.f);
        RayDiff d;
        d.dOx = ray.ox * scale;
        d.dOy = ray.oy * scale;
        d.dDx = normalize(ray.direction + (ray.dx - ray.direction) * scale) - ray.direction;
        d.dDy = normalize(ray.direction + (ray.dy - ray.direction) * scale) - ray.direction;
        d.valid = 1;
        st.incoming = d;
        transferDifferentials(d, ray.origin, ray.direction, st);
        // A subdivision surface's: transferAndComputeDerivatives over its limit derivatives, as every MoonRay hit's.
        if (m.dpds && m.dpdt) differentialTexDerivatives(st);
    } else {
        transferDifferentials(st.incoming, ray.origin, ray.direction, st);
        differentialTexDerivatives(st);
    }
    scaleTexDerivatives(st);
    normalDifferentials(m, prim, ia, ib, ic, st);
}
// LineSegments::postIntersect and CubicSpline::postIntersect: a curve hit's shading state. St is the "st" or "uv"
// primitive attribute, else the span's share of its curve and v across it; dPds is the tangent (a cubic's from
// its derivative weights); N faces the ray (always on a linear curve, whose round program's normal MoonRay leaves
// unused), or is a round cubic's geometric normal. The geometric normal is the shading normal.
// A place along a span at time t (the shading points'): a linear span's lerp, a cubic's evalCubic.
X2_HD inline V3 curvePlace(const curves::Set& c, U32 vertex, float u, float t) {
    if (c.basis == curves::BasisLinear) {
        const curves::float4 a = curves::shadingPoint(c, vertex, t), b = curves::shadingPoint(c, vertex + 1, t);
        return V3{a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u};
    }
    float w[4];
    curves::cubicWeights(c.basis, u, w);
    V3 p{0, 0, 0};
    for (U32 k = 0; k < 4; ++k) {
        const curves::float4 q = curves::shadingPoint(c, vertex + k, t);
        p = p + V3{q.x, q.y, q.z} * w[k];
    }
    return p;
}
X2_CALL bool curveState(const MeshAttributes& m, U32 span, float u, float v, V3 roundNg, V3 P, V3 rayDir, State& st) {
    const curves::Set& c = m.curve;
    if (span >= c.spanCount) return false;
    const U32 vertex = curves::spanVertex(c, span), chain = curves::spanCurve(c, span), index = curves::spanIndex(c, span);
    st.curve = c.basis;
    st.chain = chain;
    st.varying = chain + span;
    st.prim = span;
    st.u = u;
    st.v = v;
    st.ia = st.ib = st.ic = vertex;
    st.prims = m.prims;
    st.primData = m.primData;
    st.primCount = m.primCount;
    st.instancePrims = m.instancePrims;
    st.instancePrimData = m.instancePrimData;
    st.instancePrimCount = m.instancePrimCount;
    float uv[3];
    if (primValue(st, primKey("st", PrimVec2f), uv) || primValue(st, primKey("uv", PrimVec2f), uv)) {
        st.s = uv[0];
        st.t = uv[1];
    } else {
        const float stSpanRange = 1.0f / float(c.curveSpans[chain]);
        const float ststart = stSpanRange * float(index);
        st.s = ststart + u * stSpanRange;
        st.t = v * 0.5f + 0.5f;  // v is in (-1,1)
    }
    const curves::float4 cp0 = curves::shadingPoint(c, vertex, c.time), cp1 = curves::shadingPoint(c, vertex + 1, c.time);
    V3 dPds, dPdt, N;
    const V3 flippedRayDir = normalize(-rayDir);
    const bool rayFacing = c.basis == curves::BasisLinear || !c.round;
    if (c.basis == curves::BasisLinear) {
        dPds = V3{cp1.x - cp0.x, cp1.y - cp0.y, cp1.z - cp0.z};
    } else {
        const curves::float4 cp[4] = {cp0, cp1, curves::shadingPoint(c, vertex + 2, c.time), curves::shadingPoint(c, vertex + 3, c.time)};
        dPds = curves::cubicTangent(c.basis, cp, u);
    }
    if (rayFacing) {
        dPdt = cross(dPds, flippedRayDir);
        if (dot(dPdt, dPdt) < Epsilon) {  // clustered points, or the ray along the curve
            N = flippedRayDir;
            const Frame f = frame(N);
            dPds = f.x;
            dPdt = f.y;
        } else {
            N = cross(dPdt, dPds);
        }
        N = normalize(N);
    } else {
        N = normalize(roundNg);
        dPdt = cross(dPds, N);
        if (dot(dPdt, dPdt) < Epsilon) {
            const Frame f = frame(N);
            dPds = f.x;
            dPdt = f.y;
        }
    }
    st.Ng = N;
    st.dPds = dPds;
    st.dPdt = dPdt;
    st.P = P;
    st.wo = -rayDir;
    // initIntersectionPhase1: N and Ng flipped to face the viewer when the ray leaves the surface.
    st.entering = dot(N, rayDir) < 0.0f;
    if (!st.entering) N = -N;
    st.N = N;
    st.Ng = N;
    // The motion vector: the hit's place along the span at the times about the ray's (sHalfDt 0.01).
    st.motion = V3{0, 0, 0};
    if (c.stepCount > 1) {
        const V3 p0 = curvePlace(c, vertex, u, c.time - 0.01f), p1 = curvePlace(c, vertex, u, c.time + 0.01f);
        st.motion = (p1 - p0) / (2.0f * 0.01f);
    }
    return finite3(st.N) && finite3(st.dPds);
}
// A curve hit's differentials (Intersection::transferAndComputeDerivatives over the hit's dPds and dPdt) and
// curvature: computeIntersectCurvature gives a linear curve dNds zero and dNdt its dPdt, a cubic both zero.
X2_HD inline void curveDifferentials(const MeshAttributes& m, const surface::Ray& ray, U32 spp, State& st) {
    if (!isExactlyZero(ray.dx) && !isExactlyZero(ray.dy) && spp) {
        const float scale = 1.f / fminf(sqrtf(float(spp)), 2.f);
        RayDiff d;
        d.dOx = ray.ox * scale;
        d.dOy = ray.oy * scale;
        d.dDx = normalize(ray.direction + (ray.dx - ray.direction) * scale) - ray.direction;
        d.dDy = normalize(ray.direction + (ray.dy - ray.direction) * scale) - ray.direction;
        d.valid = 1;
        st.incoming = d;
    }
    transferDifferentials(st.incoming, ray.origin, ray.direction, st);
    differentialTexDerivatives(st);
    scaleTexDerivatives(st);
    st.dNdx = st.dNdy = V3{0, 0, 0};
    if (m.curve.basis != curves::BasisLinear) return;
    const V3 dNdt = st.dPdt;
    const float sign = st.entering ? 1.0f : -1.0f;
    const V3 auxX = normalize(st.N * sign + dNdt * st.dtdx);
    const V3 auxY = normalize(st.N * sign + dNdt * st.dtdy);
    st.dNdx = auxX * sign - st.N;
    st.dNdy = auxY * sign - st.N;
    if (!finite3(st.dNdx) || !finite3(st.dNdy)) st.dNdx = st.dNdy = V3{0, 0, 0};
}
// scatterAndScale: a BSDF sample's ray differentials, from the hit's and the lobe's differentials function;
// a lobe that ignores the incoming differentials scales them by the primary rays' scale.
X2_HD inline bool lobeIgnoresIncomingDifferentials(const Bsdf& b, U32 record) {
    const Lobe* l = &b.records[record];
    for (int k = 0; k < 8 && (l->kind == KindUnder || l->kind == KindUnderClearcoat || l->kind == KindUnderClearcoatTransmission) &&
                    l->child != NoIndex;
         ++k)
        l = &b.records[l->child];
    if (l->kind == KindIridescence) return false;
    return (l->flags & FlagIgnoresIncomingDifferentials) != 0;
}
X2_HD inline RayDiff scatterDifferentials(const Bsdf& b, U32 record, const State& st, V3 wi, float r1, float r2, V3 childOrigin,
                                          float primaryScale) {
    RayDiff d;
    if (!st.diffValid) return d;
    V3 dDdx = st.incoming.dDx, dDdy = st.incoming.dDy;
    lobeDifferentials(b, record, st.wo, wi, r1, r2, st.dNdx, st.dNdy, dDdx, dDdy);
    const float scale = lobeIgnoresIncomingDifferentials(b, record) ? primaryScale : 1.0f;
    const V3 shift = st.P - childOrigin;
    d.dOx = st.dPdx * scale + shift;
    d.dOy = st.dPdy * scale + shift;
    d.dDx = normalize(wi + dDdx * scale) - wi;
    d.dDy = normalize(wi + dDdy * scale) - wi;
    d.valid = finite3(d.dOx) && finite3(d.dOy) && finite3(d.dDx) && finite3(d.dDy) ? 1u : 0u;
    return d;
}

// SwitchMaterial and RaySwitchMaterial (moonray dso/material): copies of their materials' records (kind 0 for none).
// A SwitchMaterial shades its chosen material (def). A RaySwitchMaterial is replaced once, by the integrator, with
// the material for the type of the lobe whose sample the ray is (PathIntegratorUtil lobeTypeToRayType: none for a
// camera ray, then diffuse, mirror, glossy in that order); shaded itself it shades its default material, or, cutting
// out camera rays, ends a primary path (BsdfBuilder early termination) after its default material's presence.
struct SwitchRecord {
    U32 byRay, cutout, pad[2];
    MaterialRecord def, camera, mirror, glossy, diffuse;
};
constexpr U32 SwitchRecordFloats = sizeof(SwitchRecord) / sizeof(float);
X2_HD inline const MaterialRecord* resolveSwitch(const MaterialRecord& m, const float* pool, const State& st, bool& cutout) {
    const MaterialRecord* cur = &m;
    bool first = true;
    for (U32 hop = 0; hop < 16 && cur && cur->kind == MatSwitch; ++hop) {
        if (cur->count < SwitchRecordFloats) return nullptr;
        const SwitchRecord& s = *reinterpret_cast<const SwitchRecord*>(pool + cur->offset);
        const MaterialRecord* next = nullptr;
        if (first && s.byRay) {
            const U32 t = st.rayLobe;
            if (t == 0) next = s.cutout ? nullptr : &s.camera;
            else if (t & Diffuse) next = &s.diffuse;
            else if (t & Mirror) next = &s.mirror;
            else if (t & Glossy) next = &s.glossy;
            if (next && next->kind == 0) next = nullptr;
        }
        first = false;
        if (!next) {
            if (s.byRay && s.cutout && st.pathType == PathPrimary) cutout = true;
            next = s.def.kind ? &s.def : nullptr;
        }
        cur = next;
    }
    return cur && cur->kind != MatSwitch ? cur : nullptr;
}

// Runs a material into a Bsdf: its maps, then the material over its bound values. The status is
// a texture lookup's: Ready, Miss (a tile is on its way: try the sample again later) or Invalid
// (the material is unknown, its Bsdf overflowed, or it needs what is not shaded yet).
template <class Tex>
X2_CALL U32 shadeChosenMaterial(const MaterialRecord& m, const float* pool, const State& st, const Tex& tex, Bsdf& bsdf, float& presence,
                                DwaParams* slots, GlitterScratch* glitter, bool cutout);
template <class Tex>
X2_CALL U32 shadeRootMaterial(const MaterialRecord& m, const float* pool, const State& st, const Tex& tex, Bsdf& bsdf, float& presence,
                              DwaParams* slots, GlitterScratch* glitter) {
    bool cutout = false;
    if (m.kind == MatSwitch) {
        const MaterialRecord* chosen = resolveSwitch(m, pool, st, cutout);
        if (!chosen) {  // nothing to shade: no lobes (presence one)
            presence = 1.0f;
            if (cutout) bsdf.earlyTermination = true;
            return tex::Ready;
        }
        return shadeChosenMaterial(*chosen, pool, st, tex, bsdf, presence, slots, glitter, cutout);
    }
    return shadeChosenMaterial(m, pool, st, tex, bsdf, presence, slots, glitter, false);
}
// The root material's extra AOVs at the hit (aovAccumExtraAovs and the post scatter ones' values): each one's colour,
// times its bound map's result where its colour is not black (evalColor).
template <class Tex>
X2_CALL U32 extraAovValues(const MaterialRecord& m, const float* pool, const State& st, const Tex& tex, Bsdf& bsdf) {
    const U32* w = reinterpret_cast<const U32*>(pool + m.extras);
    const U32 program = w[0], count = w[1] < Bsdf::MaxExtraAovs ? w[1] : Bsdf::MaxExtraAovs;
    X2_MAP_REGISTERS(regs, tex, 0);
    if (program != NoMapProgram) {
        const U32 status = runMaps(pool, program, st, tex, regs);
        if (status != tex::Ready) return status;
    }
    const float* e = pool + m.extras + 2;
    for (U32 i = 0; i < count; ++i, e += ExtraAovFloats) {
        const U32* ew = reinterpret_cast<const U32*>(e);
        Color v{e[3], e[4], e[5]};
        if (ew[2] != MapNoRegister && !isBlack(v)) v = v * regs[ew[2]];
        bsdf.extraLabel[i] = int(ew[0]);
        bsdf.extraValue[i] = v;
        if (ew[1]) bsdf.extraPost |= 1u << i;
    }
    bsdf.extraCount = count;
    return tex::Ready;
}
template <class Tex>
X2_CALL U32 shadeMaterial(const MaterialRecord& m, const float* pool, const State& st, const Tex& tex, Bsdf& bsdf, float& presence,
                          DwaParams* slots, GlitterScratch* glitter = nullptr) {
    const U32 status = shadeRootMaterial(m, pool, st, tex, bsdf, presence, slots, glitter);
    if (status != tex::Ready || m.extras == NoExtraAovs || bsdf.fatal || bsdf.earlyTermination) return status;
    return extraAovValues(m, pool, st, tex, bsdf);
}
// A material's presence alone (shading::presence, for presence shadows): its maps run, no lobes made.
template <class Tex>
X2_CALL U32 materialPresence(const MaterialRecord& m0, const float* pool, const State& st, const Tex& tex, float& presence) {
    presence = 1.0f;
    const MaterialRecord* m = &m0;
    if (m0.kind == MatSwitch) {
        bool cutout = false;
        m = resolveSwitch(m0, pool, st, cutout);
        if (!m) return tex::Ready;
    }
    X2_MAP_REGISTERS(regs, tex, 0);
    if (m->maps != NoMapProgram) {
        if (!mapRequirementsMet(pool, m->maps, st)) return tex::Ready;
        const U32 status = runMaps(pool, m->maps, st, tex, regs);
        if (status != tex::Ready) return status;
    }
    if (m->kind == MatUsdPreviewSurface && m->count >= UsdPreviewSurfaceFloats) {
        UsdPreviewSurfaceParams p = *reinterpret_cast<const UsdPreviewSurfaceParams*>(pool + m->offset);
        applyBindings(reinterpret_cast<float*>(&p), reinterpret_cast<const MapBinding*>(pool + m->bindings), m->bindingCount, regs);
        presence = presenceUsdPreviewSurface(p);
    } else if (m->kind == MatDwaBase && m->count >= DwaTreeFloats) {
        presence = dwaTreePresence(pool, *reinterpret_cast<const DwaTree*>(pool + m->offset), st, regs);
    }
    return tex::Ready;
}
// The DwaBase module's callables' places in the GPU program's binding table (async_gpu_device.h's MaterialCall).
// (An enumerator, not a constexpr: no global in every relocatable module.)
enum : U32 { DwaResolveCallable = 12, DwaLobesCallable = 13 };
// A DwaBase tree resolved and its lobes made as the material callables call them: on the GPU, in a program built in
// modules (XPU2_DWA_MODULE), the DwaBase module's callables (async_dwa.cu); elsewhere the functions themselves.
X2_HD inline bool resolveDwaTreeCall(const float* pool, const DwaTree& tree, const State& st, DwaParams* slots, const Color* regs) {
#if defined(__CUDACC__) && defined(XPU2_DWA_MODULE)
    return optixDirectCall<bool, const float*, const DwaTree*, const State*, DwaParams*, const Color*>(DwaResolveCallable, pool, &tree, &st, slots,
                                                                                                        regs);
#else
    return resolveDwaTree(pool, tree, st, slots, regs);
#endif
}
X2_HD inline bool createDwaLobesCall(const DwaParams& p, const DwaUniformParams& u, const State& st, Builder& bb, const float* pool,
                                     GlitterScratch* glitter) {
#if defined(__CUDACC__) && defined(XPU2_DWA_MODULE)
    return optixDirectCall<bool, const DwaParams*, const DwaUniformParams*, const State*, Builder*, const float*, GlitterScratch*>(
        DwaLobesCallable, &p, &u, &st, &bb, pool, glitter);
#else
    return createDwaLobes(p, u, st, bb, pool, glitter);
#endif
}
template <class Tex>
X2_CALL U32 shadeChosenMaterial(const MaterialRecord& m, const float* pool, const State& st, const Tex& tex, Bsdf& bsdf, float& presence,
                                DwaParams* slots, GlitterScratch* glitter, bool cutout) {
    X2_MAP_REGISTERS(regs, tex, 0);
    // A hit without an attribute its maps require: the fatal colour, and the path ends (hasAllRequiredAttributes).
    if (m.maps != NoMapProgram && !mapRequirementsMet(pool, m.maps, st)) {
        Builder fatal(bsdf, st);
        presence = 1.0f;
        addEmission(fatal, Color{1.0f, 0.0f, 1.0f});
        bsdf.fatal = true;
        return tex::Ready;
    }
    if (m.maps != NoMapProgram) {
        const U32 status = runMaps(pool, m.maps, st, tex, regs);
        if (status != tex::Ready) return status;
    }
    Builder bb(bsdf, st);
    bb.modoToon = (m.flags & MatModoToonTerminator) != 0;
    bb.modoTransmissionRoughness = (m.flags & MatModoTransmissionRoughness) != 0;
    if (m.labels != 0xffffffffu) {
        bb.labels = reinterpret_cast<const U32*>(pool + m.labels);
        bsdf.materialLabels = encodeLabel(bb, 0);
    }
    presence = 1.0f;
    switch (m.kind) {
    case MatUsdPreviewSurface: {
        if (m.count < UsdPreviewSurfaceFloats) return tex::Invalid;
        UsdPreviewSurfaceParams p = *reinterpret_cast<const UsdPreviewSurfaceParams*>(pool + m.offset);
        applyBindings(reinterpret_cast<float*>(&p), reinterpret_cast<const MapBinding*>(pool + m.bindings), m.bindingCount, regs);
        presence = presenceUsdPreviewSurface(p);
        if (presence <= 0.0f) return tex::Ready;  // a cut out: the integrator continues through it
        if (cutout) {
            bsdf.earlyTermination = true;
            return tex::Ready;
        }
        shadeUsdPreviewSurface(p, m.flags, st, bb);
        break;
    }
    case MatDwaBase: {
        if (m.count < DwaTreeFloats || !slots) return tex::Invalid;
        const DwaTree& tree = *reinterpret_cast<const DwaTree*>(pool + m.offset);
        // A geometry without a required attribute shades as the fatal colour (validateRequiredAttributes).
        if (tree.requiresRefP && !primProvided(st, primKey("ref_P", PrimVec3f))) {
            addEmission(bb, Color{1.0f, 0.0f, 1.0f});
            bsdf.fatal = true;
            return tex::Ready;
        }
        presence = dwaTreePresence(pool, tree, st, regs);
        if (fabsf(presence) <= 1e-6f) return tex::Ready;  // nothing of it shows (isEqual(presence, 0))
        if (cutout) {
            bsdf.earlyTermination = true;
            return tex::Ready;
        }
        initColorCorrectParameters(slots[0]);
        // A material whose resolve fails (a missing input) gives no lobes, as in MoonRay.
        if (resolveDwaTreeCall(pool, tree, st, slots, regs) && !createDwaLobesCall(slots[0], tree.up, st, bb, pool, glitter)) return tex::Invalid;
        break;
    }
    default: return tex::Invalid;
    }
    return bb.overflow ? tex::Invalid : tex::Ready;
}
}}}
