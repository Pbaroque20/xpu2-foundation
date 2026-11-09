// xpu2_geom.dll: MoonRay's own subdivision surface tessellation for XPU2 (xpu2_geom.h). It builds the control mesh as
// RdlMeshGeometry's createSubdMesh does, through MoonRay's geometry library, and tessellates it with OpenSubdivMesh's
// own tessellate (OpenSubdiv's limit surface, MoonRay's uniform or camera-adaptive edge factors), so XPU2 renders the
// very quads MoonRay intersects. Built with MoonRay's compile flags (build_geom.py), apart from XPU2's own build.
//
// Uses MoonRay's geometry library (rendering_geom, Apache-2.0, copyright 2023-2024 DreamWorks Animation LLC; see
// MOONRAY_LICENSE.txt). The tessellated buffers are OpenSubdivMesh's private members, read in place.
#include "xpu2_geom.h"

// MoonRay's include order (prim/Mesh.h): its interleaved traits before any vertex buffer.
#include <moonray/rendering/geom/internal/InterleavedTraits.h>
#include <moonray/rendering/geom/Api.h>
#include <moonray/rendering/geom/LayerAssignmentId.h>
#include <moonray/rendering/geom/MotionBlurParams.h>
#include <moonray/rendering/geom/PrimitiveUserData.h>
#include <moonray/rendering/geom/SubdivisionMesh.h>
#include <moonray/rendering/geom/prim/Mesh.h>
#include <moonray/rendering/geom/PolygonMesh.h>
#include <moonray/rendering/geom/prim/PolyMeshCalcNv.h>
#include <moonray/rendering/shading/Attributes.h>
#include <moonray/rendering/shading/PrimitiveAttribute.h>
#include <moonray/rendering/geom/prim/PrimitivePrivateAccess.h>
#include <moonray/rendering/geom/prim/SubdMesh.h>
#include <moonray/rendering/mcrt_common/Frustum.h>
#include <moonray/rendering/mcrt_common/ThreadLocalState.h>
#include <moonray/rendering/geom/prim/GeomTLState.h>
#include <moonray/rendering/shading/AttributeKey.h>
#include <moonray/rendering/shading/AttributeTable.h>
#include <moonray/rendering/shading/RootShader.h>
#include <moonray/rendering/shading/Intersection.h>
#include <moonray/rendering/shading/ShadingTLState.h>
#include <moonray/rendering/shading/State.h>
#include <moonray/rendering/shading/ThreadLocalObjectState.h>
#include <moonray/rendering/shading/Xform.h>
#include <scene_rdl2/scene/rdl2/rdl2.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <limits>
#include <array>
#include <chrono>
#include <set>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cstdlib>
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/io/GridDescriptor.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Morphology.h>
#include <openvdb/tools/ValueTransformer.h>
#define NANOVDB_USE_OPENVDB
#include <nanovdb/tools/CreateNanoGrid.h>

// The tessellated buffers (mTessellatedVertices, mSurfaceNormal, ...) are private: open them to this file alone, its
// dependencies above already included as they are.
#define private public
#define protected public
#include <moonray/rendering/geom/prim/OpenSubdivMesh.h>
#include <moonray/rendering/geom/prim/PolyMesh.h>
#include <moonray/rendering/geom/prim/QuadMesh.h>
#include <moonray/rendering/geom/prim/TriMesh.h>
#undef private
#undef protected

namespace {
using namespace moonray;
using scene_rdl2::math::Vec2f;
using scene_rdl2::math::Vec3f;
using scene_rdl2::math::Vec3fa;

// The caller's displacement for this tessellation: every Displacement the mesh's assignments have takes it as its
// displace function (the scene's displacements are proxies, without MoonRay's code), one tessellation at a time.
std::mutex displaceMutex;
X2SubdDisplaceFn displaceFn = nullptr;
X2SubdDisplaceAttrFn displaceWithFn = nullptr;
void* displaceUser = nullptr;
// The control mesh's primitive attributes this tessellation's displacements ask for (each its key and type).
const X2SubdAttribute* displaceAttributes = nullptr;
unsigned displaceAttributeCount = 0;
std::vector<shading::AttributeKey> displaceKeys;
void __cdecl displaceThroughCaller(const scene_rdl2::rdl2::Displacement* self, shading::TLState*, const shading::State& state,
                                   Vec3f* result) {
    const Vec3f P = state.getP(), N = state.getN(), dPds = state.getdPds(), dPdt = state.getdPdt();
    const Vec2f st = state.getSt();
    const float footprint[4] = {state.getdSdx(), state.getdSdy(), state.getdTdx(), state.getdTdy()};
    float out[3] = {0.f, 0.f, 0.f};
    static std::once_flag first;
    std::call_once(first, [&] {
        std::fprintf(stderr, "xpu2_geom: displacing through the caller, P %g %g %g\n", P.x, P.y, P.z);
        std::fflush(stderr);
    });
    if (displaceWithFn) {
        // The attributes as the displacement's state holds them (fillDisplacementAttributes, which its table asks for).
        thread_local std::vector<float> values;
        values.assign(size_t(displaceAttributeCount) * 3, 0.0f);
        for (unsigned i = 0; i < displaceAttributeCount && i < displaceKeys.size(); ++i) {
            float* v = values.data() + 3 * i;
            const shading::AttributeKey key = displaceKeys[i];
            const int type = displaceAttributes[i].type;
            if (!state.isProvided(key)) {
                v[0] = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
            if (type == 1) {
                const Vec2f x = state.getAttribute(shading::TypedAttributeKey<Vec2f>(key));
                v[0] = x.x;
                v[1] = x.y;
            } else if (type == 2) {
                const Vec3f x = state.getAttribute(shading::TypedAttributeKey<Vec3f>(key));
                v[0] = x.x;
                v[1] = x.y;
                v[2] = x.z;
            } else if (type == 3) {
                const scene_rdl2::math::Color x = state.getAttribute(shading::TypedAttributeKey<scene_rdl2::math::Color>(key));
                v[0] = x.r;
                v[1] = x.g;
                v[2] = x.b;
            } else if (type == 4) {
                v[0] = float(state.getAttribute(shading::TypedAttributeKey<int>(key)));
            } else {
                v[0] = state.getAttribute(shading::TypedAttributeKey<float>(key));
            }
        }
        displaceWithFn(displaceUser, self, &P.x, &N.x, &dPds.x, &dPdt.x, &st.x, footprint, displaceAttributes, displaceAttributeCount,
                       values.data(), out);
    } else if (displaceFn) displaceFn(displaceUser, self, &P.x, &N.x, &dPds.x, &dPdt.x, &st.x, footprint, out);
    *result = Vec3f(out[0], out[1], out[2]);
}
// A tessellation's displacement setup: the caller's function, and each Displacement its assignments have displacing
// through it, its table asking for every primitive attribute of the control mesh (the proxies have no table, and
// MoonRay fills a displacement's state with the attributes its table asks for).
void displaceThrough(const X2SubdInput* in, const scene_rdl2::rdl2::Layer* layer) {
    displaceFn = in->displace;
    displaceWithFn = in->displaceWith;
    displaceUser = in->displaceUser;
    displaceAttributes = in->attributes;
    displaceAttributeCount = in->attributes ? in->attributeCount : 0;
    displaceKeys.clear();
    shading::AttributeKeySet keys;
    for (unsigned i = 0; i < displaceAttributeCount; ++i) {
        const X2SubdAttribute& a = in->attributes[i];
        shading::AttributeKey key = a.type == 1   ? shading::AttributeKey(shading::TypedAttributeKey<Vec2f>(a.name))
                                    : a.type == 2 ? shading::AttributeKey(shading::TypedAttributeKey<Vec3f>(a.name))
                                    : a.type == 3 ? shading::AttributeKey(shading::TypedAttributeKey<scene_rdl2::math::Color>(a.name))
                                    : a.type == 4 ? shading::AttributeKey(shading::TypedAttributeKey<int>(a.name))
                                                  : shading::AttributeKey(shading::TypedAttributeKey<float>(a.name));
        displaceKeys.push_back(key);
        keys.insert(key);
    }
    for (unsigned f = 0; f < in->faceCount; ++f) {
        const int id = in->assignmentIds[f];
        if (id < 0) continue;
        if (const scene_rdl2::rdl2::Displacement* d = layer->lookupDisplacement(id)) {
            auto* displacement = const_cast<scene_rdl2::rdl2::Displacement*>(d);
            displacement->mDisplaceFunc = &displaceThroughCaller;
            // shading::displace times each call in the shader's thread local states (RenderContext's create callback
            // gives every shader them; the proxies here have none).
            if (!displacement->getThreadLocalObjectState())
                displacement->setThreadLocalObjectState(shading::ThreadLocalObjectState::alignedAlloc(int(mcrt_common::getMaxNumTLS())));
            if (displaceWithFn)
                displacement->getOrCreate<shading::RootShader>().setAttributeTable(
                    std::unique_ptr<shading::AttributeTable>(new shading::AttributeTable(keys, shading::AttributeKeySet())));
        }
    }
}

// OpenSubdivMesh.cc's FaceVaryingAttributes, which no header declares: its members in their order (its tessellated
// buffers, each with four indices a quad), read in place.
struct FaceVaryingBufferMirror {
    std::vector<float> mData;
    geom::SubdivisionMesh::IndexBuffer mIndices;
    int mFloatPerVertex;
    int mChannel;
};
struct FaceVaryingMirror {
    std::vector<FaceVaryingBufferMirror> mAttributeBuffers;
    std::vector<int> mKeyToBufferIndex;
};
// An input attribute's floats a value.
unsigned components(int type) { return type == 1 ? 2u : (type == 2 || type == 3) ? 3u : 1u; }
template <typename T> std::vector<T> typedValues(const X2SubdAttribute& a) {
    std::vector<T> v(a.count);
    const unsigned n = components(a.type);
    for (unsigned i = 0; i < a.count; ++i) std::memcpy(static_cast<void*>(&v[i]), a.values + size_t(i) * n, sizeof(float) * n);
    return v;
}
template <typename T> void addTyped(shading::PrimitiveAttributeTable& table, const X2SubdAttribute& a) {
    table.addAttribute(shading::TypedAttributeKey<T>(a.name), shading::AttributeRate(a.rate), typedValues<T>(a));
}
void addAttribute(shading::PrimitiveAttributeTable& table, const X2SubdAttribute& a) {
    if (a.type == 4) {
        std::vector<int> v(a.count);
        for (unsigned i = 0; i < a.count; ++i) v[i] = int(a.values[i]);
        table.addAttribute(shading::TypedAttributeKey<int>(a.name), shading::AttributeRate(a.rate), std::move(v));
    } else if (a.type == 1) addTyped<Vec2f>(table, a);
    else if (a.type == 2) addTyped<Vec3f>(table, a);
    else if (a.type == 3) addTyped<scene_rdl2::math::Color>(table, a);
    else addTyped<float>(table, a);
}
// The attribute over the tessellation (OpenSubdivMesh::setRequiredAttributes and its face varying attributes).
template <typename T> void tessellatedTyped(const geom::internal::OpenSubdivMesh& m, const X2SubdAttribute& a, X2SubdAttributeOut& out) {
    const shading::TypedAttributeKey<T> key(a.name);
    const unsigned n = sizeof(T) / sizeof(float);
    const size_t quads = m.mTessellatedIndices.size() / 4, vertices = m.mTessellatedVertices.size();
    auto put = [&](size_t i, const T& v) { std::memcpy(out.values + i * n, static_cast<const void*>(&v), sizeof(float) * n); };
    if (a.rate == shading::RATE_FACE_VARYING) {
        const auto* fv = reinterpret_cast<const FaceVaryingMirror*>(m.mFaceVaryingAttributes.get());
        const int index = key;
        if (!fv || index < 0 || size_t(index) >= fv->mKeyToBufferIndex.size() || fv->mKeyToBufferIndex[size_t(index)] < 0)
            throw std::runtime_error(std::string("the face varying attribute ") + a.name + " is not in the tessellation");
        const auto& b = fv->mAttributeBuffers[size_t(fv->mKeyToBufferIndex[size_t(index)])];
        const T* data = reinterpret_cast<const T*>(b.mData.data());
        out.kind = 4;
        out.count = unsigned(quads * 4);
        out.values = new float[size_t(out.count) * n];
        for (size_t q = 0; q < quads; ++q)
            for (size_t c = 0; c < 4; ++c) put(q * 4 + c, data[b.mIndices[4 * q + c]]);
        return;
    }
    const shading::Attributes* attributes = m.getAttributes();
    if (a.rate == shading::RATE_CONSTANT) {
        out.kind = 1;
        out.count = 1;
        out.values = new float[n];
        put(0, attributes->getConstant(key));
    } else if (a.rate == shading::RATE_UNIFORM) {
        out.kind = 2;
        out.count = unsigned(quads);
        out.values = new float[quads * n];
        for (size_t q = 0; q < quads; ++q) put(q, attributes->getUniform(key, size_t(m.mTessellatedToControlFace[q])));
    } else {
        out.kind = 3;
        out.count = unsigned(vertices);
        out.values = new float[vertices * n];
        for (size_t v = 0; v < vertices; ++v)
            put(v, a.rate == shading::RATE_VERTEX ? attributes->getVertex(key, v) : attributes->getVarying(key, v));
    }
}
void tessellatedAttribute(const geom::internal::OpenSubdivMesh& m, const X2SubdAttribute& a, X2SubdAttributeOut& out) {
    if (a.type == 4) {  // an int: constant or uniform alone (the caller's rule)
        const shading::TypedAttributeKey<int> key(a.name);
        const shading::Attributes* attributes = m.getAttributes();
        const size_t quads = m.mTessellatedIndices.size() / 4;
        if (a.rate == shading::RATE_CONSTANT) {
            out.kind = 1;
            out.count = 1;
            out.values = new float[1];
            out.values[0] = float(attributes->getConstant(key));
        } else {
            out.kind = 2;
            out.count = unsigned(quads);
            out.values = new float[quads];
            for (size_t q = 0; q < quads; ++q) out.values[q] = float(attributes->getUniform(key, size_t(m.mTessellatedToControlFace[q])));
        }
    } else if (a.type == 1) tessellatedTyped<Vec2f>(m, a, out);
    else if (a.type == 2) tessellatedTyped<Vec3f>(m, a, out);
    else if (a.type == 3) tessellatedTyped<scene_rdl2::math::Color>(m, a, out);
    else tessellatedTyped<float>(m, a, out);
}

void fail(char* error, unsigned size, const std::string& why) {
    std::fprintf(stderr, "xpu2_geom: %s\n", why.c_str());
    std::fflush(stderr);
    if (error && size) {
        std::strncpy(error, why.c_str(), size - 1);
        error[size - 1] = 0;
    }
}
}  // namespace

// RenderContext's start: MoonRay's standard attribute keys and its thread local states, once.
void prepareMoonRay() {
    // RenderContext's start: MoonRay's standard attribute keys (st, normal, velocity, ...), named once.
    static std::once_flag keys;
    std::call_once(keys, [] { shading::StandardAttributes::init(); });
    // RenderDriver's thread local states (mcrt_common::initTLS): the arenas tessellation works in, each thread's.
    static std::once_flag tls;
    std::call_once(tls, [] {
        mcrt_common::TLSInitParams params;
        params.initGeomTls = geom::internal::TLState::allocTls;
        // displaceMesh evaluates displacements through each thread's shading state.
        params.initShadingTls = shading::TLState::allocTls;
        mcrt_common::initTLS(params);
    });
}

extern "C" __declspec(dllexport) int x2SubdTessellate(const X2SubdInput* in, X2SubdOutput* out, char* error, unsigned errorSize) {
    const char* stage = "the standard attributes";
    try {
        prepareMoonRay();
        stage = "the control mesh";
        std::memset(out, 0, sizeof(*out));
        const auto* rdlGeometry = static_cast<const scene_rdl2::rdl2::Geometry*>(in->geometry);
        const auto* layer = static_cast<const scene_rdl2::rdl2::Layer*>(in->layer);
        geom::SubdivisionMesh::FaceVertexCount faceVertexCount(in->faceVertexCount, in->faceVertexCount + in->faceCount);
        geom::SubdivisionMesh::IndexBuffer indices(in->indices, in->indices + in->indexCount);
        if (indices.empty() || faceVertexCount.empty()) {
            fail(error, errorSize, "an empty subdivision mesh");
            return 0;
        }
        const size_t partCount = in->faceToPart ? in->partCount : 0;
        geom::SubdivisionMesh::FaceToPartBuffer faceToPart(in->faceCount, 0);
        if (in->faceToPart)
            for (unsigned f = 0; f < in->faceCount; ++f) faceToPart[f] = in->faceToPart[f];
        const geom::RateCounts rates{partCount, in->faceCount, in->vertexCount, in->vertexCount, in->indexCount};

        // getVertexData: the time steps' vertices (one for a still mesh), uvs and normals.
        shading::PrimitiveAttributeTable table;
        const unsigned steps = in->stepCount > 1 && in->stepVertices ? in->stepCount : 1u;
        geom::SubdivisionMesh::VertexBuffer vertices(in->vertexCount, steps);
        for (unsigned t = 0; t < steps; ++t)
            for (unsigned i = 0; i < in->vertexCount; ++i) {
                const float* p = t == 0 ? in->vertices + 3 * size_t(i) : in->stepVertices + (size_t(t - 1) * in->vertexCount + i) * 3;
                vertices(i, t) = Vec3fa(p[0], p[1], p[2], 0.f);
            }
        if (in->uvCount) {
            const shading::AttributeRate rate(geom::pickRate(rdlGeometry, "uv_list", in->uvCount, rates));
            std::vector<Vec2f> uv(in->uvCount);
            for (unsigned i = 0; i < in->uvCount; ++i) uv[i] = Vec2f(in->uvs[2 * i], in->uvs[2 * i + 1]);
            table.addAttribute(shading::StandardAttributes::sSt, rate, std::move(uv));
        }
        if (in->normalCount) {
            const shading::AttributeRate rate(geom::pickRate(rdlGeometry, "normal_list", in->normalCount, rates));
            std::vector<Vec3f> n(in->normalCount);
            for (unsigned i = 0; i < in->normalCount; ++i) n[i] = Vec3f(in->normals[3 * i], in->normals[3 * i + 1], in->normals[3 * i + 2]);
            table.addAttribute(shading::StandardAttributes::sNormal, rate, std::move(n));
        }
        for (unsigned k = 0; k < in->attributeCount; ++k) addAttribute(table, in->attributes[k]);
        if (in->leftHanded) {
            size_t offset = 0;
            for (size_t f = 0; f < faceVertexCount.size(); ++f) {
                std::reverse(indices.begin() + offset, indices.begin() + offset + faceVertexCount[f]);
                offset += faceVertexCount[f];
            }
            table.reverseFaceVaryingAttributes(faceVertexCount);
        }
        std::vector<int> ids(in->assignmentIds, in->assignmentIds + in->faceCount);
        geom::LayerAssignmentId assignment(std::move(ids));
        geom::SubdivisionMesh::Scheme scheme = geom::SubdivisionMesh::Scheme::CATMULL_CLARK;
        if (geom::SubdivisionMesh::isValidScheme(in->scheme)) scheme = static_cast<geom::SubdivisionMesh::Scheme>(in->scheme);
        stage = "createSubdivisionMesh";
        std::unique_ptr<geom::SubdivisionMesh> mesh =
            geom::createSubdivisionMesh(scheme, std::move(faceVertexCount), std::move(indices), std::move(vertices), std::move(assignment),
                                        std::move(table));
        if (!mesh) {
            fail(error, errorSize, "MoonRay made no subdivision mesh");
            return 0;
        }
        auto boundary = geom::SubdivisionMesh::BoundaryInterpolation::EDGE_AND_CORNER;
        if (geom::SubdivisionMesh::isValidBoundaryInterpolation(in->boundary))
            boundary = static_cast<geom::SubdivisionMesh::BoundaryInterpolation>(in->boundary);
        auto fvarLinear = geom::SubdivisionMesh::FVarLinearInterpolation::CORNERS_ONLY;
        if (geom::SubdivisionMesh::isValidFVarLinearInterpolation(in->fvarLinear))
            fvarLinear = static_cast<geom::SubdivisionMesh::FVarLinearInterpolation>(in->fvarLinear);
        mesh->setSubdBoundaryInterpolation(boundary);
        mesh->setSubdFVarLinearInterpolation(fvarLinear);
        geom::SubdivisionMesh::IndexBuffer creaseIndices(in->creaseIndices, in->creaseIndices + in->creaseIndexCount);
        geom::SubdivisionMesh::SharpnessBuffer creaseSharpnesses(in->creaseSharpnesses, in->creaseSharpnesses + in->creaseSharpnessCount);
        geom::SubdivisionMesh::IndexBuffer cornerIndices(in->cornerIndices, in->cornerIndices + in->cornerIndexCount);
        geom::SubdivisionMesh::SharpnessBuffer cornerSharpnesses(in->cornerSharpnesses, in->cornerSharpnesses + in->cornerSharpnessCount);
        if (!creaseIndices.empty() && !creaseSharpnesses.empty()) mesh->setSubdCreases(std::move(creaseIndices), std::move(creaseSharpnesses));
        if (!cornerIndices.empty() && !cornerSharpnesses.empty()) mesh->setSubdCorners(std::move(cornerIndices), std::move(cornerSharpnesses));
        mesh->setMeshResolution(in->meshResolution);
        mesh->setAdaptiveError(in->adaptiveError);
        mesh->setName("xpu2");
        mesh->setIsSingleSided(in->singleSided != 0);
        mesh->setIsNormalReversed(in->reverseNormals != 0);
        mesh->setModifiability(geom::Primitive::Modifiability::DEFORMABLE);
        mesh->setParts(partCount, std::move(faceToPart));

        // Procedural::addPrimitive's transform into render space: the vertices are there already.
        stage = "transformPrimitive";
        auto* impl = static_cast<geom::internal::OpenSubdivMesh*>(geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(mesh.get()));
        if (steps == 1) {
            const geom::MotionBlurParams motion(std::vector<float>{0.0f}, 0.0f, 0.0f, false, 24.0f);
            const shading::XformSamples prim2render{scene_rdl2::math::Xform3f(scene_rdl2::math::one)};
            geom::internal::PrimitivePrivateAccess::transformPrimitive(mesh.get(), motion, prim2render);
        } else {
            // SubdivisionMesh::transformPrimitive's setTransform alone: the steps are in render space already, and its
            // transformVertexBuffer without motion blur would keep the first.
            impl->setTransform(shading::XformSamples(steps, scene_rdl2::math::Xform3f(scene_rdl2::math::one)), 0.0f, 0.0f);
        }
        impl->setRdlGeometry(rdlGeometry);
        if (in->reference) impl->setIsReference(true);

        std::vector<mcrt_common::Frustum> frustums(in->frustumCount);
        for (unsigned k = 0; k < in->frustumCount; ++k) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) frustums[k].mC2S[i][j] = in->frustumC2S[16 * k + 4 * i + j];
            for (int p = 0; p < 6; ++p)
                for (int c = 0; c < 4; ++c) frustums[k].mClipPlanes[p][c] = in->frustumPlanes[24 * k + 4 * p + c];
            for (int i = 0; i < 4; ++i) frustums[k].mViewport[i] = in->frustumViewports[4 * k + i];
        }
        std::vector<mcrt_common::Fishtum> fishtums;
        if (in->hasFishtum) {
            mcrt_common::Fishtum f;
            f.mRadialScale = in->fishtum[0];
            f.mDerivScale = in->fishtum[1];
            f.mMapping = int(in->fishtum[2]);
            f.mApertureWindowWidth = in->fishtum[3];
            f.mApertureWindowHeight = in->fishtum[4];
            f.mRegionWindowXMin = in->fishtum[5];
            f.mRegionWindowYMin = in->fishtum[6];
            f.mRegionWindowXMax = in->fishtum[7];
            f.mRegionWindowYMax = in->fishtum[8];
            fishtums.push_back(f);
        }
        scene_rdl2::math::Mat4d world2render;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) world2render[i][j] = in->world2render[4 * i + j];
        const bool displacing = in->enableDisplacement != 0 && (in->displace != nullptr || in->displaceWith != nullptr);
        std::unique_lock<std::mutex> displaceLock(displaceMutex, std::defer_lock);
        if (displacing) {
            displaceLock.lock();
            displaceThrough(in, layer);
        }
        if (displacing) {
            std::fprintf(stderr, "xpu2_geom: displacement on, shading state %p\n",
                         static_cast<void*>(mcrt_common::getFrameUpdateTLS()->mShadingTls.get()));
            std::fflush(stderr);
        }
        const geom::internal::TessellationParams params(layer, frustums, fishtums, world2render, displacing,
                                                        /* fastGeomUpdate */ false, /* isBaking */ false, nullptr);
        geom::internal::TessellationStats stats;
        stage = "tessellate";
        std::fprintf(stderr, "xpu2_geom: tessellating %u faces, resolution %d\n", in->faceCount, in->meshResolution);
        std::fflush(stderr);
        impl->tessellate(params, stats);
        stage = "the tessellated buffers";

        // The tessellated mesh at its shutter's open.
        const size_t vertexCount = impl->mTessellatedVertices.size(), indexCount = impl->mTessellatedIndices.size();
        out->vertexCount = unsigned(vertexCount);
        out->quadCount = unsigned(indexCount / 4);
        out->vertices = new float[3 * vertexCount];
        out->normals = new float[3 * vertexCount];
        out->st = new float[2 * vertexCount];
        out->dpds = new float[3 * vertexCount];
        out->dpdt = new float[3 * vertexCount];
        out->indices = new unsigned[indexCount];
        out->controlFace = new int[out->quadCount];
        const bool hasSt = impl->mSurfaceSt.size() == vertexCount, hasNormals = impl->mSurfaceNormal.size() == vertexCount,
                   hasDpds = impl->mSurfaceDpds.size() == vertexCount, hasDpdt = impl->mSurfaceDpdt.size() == vertexCount;
        for (size_t i = 0; i < vertexCount; ++i) {
            const Vec3fa p = impl->mTessellatedVertices(i, 0);
            const Vec3f n = hasNormals ? impl->mSurfaceNormal(i, 0) : Vec3f(0.f);
            const Vec2f s = hasSt ? impl->mSurfaceSt(i) : Vec2f(0.f);
            const Vec3f ds = hasDpds ? impl->mSurfaceDpds(i, 0) : Vec3f(0.f), dt = hasDpdt ? impl->mSurfaceDpdt(i, 0) : Vec3f(0.f);
            const float vals[] = {p.x, p.y, p.z};
            std::memcpy(out->vertices + 3 * i, vals, sizeof vals);
            out->normals[3 * i] = n.x, out->normals[3 * i + 1] = n.y, out->normals[3 * i + 2] = n.z;
            out->st[2 * i] = s.x, out->st[2 * i + 1] = s.y;
            out->dpds[3 * i] = ds.x, out->dpds[3 * i + 1] = ds.y, out->dpds[3 * i + 2] = ds.z;
            out->dpdt[3 * i] = dt.x, out->dpdt[3 * i + 1] = dt.y, out->dpdt[3 * i + 2] = dt.z;
        }
        // The later time steps on the same tessellation.
        const unsigned outSteps = unsigned(impl->mTessellatedVertices.get_time_steps());
        if (steps > 1 && outSteps > 1) {
            out->stepCount = outSteps;
            const size_t later = size_t(outSteps - 1) * vertexCount * 3;
            out->stepVertices = new float[later];
            out->stepNormals = new float[later];
            out->stepDpds = new float[later];
            out->stepDpdt = new float[later];
            const bool stepNormals = hasNormals && impl->mSurfaceNormal.get_time_steps() >= outSteps,
                       stepDpds = hasDpds && impl->mSurfaceDpds.get_time_steps() >= outSteps,
                       stepDpdt = hasDpdt && impl->mSurfaceDpdt.get_time_steps() >= outSteps;
            for (unsigned t = 1; t < outSteps; ++t)
                for (size_t i = 0; i < vertexCount; ++i) {
                    const size_t at = 3 * (size_t(t - 1) * vertexCount + i);
                    const Vec3fa p = impl->mTessellatedVertices(i, t);
                    const Vec3f n = stepNormals ? impl->mSurfaceNormal(i, t) : Vec3f(out->normals[3 * i], out->normals[3 * i + 1], out->normals[3 * i + 2]);
                    const Vec3f ds = stepDpds ? impl->mSurfaceDpds(i, t) : Vec3f(out->dpds[3 * i], out->dpds[3 * i + 1], out->dpds[3 * i + 2]);
                    const Vec3f dt = stepDpdt ? impl->mSurfaceDpdt(i, t) : Vec3f(out->dpdt[3 * i], out->dpdt[3 * i + 1], out->dpdt[3 * i + 2]);
                    out->stepVertices[at] = p.x, out->stepVertices[at + 1] = p.y, out->stepVertices[at + 2] = p.z;
                    out->stepNormals[at] = n.x, out->stepNormals[at + 1] = n.y, out->stepNormals[at + 2] = n.z;
                    out->stepDpds[at] = ds.x, out->stepDpds[at + 1] = ds.y, out->stepDpds[at + 2] = ds.z;
                    out->stepDpdt[at] = dt.x, out->stepDpdt[at + 1] = dt.y, out->stepDpdt[at + 2] = dt.z;
                }
        }
        for (size_t i = 0; i < indexCount; ++i) out->indices[i] = impl->mTessellatedIndices[i];
        for (unsigned q = 0; q < out->quadCount; ++q)
            out->controlFace[q] = q < impl->mTessellatedToControlFace.size() ? impl->mTessellatedToControlFace[q] : -1;
        stage = "the tessellated attributes";
        if (in->attributeCount) {
            out->attributes = new X2SubdAttributeOut[in->attributeCount]();
            out->attributeCount = in->attributeCount;
            for (unsigned k = 0; k < in->attributeCount; ++k) tessellatedAttribute(*impl, in->attributes[k], out->attributes[k]);
        }
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("in ") + stage + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("in ") + stage + ": an unknown exception");
    }
    return 0;
}

// A float tree's leaves and active tiles as leaf-sized cells (VolumeHDDA's leaf level: a leaf or an active tile).
void leafBlocks(const openvdb::FloatTree& tree, X2VdbGrid* out) {
    using LeafT = openvdb::FloatTree::LeafNodeType;
    const int dim = int(LeafT::DIM);
    std::vector<int> cells;
    std::set<std::array<int, 3>> seen;
    auto add = [&](int x, int y, int z) {
        const std::array<int, 3> c{x, y, z};
        if (seen.insert(c).second) cells.insert(cells.end(), {x, y, z});
    };
    for (auto leaf = tree.cbeginLeaf(); leaf; ++leaf) {
        const openvdb::Coord o = leaf->origin();
        add(o.x(), o.y(), o.z());
    }
    for (auto it = tree.cbeginValueOn(); it; ++it) {
        if (it.isVoxelValue()) continue;
        openvdb::CoordBBox b;
        it.getBoundingBox(b);
        for (int x = b.min().x(); x <= b.max().x(); x += dim)
            for (int y = b.min().y(); y <= b.max().y(); y += dim)
                for (int z = b.min().z(); z <= b.max().z(); z += dim) add(x & ~(dim - 1), y & ~(dim - 1), z & ~(dim - 1));
    }
    out->blockSize = unsigned(dim);
    out->blockCount = unsigned(cells.size() / 3);
    out->blocks = new int[cells.size() ? cells.size() : 1];
    std::copy(cells.begin(), cells.end(), out->blocks);
}

// MOONRAY_MODO_SCALAR_EMISSION (MoonRay's patched VdbVolume, tools/port_scalar_emission.py): a float emission grid glows grey.
bool modoScalarEmission() {
    static const bool on = [] {
        const char* value = std::getenv("MOONRAY_MODO_SCALAR_EMISSION");
        return value && value[0] == '1';
    }();
    return on;
}
// The float grid as an RGB grid of the same shape, each value v as (v, v, v): VdbVolume.cc's modoGreyGrid, step for step.
openvdb::VectorGrid::Ptr modoGreyGrid(const openvdb::FloatGrid& scalar) {
    const float background = scalar.background();
    openvdb::VectorGrid::Ptr grey = openvdb::VectorGrid::create(openvdb::Vec3f(background, background, background));
    grey->setTransform(scalar.transform().copy());
    grey->setName(scalar.getName());
    grey->tree().topologyUnion(scalar.tree());
    const openvdb::FloatGrid::ConstAccessor values = scalar.getConstAccessor();
    for (auto it = grey->beginValueAll(); it; ++it) {
        const float v = values.getValue(it.getCoord());
        it.setValue(openvdb::Vec3f(v, v, v));
    }
    return grey;
}
// An emission grid as RGB: an RGB grid as it is, a float one grey where the switch is on; none for another.
openvdb::VectorGrid::Ptr emissionAsRgb(openvdb::io::File& file, const openvdb::GridBase::Ptr& meta, const std::string& gridName) {
    if (!meta) return nullptr;
    if (meta->isType<openvdb::VectorGrid>()) return openvdb::gridPtrCast<openvdb::VectorGrid>(file.readGrid(gridName));
    if (meta->isType<openvdb::FloatGrid>() && modoScalarEmission()) {
        const auto scalar = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(gridName));
        return scalar ? modoGreyGrid(*scalar) : nullptr;
    }
    return nullptr;
}
// VdbVolume.cc's findNthGridByName: the nth grid of a name ("name[n]" in OpenVDB's unique-name form), the first
// without a suffix.
openvdb::GridBase::Ptr findNthGrid(const openvdb::GridPtrVec& grids, const std::string& name) {
    int gridIndex = 0;
    const std::string uniqueName = openvdb::io::GridDescriptor::stringAsUniqueName(name);
    const std::string strippedName = openvdb::io::GridDescriptor::stripSuffix(uniqueName);
    if (uniqueName != strippedName) {
        static constexpr char delimiter = 30;
        gridIndex = std::stoi(uniqueName.substr(uniqueName.find(delimiter) + 1, uniqueName.size()));
    }
    int index = 0;
    for (const auto& grid : grids)
        if (grid && grid->getName() == strippedName && gridIndex == index++) return grid;
    return openvdb::GridBase::Ptr();
}
template <typename GridT> void toNano(const GridT& grid, X2VdbGrid* out) {
    auto handle = nanovdb::tools::createNanoGrid(grid);
    out->bytes = handle.bufferSize();
    out->data = ::operator new[](size_t(out->bytes), std::align_val_t(32));  // NanoVDB's data alignment
    std::memcpy(out->data, handle.data(), size_t(out->bytes));
}

extern "C" __declspec(dllexport) int x2VdbRead(const char* path, const char* gridName, int vec3, X2VdbGrid* out, char* error, unsigned errorSize) {
    try {
        static std::once_flag init;
        std::call_once(init, [] { openvdb::initialize(); });
        std::memset(out, 0, sizeof(*out));
        openvdb::io::File file(path);
        file.open();
        const openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
        if (!grids || grids->empty()) {
            fail(error, errorSize, std::string("the VDB file ") + path + " has no grids");
            return 0;
        }
        const openvdb::GridBase::Ptr meta = findNthGrid(*grids, gridName);
        if (!meta) {
            fail(error, errorSize, std::string("the grid \"") + gridName + "\" is not in " + path);
            return 0;
        }
        const bool grey = vec3 && meta->isType<openvdb::FloatGrid>() && modoScalarEmission();
        if (vec3 ? !meta->isType<openvdb::VectorGrid>() && !grey : !meta->isType<openvdb::FloatGrid>()) {
            fail(error, errorSize, std::string("the grid \"") + gridName + (vec3 ? "\" is not an RGB grid" : "\" is not a float grid"));
            return 0;
        }
        const openvdb::GridBase::Ptr grid = grey ? openvdb::GridBase::Ptr(emissionAsRgb(file, meta, gridName)) : file.readGrid(gridName);
        if (!grid || grid->empty()) {
            fail(error, errorSize, std::string("the grid \"") + gridName + "\" is empty");
            return 0;
        }
        out->vec3 = vec3;
        const openvdb::math::Transform& xform = grid->transform();
        out->linear = xform.isLinear() ? 1 : 0;
        out->uniform = grid->hasUniformVoxels() ? 1 : 0;
        if (xform.isLinear()) {
            const openvdb::math::Mat4d m = xform.baseMap()->getAffineMap()->getMat4();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) out->indexToWorld[4 * i + j] = m(i, j);
        }
        const openvdb::Vec3d vs = grid->voxelSize();
        for (int k = 0; k < 3; ++k) out->voxelSize[k] = vs[k];
        const openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
        for (int k = 0; k < 3; ++k) {
            out->bboxMin[k] = box.min()[k];
            out->bboxMax[k] = box.max()[k];
        }
        if (vec3) toNano(*openvdb::gridPtrCast<openvdb::VectorGrid>(grid), out);
        else {
            const auto floats = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
            toNano(*floats, out);
            leafBlocks(floats->tree(), out);
        }
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("reading ") + path + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("reading ") + path + ": an unknown exception");
    }
    return 0;
}

extern "C" __declspec(dllexport) int x2VdbHistogram(const char* path, const char* gridName, float shaderLuminance, X2VdbHistogram* out,
                                                     char* error, unsigned errorSize) {
    try {
        static std::once_flag init;
        std::call_once(init, [] { openvdb::initialize(); });
        std::memset(out, 0, sizeof(*out));
        openvdb::io::File file(path);
        file.open();
        const openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
        const openvdb::GridBase::Ptr meta = grids ? findNthGrid(*grids, gridName) : openvdb::GridBase::Ptr();
        const auto grid = emissionAsRgb(file, meta, gridName);
        if (!grid) {
            fail(error, errorSize, std::string("the emission grid \"") + gridName + "\" is not an RGB grid of " + path);
            return 0;
        }
        // getGridLuminance: each active value's luminance, in the value iterator's order.
        std::vector<float> lum;
        for (auto it = grid->cbeginValueOn(); it; ++it) {
            const openvdb::Vec3s v = *it;
            lum.push_back(0.299f * v.x() + 0.587f * v.y() + 0.114f * v.z());
        }
        const float epsilon = 1.1920928955078125e-07f;  // scene_rdl2 sEpsilon
        openvdb::CoordBBox bbox;
        size_t k = 0;
        for (auto it = grid->cbeginValueOn(); it; ++it) {
            const float value = lum[k++];
            if (value <= epsilon || !std::isfinite(value)) continue;
            if (it.isVoxelValue()) bbox.expand(it.getCoord());
            else {
                openvdb::CoordBBox bound;
                it.getBoundingBox(bound);
                bbox.expand(bound);
            }
        }
        const openvdb::Coord pMin = bbox.min(), dim = bbox.dim();
        for (int a = 0; a < 3; ++a) {
            out->pMin[a] = pMin[a];
            out->res[a] = dim[a];
        }
        const size_t n = size_t(dim[0]) * size_t(dim[1]) * size_t(dim[2]);
        out->values = new float[n]();
        auto index = [&](const openvdb::Coord& ijk) {
            return size_t(ijk[2] - pMin[2]) * size_t(dim[0]) * size_t(dim[1]) + size_t(ijk[1] - pMin[1]) * size_t(dim[0]) + size_t(ijk[0] - pMin[0]);
        };
        k = 0;
        for (auto it = grid->cbeginValueOn(); it.test(); ++it) {
            float value = lum[k++];
            if (value <= epsilon || !std::isfinite(value)) continue;
            if (it.isVoxelValue()) {
                value *= shaderLuminance;
                out->values[index(it.getCoord())] += value;
            } else {
                openvdb::CoordBBox bound;
                it.getBoundingBox(bound);
                for (openvdb::CoordBBox::Iterator<true> ijk(bound); ijk; ++ijk) {
                    value *= shaderLuminance;  // as MoonRay: along the tile
                    out->values[index(*ijk)] += value;
                }
            }
        }
        const openvdb::math::Mat4d m = grid->transform().baseMap()->getAffineMap()->getMat4();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) out->indexToWorld[4 * i + j] = m(i, j);
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("reading ") + path + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("reading ") + path + ": an unknown exception");
    }
    return 0;
}

extern "C" __declspec(dllexport) int x2VdbHistogramMapped(const char* path, const char* gridName, X2VdbShaderFn shader, void* user,
                                                           X2VdbHistogram* out, char* error, unsigned errorSize) {
    try {
        static std::once_flag init;
        std::call_once(init, [] { openvdb::initialize(); });
        std::memset(out, 0, sizeof(*out));
        openvdb::io::File file(path);
        file.open();
        const openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
        const openvdb::GridBase::Ptr meta = grids ? findNthGrid(*grids, gridName) : openvdb::GridBase::Ptr();
        const auto grid = emissionAsRgb(file, meta, gridName);
        if (!grid) {
            fail(error, errorSize, std::string("the emission grid \"") + gridName + "\" is not an RGB grid of " + path);
            return 0;
        }
        std::vector<float> lum;
        for (auto it = grid->cbeginValueOn(); it; ++it) {
            const openvdb::Vec3s v = *it;
            lum.push_back(0.299f * v.x() + 0.587f * v.y() + 0.114f * v.z());
        }
        const float epsilon = 1.1920928955078125e-07f;
        openvdb::CoordBBox bbox;
        size_t k = 0;
        for (auto it = grid->cbeginValueOn(); it; ++it) {
            const float value = lum[k++];
            if (value <= epsilon || !std::isfinite(value)) continue;
            if (it.isVoxelValue()) bbox.expand(it.getCoord());
            else {
                openvdb::CoordBBox bound;
                it.getBoundingBox(bound);
                bbox.expand(bound);
            }
        }
        const openvdb::Coord pMin = bbox.min(), dim = bbox.dim();
        for (int a = 0; a < 3; ++a) {
            out->pMin[a] = pMin[a];
            out->res[a] = dim[a];
        }
        const size_t n = size_t(dim[0]) * size_t(dim[1]) * size_t(dim[2]);
        out->values = new float[n]();
        auto index = [&](const openvdb::Coord& ijk) {
            return size_t(ijk[2] - pMin[2]) * size_t(dim[0]) * size_t(dim[1]) + size_t(ijk[1] - pMin[1]) * size_t(dim[0]) + size_t(ijk[0] - pMin[0]);
        };
        auto evalShader = [&](const openvdb::Coord& c) {
            const openvdb::Vec3d pd = grid->indexToWorld(c);
            const double p[3] = {pd.x(), pd.y(), pd.z()};
            return shader(user, p);
        };
        k = 0;
        for (auto it = grid->cbeginValueOn(); it.test(); ++it) {
            float value = lum[k++];
            if (value <= epsilon || !std::isfinite(value)) continue;
            if (it.isVoxelValue()) {
                const openvdb::Coord& c = it.getCoord();
                value *= evalShader(c);
                out->values[index(c)] += value;
            } else {
                openvdb::CoordBBox bound;
                it.getBoundingBox(bound);
                for (openvdb::CoordBBox::Iterator<true> ijk(bound); ijk; ++ijk) {
                    value *= evalShader(*ijk);
                    out->values[index(*ijk)] += value;
                }
            }
        }
        const openvdb::math::Mat4d m = grid->transform().baseMap()->getAffineMap()->getMat4();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) out->indexToWorld[4 * i + j] = m(i, j);
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("reading ") + path + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("reading ") + path + ": an unknown exception");
    }
    return 0;
}

extern "C" __declspec(dllexport) int x2VdbVelocity(const char* path, const char* velocityName, const char* densityName, float scale, float sampleRate,
                                                    float tShutterOpen, float tShutterClose, X2VdbVelocity* out, char* error, unsigned errorSize) {
    try {
        static std::once_flag init;
        std::call_once(init, [] { openvdb::initialize(); });
        std::memset(out, 0, sizeof(*out));
        openvdb::io::File file(path);
        file.open();
        const openvdb::GridPtrVecPtr grids = file.readAllGridMetadata();
        const openvdb::GridBase::Ptr velocityMeta = grids ? findNthGrid(*grids, velocityName) : openvdb::GridBase::Ptr();
        if (!velocityMeta || !velocityMeta->isType<openvdb::VectorGrid>()) return 2;
        openvdb::GridBase::Ptr velocityBase = file.readGrid(velocityName);
        if (!velocityBase || velocityBase->empty()) return 2;
        openvdb::VectorGrid::Ptr velocityGrid = openvdb::gridPtrCast<openvdb::VectorGrid>(velocityBase);
        openvdb::tools::foreach(velocityGrid->beginValueOn(), [&](const openvdb::VectorGrid::ValueOnIter& iter) { iter.setValue(*iter * scale); });
        const openvdb::GridBase::Ptr densityMeta = findNthGrid(*grids, densityName);
        if (!densityMeta || !densityMeta->isType<openvdb::FloatGrid>()) {
            fail(error, errorSize, std::string("the density grid \"") + densityName + "\" is not a float grid of " + path);
            return 0;
        }
        openvdb::FloatGrid::Ptr topologyGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(densityName));
        // initMotionBlurBoundary (one thread: the padding grids' union and maximum are the same)
        {
            const openvdb::VectorGrid::ConstAccessor vAccessor = velocityGrid->getConstAccessor();
            openvdb::FloatGrid::Ptr paddedGrid = openvdb::FloatGrid::create();
            paddedGrid->setTransform(topologyGrid->transformPtr());
            openvdb::FloatGrid::Accessor paddedGridAccessor = paddedGrid->getAccessor();
            openvdb::VectorGrid::Ptr paddedVGrid = openvdb::VectorGrid::create();
            paddedVGrid->setTransform(velocityGrid->transformPtr());
            const float voxelLength = float(topologyGrid->voxelSize().z());
            for (auto leaf = topologyGrid->tree().cbeginLeaf(); leaf; ++leaf) {
                for (auto iter = leaf->cbeginValueOn(); iter; ++iter) {
                    openvdb::Vec3s velocity;
                    const openvdb::math::Coord& coord = iter.getCoord();
                    const openvdb::Vec3d p = topologyGrid->indexToWorld(coord);
                    openvdb::math::Coord vcoord = openvdb::math::Coord::round(velocityGrid->worldToIndex(p));
                    if (!vAccessor.probeValue(vcoord, velocity)) continue;
                    const float dt = voxelLength / velocity.length();
                    const float maxT = std::max(std::abs(tShutterOpen), std::abs(tShutterClose));
                    for (float t = -maxT; t <= maxT; t += dt) {
                        const openvdb::Vec3d pPrime = p + velocity * t;
                        paddedGridAccessor.setValueOn(openvdb::math::Coord::round(paddedGrid->worldToIndex(pPrime)));
                        vcoord = openvdb::math::Coord::round(velocityGrid->worldToIndex(pPrime));
                        if (!vAccessor.isValueOn(vcoord)) openvdb::tools::setValueOnMax(paddedVGrid->tree(), vcoord, velocity);
                    }
                }
            }
            topologyGrid->topologyUnion(*paddedGrid);
            openvdb::tools::compMax(*velocityGrid, *paddedVGrid);
            openvdb::tools::dilateActiveValues(topologyGrid->tree(), 1, openvdb::tools::NN_FACE, openvdb::tools::EXPAND_TILES);
        }
        const float rate = std::min(std::max(sampleRate, 0.0f), 1.0f);
        if (rate > 0.0f && rate < 1.0f) {
            openvdb::VectorGrid::Ptr decimated(new openvdb::VectorGrid());
            decimated->setTransform(velocityGrid->transform().copy());
            decimated->transform().preScale(1.0f / rate);
            openvdb::tools::resampleToMatch<openvdb::tools::BoxSampler>(*velocityGrid, *decimated);
            velocityGrid = decimated;
        }
        if (!velocityGrid->transform().isLinear()) {
            fail(error, errorSize, std::string("the velocity grid \"") + velocityName + "\" has a frustum transform");
            return 0;
        }
        X2VdbGrid& v = out->velocity;
        v.vec3 = 1;
        v.linear = 1;
        v.uniform = velocityGrid->hasUniformVoxels() ? 1 : 0;
        const openvdb::math::Mat4d m = velocityGrid->transform().baseMap()->getAffineMap()->getMat4();
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) v.indexToWorld[4 * i + j] = m(i, j);
        const openvdb::Vec3d vs = velocityGrid->voxelSize();
        for (int a = 0; a < 3; ++a) v.voxelSize[a] = vs[a];
        const openvdb::CoordBBox vbox = velocityGrid->evalActiveVoxelBoundingBox();
        for (int a = 0; a < 3; ++a) {
            v.bboxMin[a] = vbox.min()[a];
            v.bboxMax[a] = vbox.max()[a];
        }
        toNano(*velocityGrid, &v);
        leafBlocks(topologyGrid->tree(), &v);  // the padded density's cells, which the march then covers
        const openvdb::CoordBBox box = topologyGrid->evalActiveVoxelBoundingBox();
        for (int a = 0; a < 3; ++a) {
            out->densityMin[a] = box.min()[a];
            out->densityMax[a] = box.max()[a];
        }
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("reading ") + path + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("reading ") + path + ": an unknown exception");
    }
    return 0;
}

extern "C" __declspec(dllexport) void x2VdbHistogramFree(X2VdbHistogram* histogram) {
    delete[] histogram->values;
    std::memset(histogram, 0, sizeof(*histogram));
}

extern "C" __declspec(dllexport) void x2VdbFree(X2VdbGrid* grid) {
    if (grid->data) ::operator delete[](grid->data, std::align_val_t(32));
    delete[] grid->blocks;
    std::memset(grid, 0, sizeof(*grid));
}

// The camera frustums MoonRay tessellates adaptively by, as the input gives them.
void cameraFrustums(const X2SubdInput* in, std::vector<mcrt_common::Frustum>& frustums, std::vector<mcrt_common::Fishtum>& fishtums) {
    frustums.assign(in->frustumCount, mcrt_common::Frustum());
    for (unsigned k = 0; k < in->frustumCount; ++k) {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) frustums[k].mC2S[i][j] = in->frustumC2S[16 * k + 4 * i + j];
        for (int p = 0; p < 6; ++p)
            for (int c = 0; c < 4; ++c) frustums[k].mClipPlanes[p][c] = in->frustumPlanes[24 * k + 4 * p + c];
        for (int i = 0; i < 4; ++i) frustums[k].mViewport[i] = in->frustumViewports[4 * k + i];
    }
    fishtums.clear();
    if (in->hasFishtum) {
        mcrt_common::Fishtum f;
        f.mRadialScale = in->fishtum[0];
        f.mDerivScale = in->fishtum[1];
        f.mMapping = int(in->fishtum[2]);
        f.mApertureWindowWidth = in->fishtum[3];
        f.mApertureWindowHeight = in->fishtum[4];
        f.mRegionWindowXMin = in->fishtum[5];
        f.mRegionWindowYMin = in->fishtum[6];
        f.mRegionWindowXMax = in->fishtum[7];
        f.mRegionWindowYMax = in->fishtum[8];
        fishtums.push_back(f);
    }
}

// QuadMesh.cc's bilinearInterpolate.
template <typename T> T bilinear(const Vec2f& uv, const T& a0, const T& a1, const T& a2, const T& a3) {
    return (1.0f - uv[0]) * (1.0f - uv[1]) * a0 +
           (       uv[0]) * (1.0f - uv[1]) * a1 +
           (       uv[0]) * (       uv[1]) * a2 +
           (1.0f - uv[0]) * (       uv[1]) * a3;
}
// QuadMesh::getAttribute and TriMesh::getAttribute at a hit face's corner: a quad base face's four values bilinear at
// the corner's place in it, a triangle's weighted (1 - u - v, u, v); a vertex rate value the face vertex's own.
struct PolyCorners {
    const geom::internal::PolyMesh& m;
    bool quadBase, tessellated;
    size_t nb, nt;  // a base face's and a hit face's vertices
    template <typename T> T at(const shading::TypedAttributeKey<T>& key, int baseFace, int face, int corner) const {
        const shading::Attributes* a = m.getAttributes();
        const size_t b = size_t(baseFace), f = size_t(face), c = size_t(corner);
        switch (a->getRate(key)) {
        case shading::RATE_CONSTANT: return a->getConstant(key);
        case shading::RATE_UNIFORM: return a->getUniform(key, b);
        case shading::RATE_VERTEX: return a->getVertex(key, m.mIndices[nt * f + c]);
        case shading::RATE_VARYING: {
            if (!tessellated) return a->getVarying(key, m.mIndices[nb * b + c]);
            const Vec2f uv = m.mFaceVaryingUv[4 * f + c];
            const size_t o = nb * b;
            if (quadBase)
                return bilinear(uv, a->getVarying(key, m.mBaseIndices[o]), a->getVarying(key, m.mBaseIndices[o + 1]),
                                a->getVarying(key, m.mBaseIndices[o + 2]), a->getVarying(key, m.mBaseIndices[o + 3]));
            return (1.0f - uv[0] - uv[1]) * a->getVarying(key, m.mBaseIndices[o]) + uv[0] * a->getVarying(key, m.mBaseIndices[o + 1]) +
                   uv[1] * a->getVarying(key, m.mBaseIndices[o + 2]);
        }
        case shading::RATE_FACE_VARYING: {
            if (!tessellated) return a->getFaceVarying(key, b, c);
            const Vec2f uv = m.mFaceVaryingUv[4 * f + c];
            if (quadBase)
                return bilinear(uv, a->getFaceVarying(key, b, 0), a->getFaceVarying(key, b, 1), a->getFaceVarying(key, b, 2),
                                a->getFaceVarying(key, b, 3));
            return (1.0f - uv[0] - uv[1]) * a->getFaceVarying(key, b, 0) + uv[0] * a->getFaceVarying(key, b, 1) +
                   uv[1] * a->getFaceVarying(key, b, 2);
        }
        default: throw std::runtime_error("an attribute of an unexpected rate");
        }
    }
};
// An input attribute over the hit faces.
template <typename T> void polyTyped(const PolyCorners& pc, const std::vector<int>& baseOf, const X2SubdAttribute& a, X2SubdAttributeOut& out) {
    const shading::TypedAttributeKey<T> key(a.name);
    const shading::Attributes* attributes = pc.m.getAttributes();
    if (!attributes->isSupported(key)) return;  // kind 0: removed with the unassigned faces
    const unsigned n = sizeof(T) / sizeof(float);
    const size_t faces = baseOf.size();
    auto put = [&](size_t i, const T& v) { std::memcpy(out.values + i * n, static_cast<const void*>(&v), sizeof(float) * n); };
    const shading::AttributeRate rate = attributes->getRate(key);
    if (rate == shading::RATE_CONSTANT) {
        out.kind = 1;
        out.count = 1;
        out.values = new float[n];
        put(0, attributes->getConstant(key));
    } else if (rate == shading::RATE_UNIFORM) {
        out.kind = 2;
        out.count = unsigned(faces);
        out.values = new float[faces * n];
        for (size_t f = 0; f < faces; ++f) put(f, attributes->getUniform(key, size_t(baseOf[f])));
    } else {
        out.kind = 4;
        out.count = unsigned(faces * 4);
        out.values = new float[faces * 4 * n];
        for (size_t f = 0; f < faces; ++f)
            for (int c = 0; c < 4; ++c) put(f * 4 + size_t(c), pc.at(key, baseOf[f], int(f), c < int(pc.nt) ? c : int(pc.nt) - 1));
    }
}
void polyAttribute(const PolyCorners& pc, const std::vector<int>& baseOf, const X2SubdAttribute& a, X2SubdAttributeOut& out) {
    if (a.type == 4) {  // an int: constant or uniform alone (the caller's rule)
        const shading::TypedAttributeKey<int> key(a.name);
        const shading::Attributes* attributes = pc.m.getAttributes();
        if (!attributes->isSupported(key)) return;
        if (attributes->getRate(key) == shading::RATE_CONSTANT) {
            out.kind = 1;
            out.count = 1;
            out.values = new float[1];
            out.values[0] = float(attributes->getConstant(key));
        } else if (attributes->getRate(key) == shading::RATE_UNIFORM) {
            out.kind = 2;
            out.count = unsigned(baseOf.size());
            out.values = new float[baseOf.size()];
            for (size_t f = 0; f < baseOf.size(); ++f) out.values[f] = float(attributes->getUniform(key, size_t(baseOf[f])));
        } else {
            throw std::runtime_error(std::string("the int attribute ") + a.name + " varies over the faces");
        }
    } else if (a.type == 1) polyTyped<Vec2f>(pc, baseOf, a, out);
    else if (a.type == 2) polyTyped<Vec3f>(pc, baseOf, a, out);
    else if (a.type == 3) polyTyped<scene_rdl2::math::Color>(pc, baseOf, a, out);
    else polyTyped<float>(pc, baseOf, a, out);
}

extern "C" __declspec(dllexport) int x2PolyTessellate(const X2PolyInput* pin, X2PolyOutput* out, char* error, unsigned errorSize) {
    const char* stage = "the standard attributes";
    try {
        prepareMoonRay();
        std::memset(out, 0, sizeof(*out));
        const X2SubdInput* in = &pin->mesh;
        const auto* rdlGeometry = static_cast<const scene_rdl2::rdl2::Geometry*>(in->geometry);
        const auto* layer = static_cast<const scene_rdl2::rdl2::Layer*>(in->layer);
        stage = "the polygon mesh";
        geom::PolygonMesh::FaceVertexCount faceVertexCount(in->faceVertexCount, in->faceVertexCount + in->faceCount);
        geom::PolygonMesh::IndexBuffer indices(in->indices, in->indices + in->indexCount);
        if (indices.empty() || faceVertexCount.empty()) {
            fail(error, errorSize, "an empty polygon mesh");
            return 0;
        }
        const size_t partCount = in->faceToPart ? in->partCount : 0;
        geom::PolygonMesh::FaceToPartBuffer faceToPart(in->faceCount, 0);
        if (in->faceToPart)
            for (unsigned f = 0; f < in->faceCount; ++f) faceToPart[f] = geom::Primitive::IndexType(in->faceToPart[f]);
        const geom::RateCounts rates{partCount, in->faceCount, in->vertexCount, in->vertexCount, in->indexCount};

        // getVertexData: the time steps' vertices, uvs and normals; processArbitraryData's attributes.
        const unsigned steps = pin->stepCount ? pin->stepCount : 1u;
        geom::PolygonMesh::VertexBuffer vertices(in->vertexCount, steps);
        for (unsigned t = 0; t < steps; ++t)
            for (unsigned i = 0; i < in->vertexCount; ++i) {
                const float* p = t == 0 ? in->vertices + 3 * size_t(i) : pin->stepVertices + (size_t(t - 1) * in->vertexCount + i) * 3;
                vertices(i, t) = Vec3fa(p[0], p[1], p[2], 0.f);
            }
        shading::PrimitiveAttributeTable table;
        if (in->uvCount) {
            const shading::AttributeRate rate(geom::pickRate(rdlGeometry, "uv_list", in->uvCount, rates));
            std::vector<Vec2f> uv(in->uvCount);
            for (unsigned i = 0; i < in->uvCount; ++i) uv[i] = Vec2f(in->uvs[2 * i], in->uvs[2 * i + 1]);
            table.addAttribute(shading::StandardAttributes::sSt, rate, std::move(uv));
        }
        if (in->normalCount) {
            const shading::AttributeRate rate(geom::pickRate(rdlGeometry, "normal_list", in->normalCount, rates));
            std::vector<Vec3f> n(in->normalCount);
            for (unsigned i = 0; i < in->normalCount; ++i) n[i] = Vec3f(in->normals[3 * i], in->normals[3 * i + 1], in->normals[3 * i + 2]);
            table.addAttribute(shading::StandardAttributes::sNormal, rate, std::move(n));
        }
        for (unsigned k = 0; k < in->attributeCount; ++k) addAttribute(table, in->attributes[k]);

        // removeUnassignedFaces: the faces with a material or a volume shader stay, in order.
        std::vector<int> ids(in->assignmentIds, in->assignmentIds + in->faceCount);
        std::vector<int> keptFaces;
        for (unsigned f = 0; f < in->faceCount; ++f) {
            const int id = ids[f];
            if (id != -1 && (layer->lookupMaterial(id) != nullptr || layer->lookupVolumeShader(id) != nullptr)) keptFaces.push_back(int(f));
        }
        geom::LayerAssignmentId assignment(std::move(ids));
        geom::removeUnassignedFaces(layer, assignment, faceToPart, faceVertexCount, indices, &table);
        if (faceVertexCount.empty() || indices.empty()) {
            fail(error, errorSize, "no face of the polygon mesh has a material");
            return 0;
        }
        if (keptFaces.size() != faceVertexCount.size()) {
            fail(error, errorSize, "removeUnassignedFaces kept other faces");
            return 0;
        }
        if (in->leftHanded) {
            size_t offset = 0;
            for (size_t f = 0; f < faceVertexCount.size(); ++f) {
                std::reverse(indices.begin() + offset, indices.begin() + offset + faceVertexCount[f]);
                offset += faceVertexCount[f];
            }
            table.reverseFaceVaryingAttributes(faceVertexCount);
        }
        const geom::PolygonMesh::FaceVertexCount keptCounts = faceVertexCount;
        stage = "createPolygonMesh";
        std::unique_ptr<geom::PolygonMesh> mesh = geom::createPolygonMesh(std::move(faceVertexCount), std::move(indices), std::move(vertices),
                                                                          std::move(assignment), std::move(table));
        if (!mesh) {
            fail(error, errorSize, "MoonRay made no polygon mesh");
            return 0;
        }
        mesh->setMeshResolution(in->meshResolution);
        mesh->setAdaptiveError(in->adaptiveError);
        mesh->setName("xpu2");
        mesh->setIsSingleSided(in->singleSided != 0);
        mesh->setIsNormalReversed(in->reverseNormals != 0);
        mesh->setParts(partCount, std::move(faceToPart));
        mesh->setSmoothNormal(pin->smoothNormal != 0);
        mesh->setModifiability(geom::Primitive::Modifiability::DEFORMABLE);

        // PolygonMesh::transformPrimitive's setTransform alone: the vertices are in render space at their time steps.
        stage = "the transform";
        auto* impl = static_cast<geom::internal::PolyMesh*>(geom::internal::PrimitivePrivateAccess::getPrimitiveImpl(mesh.get()));
        impl->setTransform(shading::XformSamples(steps, scene_rdl2::math::Xform3f(scene_rdl2::math::one)), 0.0f, 0.0f);
        impl->setRdlGeometry(rdlGeometry);
        if (in->reference) impl->setIsReference(true);

        std::vector<mcrt_common::Frustum> frustums;
        std::vector<mcrt_common::Fishtum> fishtums;
        cameraFrustums(in, frustums, fishtums);
        scene_rdl2::math::Mat4d world2render;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) world2render[i][j] = in->world2render[4 * i + j];
        const bool displacing = in->enableDisplacement != 0 && (in->displace != nullptr || in->displaceWith != nullptr);
        std::unique_lock<std::mutex> displaceLock(displaceMutex, std::defer_lock);
        if (displacing) {
            displaceLock.lock();
            displaceThrough(in, layer);
        }
        const geom::internal::TessellationParams params(layer, frustums, fishtums, world2render, displacing,
                                                        /* fastGeomUpdate */ false, /* isBaking */ false, nullptr);
        geom::internal::TessellationStats stats;
        stage = "tessellate";
        const auto tessellating = std::chrono::steady_clock::now();
        impl->tessellate(params, stats);
        const auto packing = std::chrono::steady_clock::now();
        // The buffers below, element by element and each independent: over TBB's threads.
        auto forEach = [](size_t n, const auto& f) {
            tbb::parallel_for(tbb::blocked_range<size_t>(0, n, 4096), [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i < r.end(); ++i) f(i);
            });
        };
        struct Timing {
            std::chrono::steady_clock::time_point tessellating, packing;
            ~Timing() {
                const auto now = std::chrono::steady_clock::now();
                std::fprintf(stderr, "xpu2_geom: polygon tessellation %.2f s (MoonRay's), its buffers %.2f s\n",
                             std::chrono::duration<double>(packing - tessellating).count(), std::chrono::duration<double>(now - packing).count());
            }
        } timing{tessellating, packing};

        stage = "the tessellated buffers";
        const bool quadBase = impl->getBaseFaceType() == geom::internal::MeshIndexType::QUAD;
        const bool tessellated = impl->mIsTessellated;
        const size_t nb = quadBase ? 4 : 3, nt = tessellated ? 4 : nb;
        const size_t faces = impl->mIndices.size() / nt;
        const size_t baseFaces = (tessellated ? impl->mBaseIndices.size() : impl->mIndices.size()) / nb;
        // Each base face's input face (splitNGons: (n - 1) / 2 quads an n-gon, or n - 2 triangles).
        std::vector<int> baseToInput;
        for (size_t k = 0; k < keptFaces.size(); ++k) {
            const size_t n = keptCounts[k], made = quadBase ? (n - 1) >> 1 : n - 2;
            baseToInput.insert(baseToInput.end(), made, keptFaces[k]);
        }
        if (baseToInput.size() != baseFaces) throw std::runtime_error("the base faces do not follow the input faces");
        std::vector<int> baseOf(faces);
        for (size_t f = 0; f < faces; ++f) {
            baseOf[f] = tessellated ? impl->mTessellatedToBaseFace[f] : int(f);
            if (baseOf[f] < 0 || size_t(baseOf[f]) >= baseFaces) throw std::runtime_error("a face of no base face");
        }
        const size_t vertexCount = impl->mVertices.size();
        const unsigned vsteps = unsigned(impl->mVertices.get_time_steps());
        out->vertexCount = unsigned(vertexCount);
        out->faceCount = unsigned(faces);
        out->stepCount = vsteps;
        out->quadBase = quadBase ? 1 : 0;
        out->tessellated = tessellated ? 1 : 0;
        out->vertices = new float[3 * vertexCount * vsteps];
        for (unsigned t = 0; t < vsteps; ++t)
            forEach(vertexCount, [&](size_t i) {
                const Vec3fa p = impl->mVertices(i, t);
                float* o = out->vertices + 3 * (size_t(t) * vertexCount + i);
                o[0] = p.x, o[1] = p.y, o[2] = p.z;
            });
        out->indices = new unsigned[4 * faces];
        out->controlFace = new int[faces];
        forEach(faces, [&](size_t f) {
            for (size_t c = 0; c < 4; ++c) out->indices[4 * f + c] = impl->mIndices[nt * f + (c < nt ? c : nt - 1)];
            out->controlFace[f] = baseToInput[size_t(baseOf[f])];
        });
        const PolyCorners pc{*impl, quadBase, tessellated, nb, nt};
        const shading::Attributes* attributes = impl->getAttributes();
        // getQuadST and getTriangleST: surface_st, else st, else uv, else the face's default corners.
        const shading::TypedAttributeKey<Vec2f>* stKey = nullptr;
        for (const auto* key : {&shading::StandardAttributes::sSurfaceST, &shading::StandardAttributes::sSt, &shading::StandardAttributes::sUv})
            if (attributes->isSupported(*key)) {
                stKey = key;
                break;
            }
        static const float quadCorners[8] = {0, 0, 1, 0, 1, 1, 0, 1}, triangleCorners[8] = {0, 0, 1, 0, 1, 1, 1, 1};
        out->st = new float[8 * faces];
        forEach(faces, [&](size_t f) {
            for (int c = 0; c < 4; ++c) {
                float* o = out->st + 8 * f + 2 * size_t(c);
                if (stKey) {
                    const Vec2f s = pc.at(*stKey, baseOf[f], int(f), c < int(nt) ? c : int(nt) - 1);
                    o[0] = s.x, o[1] = s.y;
                } else {
                    const float* dflt = nt == 4 ? quadCorners : triangleCorners;
                    o[0] = dflt[2 * c], o[1] = dflt[2 * c + 1];
                }
            }
        });
        // The shading normals: a vertex's (displacement's smooth normals, at up to two time steps), else a corner's.
        const auto& normalKey = shading::StandardAttributes::sNormal;
        if (attributes->isSupported(normalKey)) {
            if (attributes->getRate(normalKey) == shading::RATE_VERTEX) {
                out->normalKind = 3;
                out->normalSteps = unsigned(std::max<size_t>(1, std::min<size_t>(attributes->getTimeSampleCount(normalKey), 2)));
                out->normals = new float[3 * vertexCount * out->normalSteps];
                for (unsigned t = 0; t < out->normalSteps; ++t)
                    forEach(vertexCount, [&](size_t i) {
                        const Vec3f n = attributes->getVertex(normalKey, i, int(t));
                        float* o = out->normals + 3 * (size_t(t) * vertexCount + i);
                        o[0] = n.x, o[1] = n.y, o[2] = n.z;
                    });
            } else {
                out->normalKind = 4;
                out->normalSteps = 1;
                out->normals = new float[12 * faces];
                forEach(faces, [&](size_t f) {
                    for (int c = 0; c < 4; ++c) {
                        const Vec3f n = pc.at(normalKey, baseOf[f], int(f), c < int(nt) ? c : int(nt) - 1);
                        float* o = out->normals + 12 * f + 3 * size_t(c);
                        o[0] = n.x, o[1] = n.y, o[2] = n.z;
                    }
                });
            }
        }
        stage = "the tessellated attributes";
        if (in->attributeCount) {
            out->attributes = new X2SubdAttributeOut[in->attributeCount]();
            out->attributeCount = in->attributeCount;
            for (unsigned k = 0; k < in->attributeCount; ++k) polyAttribute(pc, baseOf, in->attributes[k], out->attributes[k]);
        }
        return 1;
    } catch (const std::exception& e) {
        fail(error, errorSize, std::string("in ") + stage + ": " + e.what());
    } catch (...) {
        fail(error, errorSize, std::string("in ") + stage + ": an unknown exception");
    }
    return 0;
}

extern "C" __declspec(dllexport) void x2PolyFree(X2PolyOutput* out) {
    delete[] out->vertices;
    delete[] out->normals;
    delete[] out->indices;
    delete[] out->controlFace;
    delete[] out->st;
    for (unsigned k = 0; k < out->attributeCount; ++k) delete[] out->attributes[k].values;
    delete[] out->attributes;
    std::memset(out, 0, sizeof(*out));
}

extern "C" __declspec(dllexport) void x2SubdFree(X2SubdOutput* out) {
    delete[] out->vertices;
    delete[] out->normals;
    delete[] out->st;
    delete[] out->dpds;
    delete[] out->dpdt;
    delete[] out->indices;
    delete[] out->controlFace;
    for (unsigned k = 0; k < out->attributeCount; ++k) delete[] out->attributes[k].values;
    delete[] out->attributes;
    delete[] out->stepVertices;
    delete[] out->stepNormals;
    delete[] out->stepDpds;
    delete[] out->stepDpdt;
    std::memset(out, 0, sizeof(*out));
}
