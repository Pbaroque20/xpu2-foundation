// The C interface of xpu2_geom.dll (xpu2_geom.cc): MoonRay's own subdivision surface tessellation (its
// OpenSubdivMesh, through its geometry library) for XPU2's scene preparation, built apart with MoonRay's compile flags
// so XPU2's own build never sees MoonRay's internal headers. Positions and vectors are in MoonRay's render space.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// A displacement's value at a tessellated vertex (OpenSubdivMesh::displaceMesh's shading::displace): the
// scene_rdl2::rdl2::Displacement its assignment has, and the vertex's state in render space (P, N, dPds, dPdt, st and
// the texture footprint dSdx, dSdy, dTdx, dTdy); out, the displacement in render space.
typedef void (*X2SubdDisplaceFn)(void* user, const void* displacement, const float* P, const float* N, const float* dPds,
                                 const float* dPdt, const float* st, const float* footprint, float* out);

// A primitive attribute of the control mesh (UserData, as processArbitraryData adds it): its key, type (0 float,
// 1 Vec2f, 2 Vec3f, 3 Rgb, 4 Int), MoonRay rate (shading::AttributeRate: 1 constant, 2 uniform, 3 varying, 4 face
// varying, 5 vertex) and values (count of them, floats; ints as floats).
typedef struct X2SubdAttribute {
    const char* name;
    int type, rate;
    unsigned count;
    const float* values;
} X2SubdAttribute;
// displace with the control mesh's primitive attributes at the vertex as the displacement's state holds them (MoonRay's
// fillDisplacementAttributes, the displacement asking for every one): attributes and attributeCount the input's,
// values three floats each in their order, the first NaN where the vertex has none.
typedef void (*X2SubdDisplaceAttrFn)(void* user, const void* displacement, const float* P, const float* N, const float* dPds,
                                     const float* dPdt, const float* st, const float* footprint, const X2SubdAttribute* attributes,
                                     unsigned attributeCount, const float* values, float* out);
// The attribute over the tessellation, as OpenSubdivMesh interpolates it at a hit: kind 1 one value, 2 one a
// tessellated quad (its control face's), 3 one a tessellated vertex, 4 four a quad (its corners').
typedef struct X2SubdAttributeOut {
    int kind;
    unsigned count;  // values (each its type's floats)
    float* values;
} X2SubdAttributeOut;

// A subdivision mesh as RdlMeshGeometry's createSubdMesh makes it (its attributes as the rdl object holds them).
typedef struct X2SubdInput {
    const float* vertices;  // render space, three floats a vertex
    unsigned vertexCount;
    const int* faceVertexCount;
    unsigned faceCount;
    const int* indices;
    unsigned indexCount;
    const float* uvs;  // uv_list: two floats each (vertexCount or indexCount of them, or none)
    unsigned uvCount;
    const float* normals;  // normal_list, render space: three floats each (as the uvs)
    unsigned normalCount;
    const int* creaseIndices;
    unsigned creaseIndexCount;
    const float* creaseSharpnesses;
    unsigned creaseSharpnessCount;
    const int* cornerIndices;
    unsigned cornerIndexCount;
    const float* cornerSharpnesses;
    unsigned cornerSharpnessCount;
    const int* faceToPart;  // per face (none: all part 0)
    unsigned partCount;
    const int* assignmentIds;  // per face layer assignment id (LayerAssignmentId)
    int scheme, boundary, fvarLinear, meshResolution;
    float adaptiveError;
    int singleSided, reverseNormals, leftHanded;
    const void* layer;     // the scene's scene_rdl2::rdl2::Layer
    const void* geometry;  // its scene_rdl2::rdl2::Geometry (named in MoonRay's messages)
    // The camera frustums MoonRay tessellates adaptively by (mcrt_common::Frustum: camera to screen, the six clip
    // planes, the viewport), none for a camera without one; a fisheye camera's fishtum.
    unsigned frustumCount;
    const float* frustumC2S;      // 16 floats a frustum
    const float* frustumPlanes;   // 24 floats a frustum
    const int* frustumViewports;  // 4 ints a frustum
    int hasFishtum;
    float fishtum[9];  // radial scale, derivative scale, mapping, aperture width, height, region x min, y min, x max, y max
    double world2render[16];
    // Displacement (enable_displacement and a displacement assigned): MoonRay's displaceMesh, each vertex's value
    // from displace (with displaceUser).
    int enableDisplacement;
    X2SubdDisplaceFn displace;
    void* displaceUser;
    const X2SubdAttribute* attributes;
    unsigned attributeCount;
    // A shared primitive (an instancer's reference): its vertices in its own space, displaced as if it sat at the
    // origin (Primitive::setIsReference).
    int reference;
    // In displace's place where given: displace with the primitive attributes (X2SubdDisplaceAttrFn).
    X2SubdDisplaceAttrFn displaceWith;
    // A mesh that deforms over the shutter (x2SubdTessellate only): its time steps, the first in vertices, the rest
    // here in the same space; tessellated once, every step on the one topology, as OpenSubdivMesh does under motion.
    unsigned stepCount;         // 0 or 1 for a still mesh
    const float* stepVertices;  // (stepCount - 1) * vertexCount * 3
} X2SubdInput;

// The tessellated mesh: quads (four indices each) over its vertices, each vertex's limit surface normal, texture
// coordinates and position derivatives (OpenSubdivMesh's mSurfaceNormal, mSurfaceSt, mSurfaceDpds, mSurfaceDpdt), each
// quad's control face.
typedef struct X2SubdOutput {
    unsigned vertexCount, quadCount;
    float* vertices;  // three floats a vertex
    float* normals;
    float* st;  // two floats a vertex
    float* dpds;
    float* dpdt;
    unsigned* indices;
    int* controlFace;
    X2SubdAttributeOut* attributes;  // one an input attribute
    unsigned attributeCount;
    // The later time steps of a deforming mesh (X2SubdInput's stepCount): (stepCount - 1) * vertexCount * 3 each.
    unsigned stepCount;
    float* stepVertices;
    float* stepNormals;
    float* stepDpds;
    float* stepDpdt;
} X2SubdOutput;

// An OpenVDB grid as MoonRay's VdbVolume reads it (findNthGridByName; a float grid, or a Vec3 grid for emission and
// velocity), as a NanoVDB grid for XPU2's samplers: its bytes, its index to grid world (the geometry's own space)
// transform (a linear map's matrix, MoonRay's row-vector Mat4), whether that map is linear and its voxels uniform,
// its active voxels' index bounding box (inclusive) and its voxel size.
typedef struct X2VdbGrid {
    void* data;
    unsigned long long bytes;
    int vec3;
    double indexToWorld[16];
    int linear, uniform;
    int bboxMin[3], bboxMax[3];
    double voxelSize[3];
    // A float grid's leaf nodes and active tiles as leaf-sized cells (each one's origin, three ints; blockSize voxels
    // a side): the volume MoonRay's VolumeRayIntersector marches (velocity: the padded density's).
    int* blocks;
    unsigned blockCount, blockSize;
} X2VdbGrid;
typedef int (*X2VdbReadFn)(const char* path, const char* gridName, int vec3, X2VdbGrid* out, char* error, unsigned errorSize);
typedef void (*X2VdbFreeFn)(X2VdbGrid* grid);
// An emission grid's histogram (computeEmissionDistributionImpl): the box of its active voxels whose luminance is above
// MoonRay's epsilon, from pMin (index) at res voxels, each voxel's luminance times the shader's emission luminance
// (tiles as MoonRay multiplies them along), and the grid's index to grid world transform.
typedef struct X2VdbHistogram {
    int pMin[3], res[3];
    float* values;  // res[0] * res[1] * res[2], x fastest
    double indexToWorld[16];
} X2VdbHistogram;
typedef int (*X2VdbHistogramFn)(const char* path, const char* gridName, float shaderLuminance, X2VdbHistogram* out, char* error,
                                unsigned errorSize);
typedef void (*X2VdbHistogramFreeFn)(X2VdbHistogram* histogram);
// The histogram with a volume shader's emission luminance at each voxel (computeEmissionDistributionImpl's
// evalVolumeShader): shader(user, p), p the voxel's grid world point, in MoonRay's order (a tile's value carries the
// product along).
typedef float (*X2VdbShaderFn)(void* user, const double* p);
typedef int (*X2VdbHistogramMappedFn)(const char* path, const char* gridName, X2VdbShaderFn shader, void* user, X2VdbHistogram* out,
                                      char* error, unsigned errorSize);
// A VDB geometry's velocity under motion blur (VdbVolume::initializePhase1 and 2, initMotionBlurBoundary): the
// velocity grid times velocity_scale, padded where its voxels carry the density grid's along the shutter (the
// density's topology grown, the velocities set there by setValueOnMax and compMax, dilated by one), decimated by
// velocity_sample_rate (below one) with a box resampling; the density grid's active box after the padding. Returns 2
// when the file has no such velocity grid (MoonRay then renders it still).
typedef struct X2VdbVelocity {
    X2VdbGrid velocity;
    int densityMin[3], densityMax[3];
} X2VdbVelocity;
typedef int (*X2VdbVelocityFn)(const char* path, const char* velocityName, const char* densityName, float scale, float sampleRate,
                               float tShutterOpen, float tShutterClose, X2VdbVelocity* out, char* error, unsigned errorSize);

typedef int (*X2SubdTessellateFn)(const X2SubdInput* in, X2SubdOutput* out, char* error, unsigned errorSize);
typedef void (*X2SubdFreeFn)(X2SubdOutput* out);

// A polygon mesh as RdlMeshGeometry's createPolyMesh makes it (mesh's subdivision fields unused): its smooth_normal
// and its time steps (the first in mesh.vertices, the rest here, render space), tessellated and displaced by MoonRay's
// PolyMesh (the QuadMesh or TriMesh PolygonMesh picks, after removeUnassignedFaces).
typedef struct X2PolyInput {
    X2SubdInput mesh;
    int smoothNormal;
    unsigned stepCount;         // time steps, 1 for a still mesh
    const float* stepVertices;  // (stepCount - 1) * vertexCount * 3
} X2PolyInput;
// The mesh as it is hit: faces of four indices (a three-cornered face's fourth its third), each one's input face,
// whether its base faces are quads and whether it is tessellated, its vertices at each time step, its shading normals
// (normalKind 3: one a vertex at each of normalSteps time steps; 4: four a face; 0: none), each face corner's st
// (getQuadST, getTriangleST) and the input attributes (X2SubdAttributeOut's kinds: 1 one value, 2 one a face, its base
// face's, 4 four a face; 0 not kept).
typedef struct X2PolyOutput {
    unsigned vertexCount, faceCount, stepCount;
    int quadBase, tessellated, normalKind;
    unsigned normalSteps;
    float* vertices;  // stepCount * vertexCount * 3
    float* normals;
    unsigned* indices;  // four a face
    int* controlFace;
    float* st;  // eight a face
    X2SubdAttributeOut* attributes;
    unsigned attributeCount;
} X2PolyOutput;
typedef int (*X2PolyTessellateFn)(const X2PolyInput* in, X2PolyOutput* out, char* error, unsigned errorSize);
typedef void (*X2PolyFreeFn)(X2PolyOutput* out);

#ifdef __cplusplus
}
#endif
