// Private XPU2 shading library, shared CPU/CUDA: MoonRay's maps (moonray and
// moonshine dso/map) as nodes of a material's map program (shade_maps.h). Each
// map's parameters are its attributes in the order of its X-list below, so the
// host writes them by name and the devices read them by field. Derived from
// MoonRay Apache-2.0, Copyright 2023-2024 DreamWorks Animation LLC. See
// MOONRAY_LICENSE.txt.
#pragma once
#include "shade_noise.h"
#include "shade_maps.h"
#include "shade_colorcorrect.h"
#include "texture_core.h"
namespace xpu2 { namespace interactive { namespace shade {

// Attribute types, as the parameter block stores them (a bool or int is a float).
enum MapAttrType : U32 { MapAttrBool, MapAttrInt, MapAttrFloat, MapAttrRgb, MapAttrVec2f, MapAttrVec3f, MapAttrVec4f };
#define X2_MAP_FIELD_Bool(n) float n;
#define X2_MAP_FIELD_Int(n) float n;
#define X2_MAP_FIELD_Float(n) float n;
#define X2_MAP_FIELD_Rgb(n) Color n;
#define X2_MAP_FIELD_Vec2f(n) float n[2];
#define X2_MAP_FIELD_Vec3f(n) V3 n;
#define X2_MAP_FIELD_Vec4f(n) float n[4];
#define X2_MAP_FIELD(T, n) X2_MAP_FIELD_##T(n)

// Each map: X(kind, class, list). A list names its attributes as F(type, name).
#define X2_MAP_CONSTANT_COLOR(F) F(Rgb, color_value)
#define X2_MAP_CONSTANT_SCALAR(F) F(Float, scalar_value)
#define X2_MAP_CHECKERBOARD(F) \
    F(Rgb, color_A) F(Rgb, color_B) F(Int, num_u_tiles) F(Int, num_v_tiles) F(Int, texture_coordinates) F(Vec3f, input_texture_coordinates)
#define X2_MAP_CLAMP(F) F(Rgb, input) F(Bool, clamp) F(Float, clamp_min) F(Float, clamp_max)
#define X2_MAP_FLOAT_TO_RGB(F) F(Float, R) F(Float, G) F(Float, B)
#define X2_MAP_RGB_TO_FLOAT(F) F(Rgb, input) F(Int, mode)
#define X2_MAP_HSV_TO_RGB(F) F(Rgb, input)
#define X2_MAP_RGB_TO_HSV(F) F(Rgb, input)
#define X2_MAP_REMAP(F)                                                                                                                \
    F(Int, remap_method) F(Rgb, input) F(Float, input_min) F(Float, input_max) F(Float, output_min) F(Float, output_max)              \
    F(Float, midpoint_bias) F(Bool, clamp) F(Float, clamp_min) F(Float, clamp_max) F(Rgb, input_min_RGB) F(Rgb, input_max_RGB)        \
    F(Rgb, output_min_RGB) F(Rgb, output_max_RGB) F(Rgb, midpoint_bias_RGB) F(Bool, clamp_RGB) F(Rgb, clamp_min_RGB) F(Rgb, clamp_max_RGB)
#define X2_MAP_BLEND(F) \
    F(Rgb, color_A) F(Rgb, color_B) F(Int, blend_type) F(Float, blend_amount) F(Float, threshold_min) F(Float, threshold_max)
#define X2_MAP_COLOR_CORRECT(F)                                                                                                       \
    F(Rgb, input) F(Bool, on) F(Float, mix) F(Bool, clamp) F(Float, clamp_min) F(Float, clamp_max) F(Bool, hue_shift_enabled)         \
    F(Float, hue_shift) F(Bool, saturation_enabled) F(Float, saturation) F(Bool, contrast_enabled) F(Float, contrast)                 \
    F(Bool, gamma_enabled) F(Float, gamma) F(Bool, gain_offset_enabled) F(Float, gain) F(Float, offset) F(Bool, TMI_enabled)          \
    F(Rgb, TMI) F(Bool, use_per_channel_saturation) F(Float, saturation_r) F(Float, saturation_g) F(Float, saturation_b)             \
    F(Bool, use_per_channel_contrast) F(Float, contrast_r) F(Float, contrast_g) F(Float, contrast_b) F(Bool, use_per_channel_gamma)  \
    F(Float, gamma_r) F(Float, gamma_g) F(Float, gamma_b) F(Bool, use_per_channel_gain_offset) F(Float, gain_r) F(Float, gain_g)      \
    F(Float, gain_b) F(Float, offset_r) F(Float, offset_g) F(Float, offset_b)
#define X2_MAP_CC_CONTRAST(F) \
    F(Rgb, input) F(Bool, on) F(Float, mix) F(Float, contrast) F(Bool, use_per_channel_contrast) F(Float, contrast_r) F(Float, contrast_g) F(Float, contrast_b)
#define X2_MAP_CC_GAIN_OFFSET(F)                                                                                                     \
    F(Rgb, input) F(Bool, on) F(Float, mix) F(Float, gain) F(Float, offset) F(Bool, use_per_channel_gain_offset) F(Float, gain_r)     \
    F(Float, gain_g) F(Float, gain_b) F(Float, offset_r) F(Float, offset_g) F(Float, offset_b)
#define X2_MAP_CC_GAMMA(F) \
    F(Rgb, input) F(Bool, on) F(Float, mix) F(Float, gamma) F(Bool, use_per_channel_gamma) F(Float, gamma_r) F(Float, gamma_g) F(Float, gamma_b)
#define X2_MAP_CC_HSV(F)                                                                                                             \
    F(Rgb, input) F(Bool, on) F(Float, hue_shift) F(Float, saturation_factor) F(Float, saturation_contrast) F(Float, saturation_shift) \
    F(Float, value_factor) F(Float, value_contrast) F(Float, value_shift) F(Bool, clamp)
#define X2_MAP_CC_HUE_SHIFT(F) F(Rgb, input) F(Bool, on) F(Float, mix) F(Float, hue_shift)
#define X2_MAP_CC_SATURATION(F)                                                                                                      \
    F(Rgb, input) F(Bool, on) F(Float, mix) F(Float, saturation) F(Bool, use_per_channel_saturation) F(Float, saturation_r)           \
    F(Float, saturation_g) F(Float, saturation_b)
#define X2_MAP_CC_TMI(F) F(Rgb, input) F(Bool, on) F(Float, mix) F(Rgb, TMI)
#define X2_MAP_UV_TRANSFORM(F)     F(Int, space) F(Vec2f, offset) F(Vec2f, scale) F(Float, rotation_angle) F(Vec2f, rotation_center) F(Vec3f, rotation_axis)
#define X2_MAP_CC_LEGACY(F)                                                                                                         F(Rgb, input) F(Bool, on) F(Float, mask) F(Float, brightness) F(Float, contrast) F(Float, hue) F(Float, saturation) F(Bool, invert)     F(Int, monochrome) F(Rgb, multiplier) F(Bool, clamp)
#define X2_MAP_IMAGE(F)                                                                                                              \
    F(Bool, use_default_color_when_missing) F(Rgb, default_color) F(Int, texture_coordinates) F(Bool, alpha_only)                     \
    F(Vec3f, input_texture_coordinates) F(Float, mip_bias) F(Vec2f, offset) F(Vec2f, scale) F(Float, rotation_angle)                 \
    F(Vec2f, rotation_center) F(Bool, saturation_enabled) F(Rgb, saturation) F(Bool, contrast_enabled) F(Rgb, contrast)             \
    F(Bool, gamma_enabled) F(Rgb, gamma_adjust) F(Bool, gain_offset_enabled) F(Rgb, gain) F(Rgb, offset_adjust)                       \
    F(Bool, TMI_control_enabled) F(Vec3f, TMI)

#define X2_MAP_IMAGE_NORMAL(F) \
    F(Bool, use_default_value_when_missing) F(Vec3f, default_value) F(Bool, wrap_around) F(Int, normal_encoding) \
    F(Int, texture_coordinates) F(Vec3f, input_texture_coordinates) F(Vec2f, offset) F(Vec2f, scale) F(Float, rotation_angle) \
    F(Vec2f, rotation_center)

// The kit's maps (native-port/maps).
#define X2_MAP_MODO_TEXTURE(F) \
    F(Int, mode) F(Int, blend) F(Rgb, background) F(Rgb, foreground) F(Float, opacity) F(Float, mask) F(Rgb, normal) \
    F(Float, normal_strength) F(Float, height) F(Float, bump_strength) F(Float, bump_step) F(Bool, flip_green) F(Vec2f, scale) \
    F(Int, octaves) F(Float, lacunarity) F(Float, persistence) F(Bool, use_coordinates) F(Rgb, coordinates) F(Float, distance) \
    F(Int, tile_u) F(Int, tile_v) F(Int, component) F(Vec4f, uv_affine) F(Vec2f, uv_offset)
#define X2_MAP_MODO_NORMAL(F) F(Vec3f, input)

#define X2_MAP_OP(F) \
    F(Int, operation) F(Rgb, op1) F(Rgb, op2) F(Float, op1_factor) F(Float, op2_factor) F(Bool, clamp) F(Float, tolerance)
#define X2_MAP_GRADIENT(F) \
    F(Rgb, color_A) F(Rgb, color_B) F(Vec3f, start) F(Vec3f, end) F(Int, falloff_type) F(Float, falloff_exponent) \
    F(Float, falloff_start) F(Float, falloff_end) F(Float, falloff_end_intensity) F(Float, falloff_bias) F(Int, space) \
    F(Bool, symmetric) F(Float, symmetric_center)
#define X2_MAP_RAMP(F) \
    F(Int, ramp_type) F(Float, input) F(Int, space) F(Int, texture_coordinates) F(Vec3f, input_texture_coordinates) \
    F(Int, wrap_type) F(Vec2f, uv_wave) F(Vec2f, uv_repeat)

#define X2_MAP_ATTRIBUTE(F) \
    F(Int, map_type) F(Rgb, default_value) F(Bool, warn_when_unavailable) F(Int, primitive_attribute_type) F(Rgb, color)

#define X2_MAP_TWO_SIDED(F) F(Rgb, front_color) F(Rgb, back_color)
#define X2_MAP_OP_SQRT(F) F(Rgb, input)
#define X2_MAP_LOD(F) F(Int, mode) F(Float, start) F(Float, stop) F(Rgb, near_value) F(Rgb, far_value)
#define X2_MAP_NORMAL_TO_RGB(F) F(Int, input_register)

#define X2_MAP_USD_UV_TEXTURE(F) \
    F(Int, output_mode) F(Vec2f, st) F(Rgb, fallback) F(Rgb, scale) F(Rgb, bias)
#define X2_MAP_USD_TRANSFORM2D(F) F(Rgb, in) F(Float, rotation) F(Vec2f, scale) F(Vec2f, translation)
// The USD primvar readers, one node for all: what it reads (UsdPrimvarReader's attribute map type: 0 a
// primitive attribute, 1 P, 2 st, 5 dPds, 6 dPdt), the attribute's key and components, and the fallback.
#define X2_MAP_USD_PRIMVAR(F) F(Int, mode) F(Int, key) F(Int, components) F(Vec3f, fallback)

// The projections (moonshine ProjectPlanarMap, ProjectSphericalMap, ProjectCylindricalMap): after the
// attributes, the projector's world-to-object and its normal transform (3x4 each).
#define X2_MAP_PROJECT_PLANAR(F) F(Bool, project_on_back_faces) F(Bool, black_outside_projection) F(Bool, use_reference_space)
#define X2_MAP_PROJECT_SPHERICAL(F) F(Bool, use_reference_space)
#define X2_MAP_PROJECT_CYLINDRICAL(F) \
    F(Bool, use_reference_space) F(Bool, project_on_inward_surfaces) F(Bool, project_on_outward_surfaces) F(Bool, black_outside_projection)

// NoiseMap_v2 (moonshine): after the attributes, its transform (3x4), its object's world-to-object (3x4), the
// offsets of its noises' tables (R, G, B, distortion; 0 for none), then the tables.
#define X2_MAP_NOISE(F)                                                                                                         \
    F(Int, seed) F(Float, frequency_multiplier) F(Float, amplitude) F(Float, max_level) F(Float, lacunarity) F(Float, persistence) \
    F(Float, distortion) F(Int, distortion_noise_type) F(Bool, color) F(Rgb, color_A) F(Rgb, color_B) F(Int, noise_type)          \
    F(Float, flow_angle) F(Float, flow_advection_rate) F(Bool, use_4D_noise) F(Float, time) F(Float, bias) F(Float, gain)         \
    F(Bool, invert) F(Bool, use_smoothstep) F(Vec2f, smoothstep) F(Int, space) F(Vec3f, input_texture_coordinates)                \
    F(Vec3f, translation) F(Vec3f, rotation) F(Vec3f, scale) F(Int, rotation_order) F(Int, transformation_order)

// NoiseWorleyMap_v2 and _v3 (moonshine): after the attributes, the transform (3x4), the object's world-to-object
// (3x4), the Poisson point-count bounds (low[10], high[10]), the tables' offset, then the tables.
#define X2_MAP_WORLEY(F)                                                                                                      \
    F(Int, seed) F(Float, frequency) F(Float, jitter) F(Float, max_level) F(Rgb, color_A) F(Rgb, color_B) F(Int, output_mode)    \
    F(Int, distance_method) F(Float, minkowski_number) F(Int, space) F(Vec3f, input_texture_coordinates) F(Float, point_size)    \
    F(Vec2f, remap) F(Bool, use_smoothstep) F(Vec2f, smoothstep) F(Float, bias) F(Float, gain) F(Bool, invert)                  \
    F(Vec3f, translation) F(Vec3f, scale) F(Vec3f, rotation) F(Int, rotation_order) F(Int, transformation_order) F(Float, F1)   \
    F(Float, F2) F(Float, F3) F(Float, F4) F(Int, cell_id)


// More maps (shade_mapnodes_more.h). Prepared data after the attributes where a comment says.
#define X2_MAP_AXIS_ANGLE(F) \
    F(Vec3f, input_vector) F(Int, input_space) F(Vec3f, rotation_axis) F(Int, axis_space) F(Float, angle) F(Int, output_space)
#define X2_MAP_TRANSFORM_NORMAL(F) F(Vec3f, input_normal) F(Int, transform) F(Bool, decode_input_normal)
#define X2_MAP_RGB_TO_LAB(F) F(Rgb, input_color)
#define X2_MAP_RANDOM(F) F(Rgb, input) F(Int, seed) F(Bool, monochrome) F(Float, output_min) F(Float, output_max)  // then 2048 colours
#define X2_MAP_MULTI_CHANNEL(F) F(Rgb, input) F(Int, mode) F(Float, tolerance)
#define X2_MAP_LAYER(F) F(Rgb, input_A) F(Rgb, input_B) F(Float, mask) F(Int, mode)
#define X2_MAP_HAIR(F) F(Rgb, base_color) F(Rgb, tip_color) F(Float, bias) F(Float, gain) F(Rgb, column_uv_color)
#define X2_MAP_HAIR_COLUMN(F) F(Float, unused)
#define X2_MAP_HAIR_PRESETS(F) F(Int, color)
#define X2_MAP_TOON(F)                                                                                                               \
    F(Int, mode) F(Rgb, fill_color) F(Rgb, outline_color) F(Rgb, crease_color) F(Float, outline_threshold) F(Float, outline_scale) \
    F(Float, crease_threshold) F(Float, crease_scale)
#define X2_MAP_CURVATURE(F) F(Bool, invert) F(Float, power) F(Float, scale) F(Int, mode)
#define X2_MAP_WIREFRAME(F) F(Rgb, line_color) F(Rgb, fill_color) F(Bool, raster) F(Float, line_width)
#define X2_MAP_DEFORMATION(F) F(Int, output_mode) F(Bool, use_warning_color) F(Rgb, warning_color)  // then the fatal colour
#define X2_MAP_DIRECTIONAL(F)                                                                                                       \
    F(Int, prime_direction) F(Int, polarity) F(Rgb, color_a) F(Rgb, color_b) F(Int, object_axis) F(Vec3f, custom_direction)       \
    F(Bool, use_reference_space) F(Float, input_normal_dial) F(Int, clamping_behavior) F(Float, bias) F(Bool, use_smoothstep)     \
    F(Float, smoothstep_start) F(Float, smoothstep_end) F(Int, falloff_type)  // then world-to-object (12), the input normal's register
#define X2_MAP_TRANSFORM_SPACE(F)                                                                                                   \
    F(Vec3f, input) F(Int, input_type) F(Int, from_space) F(Int, to_space) F(Bool, concatenate_instance_level_transforms)          \
    F(Bool, use_custom_window_coordinates) F(Float, window_x_min) F(Float, window_x_max) F(Float, window_y_min) F(Float, window_y_max)
#define X2_MAP_TRIPLANAR_UDIM(F)                                                                                                    \
    F(Bool, use_correct_uv) F(Bool, use_reference_space) F(Vec2f, x_offset) F(Vec2f, x_scale) F(Float, x_rotation)                \
    F(Vec2f, x_rotation_center) F(Vec2f, y_offset) F(Vec2f, y_scale) F(Float, y_rotation) F(Vec2f, y_rotation_center)            \
    F(Vec2f, z_offset) F(Vec2f, z_scale) F(Float, z_rotation) F(Vec2f, z_rotation_center)
#define X2_MAP_EXTRA_AOV(F) F(Rgb, color)
#define X2_MAP_LIST(F) F(Float, unused)
#define X2_MAP_DEBUG(F) \
    F(Bool, checkerboard) F(Int, map_type) F(Vec3f, input_normal) F(Int, input_normal_space) F(Float, input_normal_dial) F(Int, primitive_attribute_type)
#define X2_MAP_DISTORT_NORMAL(F)                                                                                                    \
    F(Int, noise_space) F(Vec3f, input_texture_coordinates) F(Bool, use_input_vectors) F(Rgb, input_U) F(Rgb, input_V) F(Int, seed) \
    F(Vec3f, frequency_U) F(Vec3f, frequency_V) F(Float, amplitude_U) F(Float, amplitude_V)
#define X2_MAP_RANDOM_NORMAL(F) F(Rgb, input) F(Int, seed)  // then 2048 floats

#define X2_MAPS(X)                                                                   \
    X(MapConstantColor, "ConstantColorMap", X2_MAP_CONSTANT_COLOR)                   \
    X(MapConstantScalar, "ConstantScalarMap", X2_MAP_CONSTANT_SCALAR)                \
    X(MapCheckerboard, "CheckerboardMap", X2_MAP_CHECKERBOARD)                       \
    X(MapClamp, "ClampMap", X2_MAP_CLAMP)                                            \
    X(MapFloatToRgb, "FloatToRgbMap", X2_MAP_FLOAT_TO_RGB)                           \
    X(MapRgbToFloat, "RgbToFloatMap", X2_MAP_RGB_TO_FLOAT)                           \
    X(MapHsvToRgb, "HsvToRgbMap", X2_MAP_HSV_TO_RGB)                                 \
    X(MapRgbToHsv, "RgbToHsvMap", X2_MAP_RGB_TO_HSV)                                 \
    X(MapRemap, "RemapMap", X2_MAP_REMAP)                                            \
    X(MapBlend, "BlendMap", X2_MAP_BLEND)                                            \
    X(MapColorCorrect, "ColorCorrectMap", X2_MAP_COLOR_CORRECT)                      \
    X(MapCCContrast, "ColorCorrectContrastMap", X2_MAP_CC_CONTRAST)                  \
    X(MapCCGainOffset, "ColorCorrectGainOffsetMap", X2_MAP_CC_GAIN_OFFSET)           \
    X(MapCCGamma, "ColorCorrectGammaMap", X2_MAP_CC_GAMMA)                           \
    X(MapCCHsv, "ColorCorrectHsvMap", X2_MAP_CC_HSV)                                 \
    X(MapCCHueShift, "ColorCorrectHueShiftMap", X2_MAP_CC_HUE_SHIFT)                 \
    X(MapCCSaturation, "ColorCorrectSaturationMap", X2_MAP_CC_SATURATION)            \
    X(MapCCTMI, "ColorCorrectTMIMap", X2_MAP_CC_TMI)                                 \
    X(MapImage, "ImageMap", X2_MAP_IMAGE) \
    X(MapImageNormal, "ImageNormalMap", X2_MAP_IMAGE_NORMAL) \
    X(MapModoTexture, "ModoTextureMap", X2_MAP_MODO_TEXTURE) \
    X(MapModoNormal, "ModoNormalMap", X2_MAP_MODO_NORMAL) \
    X(MapOp, "OpMap", X2_MAP_OP) \
    X(MapGradient, "GradientMap", X2_MAP_GRADIENT) \
    X(MapRamp, "RampMap", X2_MAP_RAMP) \
    X(MapAttribute, "AttributeMap", X2_MAP_ATTRIBUTE) \
    X(MapTwoSided, "TwoSidedMap", X2_MAP_TWO_SIDED) \
    X(MapOpSqrt, "OpSqrtMap", X2_MAP_OP_SQRT) \
    X(MapLOD, "LODMap", X2_MAP_LOD) \
    X(MapNormalToRgb, "NormalToRgbMap", X2_MAP_NORMAL_TO_RGB) \
    X(MapUsdUVTexture, "UsdUVTexture", X2_MAP_USD_UV_TEXTURE) \
    X(MapUsdTransform2d, "UsdTransform2d", X2_MAP_USD_TRANSFORM2D) \
    X(MapUsdPrimvar, "UsdPrimvarReader", X2_MAP_USD_PRIMVAR) \
    X(MapProjectPlanar, "ProjectPlanarMap", X2_MAP_PROJECT_PLANAR) \
    X(MapProjectSpherical, "ProjectSphericalMap", X2_MAP_PROJECT_SPHERICAL) \
    X(MapProjectCylindrical, "ProjectCylindricalMap", X2_MAP_PROJECT_CYLINDRICAL) \
    X(MapNoise, "NoiseMap_v2", X2_MAP_NOISE) \
    X(MapWorley2, "NoiseWorleyMap_v2", X2_MAP_WORLEY) \
    X(MapWorley3, "NoiseWorleyMap_v3", X2_MAP_WORLEY)     X(MapColorCorrectLegacy, "ColorCorrectLegacyMap", X2_MAP_CC_LEGACY)     X(MapUVTransform, "UVTransformMap", X2_MAP_UV_TRANSFORM) \
    X(MapAxisAngle, "AxisAngleMap", X2_MAP_AXIS_ANGLE) \
    X(MapTransformNormal, "TransformNormalMap", X2_MAP_TRANSFORM_NORMAL) \
    X(MapRgbToLab, "RgbToLabMap", X2_MAP_RGB_TO_LAB) \
    X(MapRandom, "RandomMap", X2_MAP_RANDOM) \
    X(MapMultiChannelToFloat, "MultiChannelToFloatMap", X2_MAP_MULTI_CHANNEL) \
    X(MapLayer, "LayerMap", X2_MAP_LAYER) \
    X(MapLayer2, "LayerMap_v2", X2_MAP_LAYER) \
    X(MapHair, "HairMap", X2_MAP_HAIR) \
    X(MapHairColumn, "HairColumnMap", X2_MAP_HAIR_COLUMN) \
    X(MapHairColorPresets, "HairColorPresetsMap", X2_MAP_HAIR_PRESETS) \
    X(MapToon, "ToonMap", X2_MAP_TOON) \
    X(MapCurvature, "CurvatureMap", X2_MAP_CURVATURE) \
    X(MapWireframe, "WireframeMap", X2_MAP_WIREFRAME) \
    X(MapDeformation, "DeformationMap", X2_MAP_DEFORMATION) \
    X(MapDirectional, "DirectionalMap", X2_MAP_DIRECTIONAL) \
    X(MapTransformSpace, "TransformSpaceMap", X2_MAP_TRANSFORM_SPACE) \
    X(MapProjectTriplanarUdim, "ProjectTriplanarUdimMap", X2_MAP_TRIPLANAR_UDIM) \
    X(MapExtraAov, "ExtraAovMap", X2_MAP_EXTRA_AOV) \
    X(MapList, "ListMap", X2_MAP_LIST) \
    X(MapDebug, "DebugMap", X2_MAP_DEBUG) \
    X(MapDistortNormal, "DistortNormalMap", X2_MAP_DISTORT_NORMAL) \
    X(MapRandomNormal, "RandomNormalMap", X2_MAP_RANDOM_NORMAL)

#define X2_MAP_KIND(kind, cls, list) kind,
enum MapKind : U32 { X2_MAPS(X2_MAP_KIND) MapSwitchColor, MapSwitchFloat, MapProjectTriplanar, MapProjectCamera, MapRgbToNormal, MapSwitchNormal,
                     MapCombineNormal, MapProjectTriplanarNormal, MapProjectPlanarNormal,
                     MapProjectCameraNormal, MapKindCount };
#undef X2_MAP_KIND
#define X2_MAP_STRUCT(kind, cls, list) struct kind##Params { list(X2_MAP_FIELD) };
X2_MAPS(X2_MAP_STRUCT)
#undef X2_MAP_STRUCT
// SwitchColorMap and SwitchFloatMap: a choice and 64 inputs (multi attributes input0..input63).
struct MapSwitchColorParams {
    float choice;
    Color input[64];
};
struct MapSwitchFloatParams {
    float choice;
    float input[64];
};
// ProjectTriplanarMap_v2, prepared on the host (moonshine projection::updateTriplanarMap): the projector's
// world-to-object and normal transform, the transition width power, each cube face's texture, and the textures
// (one, three or six): the scene's texture index (-1: it did not load, MoonRay's fatal colour), whether active,
// wrapped and linearised, and its 2D transform (Xform3f rows vx, vy, vz, p).
struct MapTriplanarTexture {
    float texture, active, wrap, gamma;
    float xform[12];
};
// ProjectCameraMap_v2, prepared on the host: world to the projector camera's screen (4x4, row vectors, its window
// from the texture's or a custom aspect ratio), the world normal into its camera space, and its switches. Its
// texture is the header's (clamped).
struct MapProjectCameraParams {
    float worldToScreen[16], normalToCamera[12];
    float projectOnBackFaces, blackOutsideProjection, normalEncoding, pad;  // ProjectCameraNormalMap: always black outside
};
// ProjectPlanarNormalMap, prepared on the host: the projector's world-to-object and normal transform, the
// transition width power (of 0.5), the normal encoding and wrap. Its texture is the header's (read linear).
struct MapProjectPlanarNormalParams {
    float toObject[12], normalToObject[12];
    float transitionWidthPower, normalEncoding, wrap, pad;
};
struct MapTriplanarParams {
    float toObject[12], normalToObject[12];
    float transitionWidthPower, normalEncoding, pad0[2];  // ProjectTriplanarNormalMap_v2: 0 [0, 1], 1 [-1, 1]
    float indices[6], pad1[2];
    MapTriplanarTexture textures[6];
};

// A map in the pool: this header, its parameters, then its bindings.
struct MapNodeHeader {
    U32 kind, reg, paramCount, bindingCount;
    U32 texture;           // ImageMap: the scene's texture index (MapNoTexture for none)
    U32 wrap, gamma;  // ImageMap: wrap_around, and whether to linearise (gamma on, or auto on 8-bit data)
    U32 object;       // 1 when the map's 'object' gives its object space: world-to-object after its prepared data
    U32 udim;         // ImageMap: a UDIM set's tile count (its tiles' texture indices after the attributes), 0 for none
};
constexpr U32 MapNoTexture = 0xffffffffu;

// A map's parameters as it reads them: its own values, and its bound attributes' maps' results.
struct MapParams {
    const float* p;
    const MapBinding* b;
    U32 nb;
    const Color* regs;
    X2_HD const MapBinding* bound(U32 offset) const {
        for (U32 i = 0; i < nb; ++i)
            if (b[i].offset == offset) return &b[i];
        return nullptr;
    }
    X2_HD float f(U32 o) const {
        float w[3] = {p[o], 0.0f, 0.0f};
        if (const MapBinding* x = bound(o)) applyRule(w, BindFloat, regs[x->reg]);
        return w[0];
    }
    X2_HD Color c(U32 o) const {
        float w[3] = {p[o], p[o + 1], p[o + 2]};
        if (const MapBinding* x = bound(o)) applyRule(w, x->rule, regs[x->reg]);
        return Color{w[0], w[1], w[2]};
    }
    X2_HD void v2(U32 o, float out[2]) const {
        float w[2] = {p[o], p[o + 1]};
        if (const MapBinding* x = bound(o)) {
            const Color r = regs[x->reg];
            w[0] = r.x;
            w[1] = r.y;
        }
        out[0] = w[0];
        out[1] = w[1];
    }
};
#define X2_OFF(Params, field) unsigned(X2_OFFSETOF(Params, field) / sizeof(float))

// The scene's textures as a map reads them: CPU and GPU give their own (tex::lookup with a sink).
// Status Tex::lookup(U32 texture, const tex::Binding& binding, tex::Footprint f, V3& out)

//----------------------------------------------------------------------------
// The maps' sample functions.

// State::getdPdx and getdPdy: the position's screen derivatives as the texture derivatives give them (so scaled as
// they are), not the ray differentials.
X2_HD inline V3 statedPdx(const State& st) { return st.dPds * st.dsdx + st.dPdt * st.dtdx; }
X2_HD inline V3 statedPdy(const State& st) { return st.dPds * st.dsdy + st.dPdt * st.dtdy; }
// getTextureOptionIndex: trilinear isotropic for displacement, anisotropic on primary paths, trilinear closest texels
// on mirror and glossy ones, one level's closest texel on diffuse ones.
X2_HD inline U32 textureMode(const State& st) {
    if (st.displacement) return 3u;
    return st.pathType == PathIndirectDiffuse ? 2u : st.pathType == PathPrimary ? 0u : 1u;
}
X2_HD inline float colorAverage(Color c) { return (c.x + c.y + c.z) / 3.0f; }
X2_HD inline bool isEqualColor(Color a, Color b) { return isEqual(a.x, b.x) && isEqual(a.y, b.y) && isEqual(a.z, b.z); }
X2_HD inline Color mixWithInput(Color input, Color sample, float mix) {
    if (!isEqual(mix, 1.0f)) {
        sample.x = lerpOpt(input.x, sample.x, mix);
        sample.y = lerpOpt(input.y, sample.y, mix);
        sample.z = lerpOpt(input.z, sample.z, mix);
    }
    return sample;
}
X2_HD inline Color remapRGB(Color input, Color inMin, Color inMax, Color outMin, Color outMax, Color b) {
    const Color inRange = inMax - inMin, outRange = outMax - outMin;
    if (isBlack(outRange)) return outMin;
    if (isBlack(inRange)) {
        return Color{!isEqual(b.x, 0.5f) ? biasf(0.5f, b.x) : 0.5f, !isEqual(b.y, 0.5f) ? biasf(0.5f, b.y) : 0.5f,
                     !isEqual(b.z, 0.5f) ? biasf(0.5f, b.z) : 0.5f};
    }
    Color r{(input.x - inMin.x) / inRange.x, (input.y - inMin.y) / inRange.y, (input.z - inMin.z) / inRange.z};
    r = Color{biasf(r.x, b.x), biasf(r.y, b.y), biasf(r.z, b.z)};
    return r * outRange + outMin;
}
X2_HD inline Color remapUniform(Color input, float inMin, float inMax, float outMin, float outMax, bool applyBias, float b) {
    const float inRange = inMax - inMin, outRange = outMax - outMin;
    if (isZero(outRange)) return rgb(outMin);
    if (isZero(inRange)) {
        float t = 0.5f;
        if (applyBias) t = biasf(t, b);
        return rgb(t * outRange + outMin);
    }
    Color r = (input - rgb(inMin)) * (1.0f / inRange);
    if (applyBias) r = Color{biasf(r.x, b), biasf(r.y, b), biasf(r.z, b)};
    return r * outRange + rgb(outMin);
}
X2_HD inline void hsvContrast(float a, float& channel) {
    if (a < 0) {
        channel = 0.5f * a + channel * (1.f + a);
    } else if (a > 0) {
        if (isOne(a)) a = 0.999f;
        a = 1.f / (1.f - a);
        channel = a * (channel - 0.5f) + 0.5f;
    }
}
X2_HD inline Color clampColor(Color c, float lo, float hi) { return Color{clampf(c.x, lo, hi), clampf(c.y, lo, hi), clampf(c.z, lo, hi)}; }

// ImageMap's rotateTexCoords: st and its derivatives about the rotation centre.
X2_HD inline void rotateTexCoords(float theta, const float centre[2], float st[2], float& dsdx, float& dsdy, float& dtdx, float& dtdy) {
    const float c = cosf(theta), s = sinf(theta);
    // st3 * R, with R = [c -s 0; s c 0; 0 0 1] (a row vector times the matrix)
    const float x = st[0] - centre[0], y = st[1] - centre[1];
    st[0] = x * c + y * s + centre[0];
    st[1] = -x * s + y * c + centre[1];
    // d * transpose(R)
    const float sx = dsdx * c - dsdy * s, sy = dsdx * s + dsdy * c;
    const float tx = dtdx * c - dtdy * s, ty = dtdx * s + dtdy * c;
    dsdx = sx;
    dsdy = sy;
    dtdx = tx;
    dtdy = ty;
}

template <class Tex>
X2_CALL U32 sampleImageMap(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    using Q = MapImageParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    if (h.texture == MapNoTexture && !h.udim) {
        out = q.use_default_color_when_missing != 0 ? q.default_color : Color{1.0f, 0.0f, 1.0f};  // MoonRay's fatal colour
        return tex::Ready;
    }
    float s[2];
    float dsdx, dtdx, dsdy, dtdy;
    const int coordinates = int(q.texture_coordinates);
    if (coordinates == 2) {  // input texture coordinates
        const Color uvw = P.c(X2_OFF(Q, input_texture_coordinates));
        s[0] = uvw.x;
        s[1] = uvw.y;
        dsdx = dsdy = dtdx = dtdy = 0.0f;
        if (uvw.z < 0.f) {
            out = rgb(0);
            return tex::Ready;
        }
    } else {
        s[0] = st.s;
        s[1] = st.t;
        dsdx = st.dsdx;
        dtdx = st.dtdx;
        dsdy = st.dsdy;
        dtdy = st.dtdy;
    }
    const float mipBias = 1.0f + P.f(X2_OFF(Q, mip_bias));
    float scale[2], offset[2];
    P.v2(X2_OFF(Q, scale), scale);
    P.v2(X2_OFF(Q, offset), offset);
    U32 texture = h.texture;
    if (h.udim) {
        // UdimTexture: the tile 1001 + int(s) + 10 int(t) at the fractional st; outside 0 <= s <= 10, t >= 0, or a
        // missing tile, the default colour when asked for, else MoonRay's fatal colour.
        const Color fatal{1.0f, 0.0f, 1.0f};
        if (s[0] < 0.f || s[0] > 10.f || s[1] < 0.f) {
            out = fatal;
            return tex::Ready;
        }
        const int udim = int(s[0]) + int(s[1]) * 10;
        s[0] = s[0] - float(int(s[0]));
        s[1] = s[1] - float(int(s[1]));
        const float tile = U32(udim) < h.udim ? P.p[sizeof(Q) / sizeof(float) + udim] : -1.0f;
        texture = tile < 0.0f ? MapNoTexture : U32(tile);
        if (texture == MapNoTexture) {
            out = q.use_default_color_when_missing != 0 ? q.default_color : fatal;
            return tex::Ready;
        }
        // Its derivatives are not scaled (lookup scales them: undone here).
        const float isx = scale[0] != 0.0f ? 1.0f / scale[0] : 0.0f, isy = scale[1] != 0.0f ? 1.0f / scale[1] : 0.0f;
        dsdx *= isx;
        dtdx *= isx;
        dsdy *= isy;
        dtdy *= isy;
    }
    const float theta = deg2rad(q.rotation_angle);
    if (!isZero(theta)) rotateTexCoords(theta, q.rotation_center, s, dsdx, dsdy, dtdx, dtdy);
    // tex::lookup applies scale, offset, the T flip and the derivative scaling as ImageMap does.
    tex::Binding b{nullptr, scale[0], scale[1], offset[0], offset[1], mipBias - 1.0f, h.wrap, h.gamma, textureMode(st)};
    const tex::Footprint f{s[0], s[1], dsdx, dtdx, dsdy, dtdy};
    V3 texel;
    const U32 status = tex.lookup(texture, b, f, texel);
    if (status != tex::Ready) return status;
    // alpha_only: the texture admitted as its alpha (host_texture.h's (a, a, a): the file's fourth channel, or the
    // fill MOONRAY_MODO_OPAQUE_ALPHA picks where it has none).
    Color result = Color{texel.x, texel.y, texel.z};
    if (q.saturation_enabled != 0) applySaturation(q.saturation.x, q.saturation.y, q.saturation.z, result.x, result.y, result.z);
    if (q.contrast_enabled != 0) applyNukeContrast(q.contrast.x, q.contrast.y, q.contrast.z, result.x, result.y, result.z);
    if (q.gamma_enabled != 0) {
        const float e = 0.000001f;
        applyGamma(1.0f / fmaxf(e, q.gamma_adjust.x), 1.0f / fmaxf(e, q.gamma_adjust.y), 1.0f / fmaxf(e, q.gamma_adjust.z), result.x,
                   result.y, result.z);
    }
    if (q.gain_offset_enabled != 0)
        applyGainAndOffset(q.gain.x, q.gain.y, q.gain.z, q.offset_adjust.x, q.offset_adjust.y, q.offset_adjust.z, result.x, result.y,
                           result.z);
    if (q.TMI_control_enabled != 0) applyTMI(q.TMI.x, q.TMI.y, q.TMI.z, result.x, result.y, result.z);
    out = result;
    return tex::Ready;
}

// ImageNormalMap's sampleNormal: a render-space normal from a tangent-space texture.
template <class Tex>
X2_CALL U32 sampleImageNormalMap(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    using Q = MapImageNormalParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    if (h.texture == MapNoTexture) {
        out = q.use_default_value_when_missing != 0 ? q.default_value : Color{1.0f, 0.0f, 1.0f};
        return tex::Ready;
    }
    float s[2];
    float dsdx, dtdx, dsdy, dtdy;
    if (int(q.texture_coordinates) == 1) {
        const Color uvw = P.c(X2_OFF(Q, input_texture_coordinates));
        s[0] = uvw.x;
        s[1] = uvw.y;
        dsdx = dsdy = dtdx = dtdy = 0.0f;
        if (uvw.z < 0.f) {
            out = rgb(0);
            return tex::Ready;
        }
    } else {
        s[0] = st.s;
        s[1] = st.t;
        dsdx = st.dsdx;
        dtdx = st.dtdx;
        dsdy = st.dsdy;
        dtdy = st.dtdy;
    }
    if (q.wrap_around == 0 && (s[0] < 0.0f || s[0] > 1.0f || s[1] < 0.0f || s[1] > 1.0f)) {
        out = st.N;
        return tex::Ready;
    }
    float scale[2], offset[2];
    P.v2(X2_OFF(Q, offset), offset);
    P.v2(X2_OFF(Q, scale), scale);
    const float theta = deg2rad(q.rotation_angle);
    if (!isZero(theta)) rotateTexCoords(theta, q.rotation_center, s, dsdx, dsdy, dtdx, dtdy);
    tex::Binding b{nullptr, scale[0], scale[1], offset[0], offset[1], 0.0f, h.wrap, 0u, textureMode(st)};
    const tex::Footprint f{s[0], s[1], dsdx, dtdx, dsdy, dtdy};
    V3 texel;
    const U32 status = tex.lookup(h.texture, b, f, texel);
    if (status != tex::Ready) return status;
    V3 normal = texel;
    if (!finite3(normal) || (isZero(normal.x) && isZero(normal.y) && isZero(normal.z))) {
        out = st.N;
        return tex::Ready;
    }
    if (int(q.normal_encoding) == 0) normal = 2.0f * normal - V3{1.0f, 1.0f, 1.0f};
    out = localToGlobal(frame(st.N, normalize(st.dPds)), normal);
    return tex::Ready;
}

// ModoTextureMap (native-port/maps/ModoTexture): Modo's texture layers. Repeat=0, Edge=1, Mirror=2, Reset=3.
X2_HD inline float modoTileCoordinate(float value, int mode) {
    if (mode == 0) return value - floorf(value);
    if (mode == 2) {
        const float period = value - 2 * floorf(value / 2);
        return period <= 1 ? period : 2 - period;
    }
    return clampf(value, 0.0f, 1.0f);
}
X2_HD inline float modoLayerGain(float value, float amount) {
    const float x = clampf(value, 0.0f, 1.0f), g = clampf(amount, .00001f, .99999f);
    const float y = x < .5f ? 2 * x : 2 - 2 * x;
    const float b = y / ((1 / (1 - g) - 2) * (1 - y) + 1);
    return x < .5f ? .5f * b : 1 - .5f * b;
}
X2_HD inline float modoHash2(int x, int y) {
    U32 h = U32(x) * 73856093u ^ U32(y) * 19349663u;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return float(h & 0xffffffu) / 16777215.0f;
}
X2_HD inline float modoNoise2(float x, float y) {
    const int ix = int(floorf(x)), iy = int(floorf(y));
    float u = x - ix, v = y - iy;
    u = u * u * (3 - 2 * u);
    v = v * v * (3 - 2 * v);
    return (1 - v) * ((1 - u) * modoHash2(ix, iy) + u * modoHash2(ix + 1, iy)) + v * ((1 - u) * modoHash2(ix, iy + 1) + u * modoHash2(ix + 1, iy + 1));
}
X2_HD inline Color modoBlend(int blend, Color a, Color b) {
    auto ch = [&](int c, float A, float B) -> float {
        switch (blend) {
        case 1: return A * B;
        case 2: return A + B;
        case 3: return A - B;
        case 4: return 1.0f - (1.0f - A) * (1.0f - B);
        case 5: return A / fmaxf(1.e-6f, B);
        case 6: return fabsf(A - B);
        case 7: return fminf(A, B);
        case 8: return fmaxf(A, B);
        case 9: return A < .5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B);
        case 10: return B < .5f ? 2 * A * B : 1 - 2 * (1 - A) * (1 - B);
        case 11: return A + B - 2 * A * B;
        case 12: return B <= .5f ? A - (1 - 2 * B) * A * (1 - A)
                                 : A + (2 * B - 1) * ((A <= .25f ? ((16 * A - 12) * A + 4) * A : sqrtf(fmaxf(0.f, A))) - A);
        case 13: return B >= 1 ? 1 : fminf(1.f, A / fmaxf(1.e-6f, 1 - B));
        case 14: return B <= 0 ? 0 : 1 - fminf(1.f, (1 - A) / fmaxf(1.e-6f, B));
        default: return B;
        }
        (void)c;
    };
    return Color{ch(0, a.x, b.x), ch(1, a.y, b.y), ch(2, a.z, b.z)};
}
// The bump mode's heights at st offset by (+e, 0), (-e, 0), (0, +e), (0, -e), computed by the caller.
struct ModoBumpHeights {
    float v[4];
};
X2_HD inline bool modoNeedsBump(const MapNodeHeader& h, const float* p) {
    return h.kind == MapModoTexture && int(reinterpret_cast<const MapModoTextureParams*>(p)->mode) == 1 &&
           reinterpret_cast<const MapModoTextureParams*>(p)->bump_strength != 0;
}
X2_HD inline Color sampleModoTexture(const MapParams& P, const State& st, const ModoBumpHeights* bump) {
    using Q = MapModoTextureParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const int mode = int(q.mode);
    if (mode == 1) {
        const Color encoded = P.c(X2_OFF(Q, normal));
        V3 n{(encoded.x * 2 - 1) * q.normal_strength, (encoded.y * 2 - 1) * q.normal_strength, fmaxf(.001f, encoded.z * 2 - 1)};
        if (q.flip_green != 0) n.y = -n.y;
        float sx = n.x / n.z, sy = n.y / n.z;
        if (q.bump_strength != 0 && bump) {
            const float e = fmaxf(1.e-7f, q.bump_step);
            const float right = bump->v[0], left = bump->v[1], up = bump->v[2], down = bump->v[3];
            const V3 N = st.N;
            V3 T = st.dPds - N * dot(N, st.dPds);
            const float lengthT = length(T);
            if (lengthT > 1.e-9f) {
                T = T / lengthT;
                const V3 B = cross(N, T);
                const float dx = (right - left) / (2 * e * lengthT), dy = (up - down) / (2 * e);
                const float crossT = dot(st.dPdt, T), crossB = dot(st.dPdt, B);
                sx -= q.bump_strength * dx;
                if (fabsf(crossB) > 1.e-9f) sy -= q.bump_strength * (dy - dx * crossT) / crossB;
            }
        }
        n = normalize(V3{sx, sy, 1});
        return n;
    }
    // A named UV set (uv_name): its primitive attribute, moved by st's offset from modo_primary_uv along the set's st
    // derivatives (where the geometry gives them; a subdivision surface does not, and the layer's placement, uv_affine,
    // stands for them), and those derivatives as the set's jacobian. A hit without either attribute gives the neutral
    // result (ModoTextureMap's missing-coordinates event).
    const float* keys = P.p + sizeof(Q) / sizeof(float);
    const bool named = (mode == 12 || mode == 13) && keys[2] != 0.0f;
    float namedUv[2] = {0, 0}, jac[4] = {1, 0, 0, 1};
    if (named) {
        const U32 key = *reinterpret_cast<const U32*>(&keys[0]), baseKey = *reinterpret_cast<const U32*>(&keys[1]);
        float value[3], base[3];
        if (!primValue(st, key, value) || !primValue(st, baseKey, base)) return mode == 13 ? Color{.5f, .5f, 1} : Color{0, 0, 0};
        float ds[3], dt[3];
        if (st.subdivision || !primPartials(st, key, ds, dt)) {
            ds[0] = q.uv_affine[0];
            ds[1] = q.uv_affine[2];
            dt[0] = q.uv_affine[1];
            dt[1] = q.uv_affine[3];
        }
        const float dx = st.s - base[0], dy = st.t - base[1];
        namedUv[0] = value[0] + ds[0] * dx + dt[0] * dy;
        namedUv[1] = value[1] + ds[1] * dx + dt[1] * dy;
        jac[0] = ds[0];
        jac[1] = dt[0];
        jac[2] = ds[1];
        jac[3] = dt[1];
    }
    if (mode == 12) {
        if (named) return Color{namedUv[0], namedUv[1], 0};
        return Color{q.uv_affine[0] * st.s + q.uv_affine[1] * st.t + q.uv_offset[0],
                     q.uv_affine[2] * st.s + q.uv_affine[3] * st.t + q.uv_offset[1], 0};
    }
    if (mode == 13) {
        const float ax = named ? jac[0] : q.uv_affine[0], ay = named ? jac[1] : q.uv_affine[1], az = named ? jac[2] : q.uv_affine[2],
                    aw = named ? jac[3] : q.uv_affine[3];
        const float det = ax * aw - ay * az;
        const V3 N = st.N;
        V3 T = st.dPds - N * dot(N, st.dPds);
        if (fabsf(det) < 1e-12f || length(T) < 1e-9f) return Color{.5f, .5f, 1};
        V3 U = (st.dPds * aw - st.dPdt * az) / det;
        U = U - N * dot(N, U);
        if (length(U) < 1e-9f) return Color{.5f, .5f, 1};
        T = normalize(T);
        U = normalize(U);
        const V3 B = cross(N, T);
        V3 V = cross(N, U);
        const V3 derivativeV = (st.dPds * (-ay) + st.dPdt * ax) / det;
        if (dot(V, derivativeV) < 0) V = V * (-1.f);
        Color encoded = P.c(X2_OFF(Q, foreground));
        const float u = named ? namedUv[0] : ax * st.s + ay * st.t + q.uv_offset[0], v = named ? namedUv[1] : az * st.s + aw * st.t + q.uv_offset[1];
        if (int(q.tile_u) == 2 && u - 2 * floorf(u / 2) > 1) encoded.x = 1 - encoded.x;
        if (int(q.tile_v) == 2 && v - 2 * floorf(v / 2) > 1) encoded.y = 1 - encoded.y;
        const V3 normal = U * (encoded.x * 2 - 1) + V * (encoded.y * 2 - 1) + N * (encoded.z * 2 - 1);
        return Color{dot(normal, T) * .5f + .5f, dot(normal, B) * .5f + .5f, dot(normal, N) * .5f + .5f};
    }
    if (mode == 5 || mode == 6) {
        const Color c = P.c(X2_OFF(Q, coordinates));
        const float x = (q.use_coordinates != 0 ? c.x : st.s) * q.scale[0], y = (q.use_coordinates != 0 ? c.y : st.t) * q.scale[1];
        const int u = int(q.tile_u), v = int(q.tile_v);
        if (mode == 6) {
            const float covered = ((u != 3 || (x >= 0 && x <= 1)) && (v != 3 || (y >= 0 && y <= 1))) ? 1.0f : 0.0f;
            return rgb(covered);
        }
        return Color{modoTileCoordinate(x, u), modoTileCoordinate(y, v), 0};
    }
    const Color a = P.c(X2_OFF(Q, background)), b = P.c(X2_OFF(Q, foreground));
    if (mode == 10) {
        const Color r1 = P.c(X2_OFF(Q, normal)), r2 = P.c(X2_OFF(Q, coordinates));
        return Color{a.x * b.x + a.y * b.y + a.z * b.z, r1.x * b.x + r1.y * b.y + r1.z * b.z, r2.x * b.x + r2.y * b.y + r2.z * b.z};
    }
    if (mode == 9) {
        const float f = sqrtf(clampf((b.x + b.y + b.z) / 3, 0.0f, .99f));
        return rgb((1 + f) / (1 - f));
    }
    if (mode == 8) return Color{modoLayerGain(b.x, q.distance), modoLayerGain(b.y, q.distance), modoLayerGain(b.z, q.distance)};
    if (mode == 7) {
        const int component = int(q.component);
        return rgb(component == 0 ? b.x : (component == 1 ? b.y : b.z));
    }
    if (mode == 11)
        return Color{-logf(clampf(b.x, 1.e-6f, 1.0f)), -logf(clampf(b.y, 1.e-6f, 1.0f)), -logf(clampf(b.z, 1.e-6f, 1.0f))} * fmaxf(0.f, a.x);
    if (mode == 4) {
        const float distance = fmaxf(1.e-9f, q.distance);
        return Color{-logf(clampf(b.x, 1.e-6f, 1.0f)), -logf(clampf(b.y, 1.e-6f, 1.0f)), -logf(clampf(b.z, 1.e-6f, 1.0f))} / distance;
    }
    if (mode >= 2) {
        const Color c = P.c(X2_OFF(Q, coordinates));
        float x = (q.use_coordinates != 0 ? c.x : st.s) * q.scale[0], y = (q.use_coordinates != 0 ? c.y : st.t) * q.scale[1], value = 0;
        if (mode == 2) {
            value = float((int(floorf(x * 2)) + int(floorf(y * 2))) & 1);
        } else {
            float amplitude = 1, total = 0;
            const int octaves = int(q.octaves) < 1 ? 1 : (int(q.octaves) > 12 ? 12 : int(q.octaves));
            for (int i = 0; i < octaves; ++i) {
                value += amplitude * modoNoise2(x, y);
                total += amplitude;
                x *= fmaxf(.01f, q.lacunarity);
                y *= fmaxf(.01f, q.lacunarity);
                amplitude *= clampf(q.persistence, 0.0f, 1.0f);
            }
            value /= total;
        }
        return a * (1 - value) + b * value;
    }
    const Color mixed = modoBlend(int(q.blend), a, b);
    const float alpha = saturate(P.f(X2_OFF(Q, opacity)) * P.f(X2_OFF(Q, mask)));
    return a * (1 - alpha) + mixed * alpha;
}

// Shading spaces (shading/ispc/Xform.ispc transformPoint from render space).
enum ShadingSpace : U32 { SpaceRender = 0, SpaceCamera, SpaceWorld, SpaceScreen, SpaceObject, SpaceReference, SpaceTexture };
// P in a space; objectMatrix: the map's own object (world to object), else the hit geometry's.
X2_HD inline V3 spacePoint(const State& st, U32 space, const float* objectMatrix) {
    switch (space) {
    case SpaceRender:
    case SpaceCamera: return st.worldToRender ? xformPoint(st.worldToRender, st.P) : st.P;  // the active camera's space
    case SpaceWorld: return st.P;
    case SpaceObject:
        if (objectMatrix) return xformPoint(objectMatrix, st.P);
        return st.worldToObject ? xformPoint(st.worldToObject, st.P) : st.P;
    case SpaceTexture: return V3{st.s, st.t, 0.0f};
    default: return st.P;  // screen and reference spaces are refused at admission
    }
}
// GradientMap's falloffs.
X2_HD inline float gradientNaturalFalloff(float x, float power) {
    auto g = [&](float v) { return 1.0f / powf(7.0f * v + 1.0f, power); };
    auto h = [&](float v) { return g(v) - g(1.0f); };
    return h(x) / h(0.0f);
}
X2_HD inline float gradientFalloff(float t, int type, float exponent) {
    float h = t;
    switch (type) {
    case 0: h = 1.0f; break;
    case 1: h = gradientNaturalFalloff(h, exponent); break;
    case 2: h = powf(h, exponent); break;
    case 3: h = h * h; break;
    case 4: h = powf(((cosf((1.0f - h) * Pi) + 1.0f) / 2.0f), exponent); break;
    case 5:
        h = 3.0f * h * h - 2.0f * h * h * h;
        h = powf(h, exponent);
        break;
    default: break;
    }
    return h;
}
X2_HD inline float opAlmostEquals(float a, float b, float e) { return fabsf(a - b) <= e ? 1.0f : 0.0f; }
X2_HD inline float opShift(float a, float b, bool left) {
    const int s = int(b), v = int(a);
    if (s < 0 || s > 31) return 0.0f;  // AVX2 variable shifts give zero past the width
    return float(left ? int(unsigned(v) << s) : (v >> s));
}
template <class F>
X2_HD inline Color opEach(Color a, Color b, F fn) {
    return Color{fn(a.x, b.x), fn(a.y, b.y), fn(a.z, b.z)};
}
X2_HD inline Color sampleOpMap(const MapParams& P) {
    using Q = MapOpParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color op1 = P.c(X2_OFF(Q, op1)), op2 = P.c(X2_OFF(Q, op2));
    const float f1 = P.f(X2_OFF(Q, op1_factor)), f2 = P.f(X2_OFF(Q, op2_factor));
    const Color a = op1 * f1, b = op2 * f2;
    const float tol = q.tolerance;
    Color r = op1;
    switch (int(q.operation)) {
    case 0: r = a + b; break;
    case 1: r = a - b; break;
    case 2: r = a * b; break;
    case 3: r = opEach(a, b, [](float x, float y) { return x / fmaxf(y, 0.00001f); }); break;
    case 4: r = opEach(a, b, [](float x, float y) { return fmaxf(x, y); }); break;
    case 5: r = opEach(a, b, [](float x, float y) { return fminf(x, y); }); break;
    case 6: r = opEach(a, b, [](float x, float y) { return powf(fmaxf(x, Epsilon), y); }); break;
    case 7: r = cross(a, b); break;
    case 8: r = rgb(dot(a, b)); break;
    case 9: r = Color{1.0f - a.x, 1.0f - a.y, 1.0f - a.z}; break;
    case 10: {
        const float l2 = dot(a, a);
        r = l2 <= Epsilon * Epsilon ? rgb(0) : normalize(a);
        break;
    }
    case 11: r = a; break;
    case 12: r = b; break;
    case 13:
        r = clampColor(opEach(a, b, [](float x, float y) { return y < 0.5f ? x * y * 2.0f : 1.0f - (1.0f - x) * (1.0f - y) * 2.0f; }), 0.0f, 1.0f);
        break;
    case 14: r = clampColor(opEach(a, b, [](float x, float y) { return 1.0f - (1.0f - x) * (1.0f - y); }), 0.0f, 1.0f); break;
    case 15: r = Color{fabsf(a.x), fabsf(a.y), fabsf(a.z)}; break;
    case 16: r = Color{ceilf(a.x), ceilf(a.y), ceilf(a.z)}; break;
    case 17: r = Color{floorf(a.x), floorf(a.y), floorf(a.z)}; break;
    case 18: r = opEach(a, b, [](float x, float y) { return fmodf(x, fmaxf(0.00001f, y)); }); break;
    case 19: r = Color{a.x - floorf(a.x), a.y - floorf(a.y), a.z - floorf(a.z)}; break;
    case 20: r = rgb(length(a)); break;
    case 21: r = Color{sinf(a.x), sinf(a.y), sinf(a.z)}; break;
    case 22: r = Color{cosf(a.x), cosf(a.y), cosf(a.z)}; break;
    case 23:
        r = Color{float(int(a.x < 0.0f ? a.x - 0.5f : a.x + 0.5f)), float(int(a.y < 0.0f ? a.y - 0.5f : a.y + 0.5f)),
                  float(int(a.z < 0.0f ? a.z - 0.5f : a.z + 0.5f))};
        break;
    case 24:
        r = Color{dwAcos(clampf(a.x, -1.0f + Epsilon, 1.0f - Epsilon)), dwAcos(clampf(a.y, -1.0f + Epsilon, 1.0f - Epsilon)),
                  dwAcos(clampf(a.z, -1.0f + Epsilon, 1.0f - Epsilon))};
        break;
    case 25: r = opEach(a, b, [](float x, float y) { return float(x < y); }); break;
    case 26: r = opEach(a, b, [](float x, float y) { return float(x <= y); }); break;
    case 27: r = opEach(a, b, [](float x, float y) { return float(x > y); }); break;
    case 28: r = opEach(a, b, [](float x, float y) { return float(x >= y); }); break;
    case 29: r = Color{opAlmostEquals(a.x, b.x, tol), opAlmostEquals(a.y, b.y, tol), opAlmostEquals(a.z, b.z, tol)}; break;
    case 30: r = Color{1.0f - opAlmostEquals(a.x, b.x, tol), 1.0f - opAlmostEquals(a.y, b.y, tol), 1.0f - opAlmostEquals(a.z, b.z, tol)}; break;
    case 31: r = opEach(a, b, [](float x, float y) { return float(x != 0.0f && y != 0.0f); }); break;
    case 32: r = opEach(a, b, [](float x, float y) { return float(x != 0.0f || y != 0.0f); }); break;
    case 33: r = Color{float(a.x == 0.0f), float(a.y == 0.0f), float(a.z == 0.0f)}; break;
    case 34: r = opEach(a, b, [](float x, float y) { return float(int(x) ^ int(y)); }); break;
    case 35: r = opEach(a, b, [](float x, float y) { return opShift(x, y, true); }); break;
    case 36: r = opEach(a, b, [](float x, float y) { return opShift(x, y, false); }); break;
    case 37: r = opEach(a, b, [](float x, float y) { return float(int(x) & int(y)); }); break;
    case 38: r = opEach(a, b, [](float x, float y) { return float(int(x) | int(y)); }); break;
    case 39: r = rgb(opAlmostEquals(length(a - b), 0.0f, tol)); break;
    case 40: r = rgb(1.0f - opAlmostEquals(length(a - b), 0.0f, tol)); break;
    default: r = rgb(0.0f); break;
    }
    if (q.clamp != 0) r = clampColor(r, 0.f, 1.f);
    return r;
}
X2_HD inline const float* mapObjectMatrix(const MapNodeHeader& h, const float* afterParams) { return h.object ? afterParams : nullptr; }
X2_HD inline Color sampleGradientMap(const MapNodeHeader& h, const MapParams& P, const State& st) {
    using Q = MapGradientParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color colorA = P.c(X2_OFF(Q, color_A)), colorB = P.c(X2_OFF(Q, color_B));
    const V3 pos = spacePoint(st, U32(q.space), mapObjectMatrix(h, P.p + sizeof(Q) / sizeof(float)));
    float blend = 0.0f;
    const V3 v = q.end - q.start;
    const float magSq = dot(v, v);
    if (magSq > 0.0f) blend = saturate(dot(pos - q.start, v) / magSq);
    if (blend < q.falloff_start) blend = 0.0f;
    else if (blend > q.falloff_end) blend = 1.0f;
    else blend = saturate((blend - q.falloff_start) / (q.falloff_end - q.falloff_start));
    const int type = int(q.falloff_type);
    if (type != 1) blend = 1.0f - blend;
    blend = 1.0f - gradientFalloff(blend, type, q.falloff_exponent);
    if (q.symmetric != 0) {
        const float centre = clampf(q.symmetric_center, Epsilon, 1.0f - Epsilon);
        blend = blend < centre ? blend / centre : 1.0f - (blend - centre) / (1.0f - centre);
    }
    blend = blend * saturate(q.falloff_end_intensity);
    blend = biasf(blend, q.falloff_bias);
    return lerp(colorA, colorB, blend);
}
// RampMap: its prepared ColorRampControl (sorted, no hue blend adjustment) after its attributes, then
// its object's world-to-object when it has one.
X2_HD inline Color sampleRampMap(const MapNodeHeader& h, const MapParams& P, const State& st) {
    using Q = MapRampParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const ColorRamp& ramp = *reinterpret_cast<const ColorRamp*>(P.p + sizeof(Q) / sizeof(float));
    V3 pos;
    if (int(q.texture_coordinates) == 1) {
        const Color uvw = P.c(X2_OFF(Q, input_texture_coordinates));
        pos = V3{uvw.x, uvw.y, 0.f};
    } else {
        pos = spacePoint(st, U32(q.space), mapObjectMatrix(h, P.p + (sizeof(Q) + sizeof(ColorRamp)) / sizeof(float)));
    }
    float u = pos.x * q.uv_repeat[0], v = pos.y * q.uv_repeat[1];
    if (q.uv_wave[0] > 0.f) u = u + sinf(2.f * Pi * v) * q.uv_wave[0];
    if (q.uv_wave[1] > 0.f) v = v + sinf(2.f * Pi * u) * q.uv_wave[1];
    if (int(q.wrap_type) == 0) {
        if (u > 1.0f || u < 0.0f) u = u - floorf(u);
        if (v > 1.0f || v < 0.0f) v = v - floorf(v);
    } else {
        u = saturate(u);
        v = saturate(v);
    }
    return rampEval2D(ramp, u, v, U32(q.ramp_type), P.f(X2_OFF(Q, input)));
}

// AttributeMap: a primitive attribute (its key after the attributes), or the state's values in
// MoonRay's render space.
X2_HD inline Color sampleAttributeMap(const MapParams& P, const State& st) {
    using Q = MapAttributeParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const U32 key = *reinterpret_cast<const U32*>(P.p + sizeof(Q) / sizeof(float));
    auto renderVector = [&](V3 v) { return st.worldToRender ? xformVector(st.worldToRender, v) : v; };
    switch (int(q.map_type)) {
    case 0: {
        float v[3];
        if (!primValue(st, key, v)) return P.c(X2_OFF(Q, default_value));
        switch (int(q.primitive_attribute_type)) {
        case PrimFloat: return rgb(v[0]);
        case PrimVec2f: return Color{v[0], v[1], 0.0f};
        case PrimVec3f:
        case PrimRgb: return Color{v[0], v[1], v[2]};
        case PrimInt: return rgb(float(int(v[0])));
        default: return P.c(X2_OFF(Q, default_value));
        }
    }
    case 1: return st.worldToRender ? xformPoint(st.worldToRender, st.P) : st.P;
    case 2: return Color{st.s, st.t, 0.0f};
    case 3: return renderVector(st.N);
    case 4: return renderVector(st.Ng);
    case 5: return renderVector(st.dPds);
    case 6: return renderVector(st.dPdt);
    case 9: return P.c(X2_OFF(Q, color));
    case 20: return renderVector(st.wo);
    default: return rgb(0.0f);  // refused at admission
    }
}

// UsdUVTexture: a lookup at its st with no filter footprint (MoonRay sets the derivatives to zero).
template <class Tex>
X2_CALL U32 sampleUsdUVTexture(const MapNodeHeader& h, const MapParams& P, const State& state, const Tex& tex, Color& out) {
    using Q = MapUsdUVTextureParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    if (h.texture == MapNoTexture) {
        out = q.fallback;
        return tex::Ready;
    }
    float st[2];
    P.v2(X2_OFF(Q, st), st);
    tex::Binding b{nullptr, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, h.wrap, h.gamma, textureMode(state)};
    const tex::Footprint f{st[0], st[1], 0.0f, 0.0f, 0.0f, 0.0f};
    V3 texel;
    const U32 status = tex.lookup(h.texture, b, f, texel);
    if (status != tex::Ready) return status;
    Color r;
    switch (int(q.output_mode)) {
    case 1: r = rgb(texel.x); break;
    case 2: r = rgb(texel.y); break;
    case 3: r = rgb(texel.z); break;
    case 4: r = rgb(1.0f); break;  // alpha: the admitted textures are RGB (refused at admission)
    default: r = Color{texel.x, texel.y, texel.z};
    }
    out = r * q.scale + q.bias;
    return tex::Ready;
}
X2_HD inline Color sampleUsdTransform2d(const MapParams& P) {
    using Q = MapUsdTransform2dParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color in = P.c(X2_OFF(Q, in));
    float u = q.scale[0] * in.x, v = q.scale[1] * in.y;
    if (!isZero(q.rotation)) {
        const float theta = deg2rad(q.rotation), s = sinf(theta), c = cosf(theta);
        const float x = u * c - v * s;
        v = u * s + v * c;
        u = x;
    }
    return Color{u + q.translation[0], v + q.translation[1], 0.f};
}
X2_HD inline Color sampleUsdPrimvar(const MapParams& P, const State& st) {
    using Q = MapUsdPrimvarParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    auto renderVector = [&](V3 v) { return st.worldToRender ? xformVector(st.worldToRender, v) : v; };
    switch (int(q.mode)) {
    case 1: return st.worldToRender ? xformPoint(st.worldToRender, st.P) : st.P;
    case 2: return Color{st.s, st.t, 0.0f};
    case 5: return renderVector(st.dPds);
    case 6: return renderVector(st.dPdt);
    default: {
        U32 key;
        const float k = q.key;
        key = *reinterpret_cast<const U32*>(&k);
        float v[3];
        if (primValue(st, key, v)) {
            const int n = int(q.components);
            return Color{v[0], n > 1 ? v[1] : (n == 1 ? v[0] : 0.0f), n > 2 ? v[2] : (n == 1 ? v[0] : 0.0f)};
        }
        const Color fb = P.c(X2_OFF(Q, fallback));
        const int n = int(q.components);
        return n == 1 ? rgb(fb.x) : Color{fb.x, fb.y, n > 2 ? fb.z : 0.0f};
    }
    }
}

// Texture coordinates from a projector, the third component a mask (-1: ImageMap shows black there).
// Without a valid projector, ProjectCylindricalMap's vector code returns black, the others (0, 0, -1).
X2_HD inline Color sampleProjection(const MapNodeHeader& h, const MapParams& P, const State& st, U32 attributeFloats) {
    if (!h.object) return h.kind == MapProjectCylindrical ? rgb(0.0f) : Color{0.0f, 0.0f, -1.0f};
    const float* toObject = P.p + attributeFloats;
    const float* normalToObject = toObject + 12;
    const V3 pos = xformPoint(toObject, st.P);
    if (h.kind == MapProjectPlanar) {
        using Q = MapProjectPlanarParams;
        const Q& q = *reinterpret_cast<const Q*>(P.p);
        // The shading normal in the projector's space: a back face when it points away (z < 0).
        if (q.project_on_back_faces == 0 && xformVector(normalToObject, st.N).z < 0.f) return Color{0.0f, 0.0f, -1.0f};
        Color uvw{pos.x + 0.5f, pos.y + 0.5f, 1.0f};
        if (q.black_outside_projection != 0) uvw.z = (uvw.x < 0.f || uvw.x > 1.f || uvw.y < 0.f || uvw.y > 1.f) ? -1.f : 1.f;
        return uvw;
    }
    if (h.kind == MapProjectSpherical) {
        Color r{0.0f, 0.0f, 0.0f};
        float len = length(pos);
        if (!isZero(len)) r.y = dwAcos(pos.y / len) / Pi;
        len = sqrtf(pos.x * pos.x + pos.z * pos.z);
        if (!isZero(len)) {
            const float theta = 0.5f * dwAcos(pos.x / len) / Pi;
            r.x = pos.z >= 0 ? (1.0f - theta) : theta;
        }
        return r;
    }
    using Q = MapProjectCylindricalParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    if (q.project_on_inward_surfaces == 0 || q.project_on_outward_surfaces == 0) {
        const V3 n = xformVector(normalToObject, normalize(st.N));
        const V3 pc{pos.x, 0.0f, pos.z};
        const float pDotN = dot(pc, st.entering ? n : -n);
        if ((q.project_on_inward_surfaces == 0 && pDotN < 0.0f) || (q.project_on_outward_surfaces == 0 && pDotN > 0.0f))
            return Color{0.0f, 0.0f, -1.0f};
    }
    Color r;
    const float len = sqrtf(pos.x * pos.x + pos.z * pos.z);
    if (isZero(len)) {
        r = pos;
    } else {
        const float theta = 0.5f * dwAcos(pos.x / len) / Pi;
        r = Color{pos.z >= 0 ? (1.0f - theta) : theta, pos.y + 0.5f, 0.0f};
    }
    if (q.black_outside_projection != 0) r.z = (r.x < 0.0f || r.x > 1.0f || r.y < 0.0f || r.y > 1.0f) ? -1.0f : 1.0f;
    return r;
}

// ProjectTriplanarMap_v2's sample (PROJECTION_fillTriplanarTextures): the hit and its differentials in the
// projector's space; per cube face whose axis weighs at least BLEND_EPSILON, the position swizzled onto the face
// through its texture's 2D transform, centred, and looked up as BasicTexture does (no T flip, the derivatives as
// given); the faces the normal looks along blended by the weights. ProjectTriplanarNormalMap_v2
// (PROJECTION_fillTriplanarNormalTextures) reads each face's texel as a tangent-space normal in the frame of the
// projector-space normal and the face's tangent, blends them and their lengths, and returns the blend in render space
// at the blended length (N without a projector).
template <class Tex>
X2_CALL U32 sampleTriplanar(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    const MapTriplanarParams& q = *reinterpret_cast<const MapTriplanarParams*>(P.p);
    const Color fatal{1.0f, 0.0f, 1.0f};
    const bool normalMap = h.kind == MapProjectTriplanarNormal;
    if (!h.object) {
        out = normalMap ? st.N : fatal;
        return tex::Ready;
    }
    const V3 pos = xformPoint(q.toObject, st.P);
    const V3 pdx = xformPoint(q.toObject, st.P + statedPdx(st)) - pos, pdy = xformPoint(q.toObject, st.P + statedPdy(st)) - pos;
    const V3 n = normalize(xformVector(q.normalToObject, st.N));
    const float twp = q.transitionWidthPower;
    V3 blend{powf(fabsf(n.x), twp), powf(fabsf(n.y), twp), powf(fabsf(n.z), twp)};
    blend = blend / (blend.x + blend.y + blend.z);
    Color tx[6];
    bool read[6];
    for (U32 f = 0; f < 6; ++f) {
        tx[f] = rgb(0.0f);
        const float w = f % 3 == 0 ? blend.x : f % 3 == 1 ? blend.y : blend.z;
        read[f] = !(w < 0.0005f);
        if (!read[f]) continue;
        V3 U, dx, dy;
        switch (f) {  // swizzleTriplanarFace
        case 0: U = V3{-pos.z, -pos.y, -pos.x}; dx = V3{-pdx.z, pdx.y, -pdx.x}; dy = V3{-pdy.z, pdy.y, -pdy.x}; break;
        case 1: U = V3{pos.x, pos.z, -pos.y}; dx = V3{pdx.x, -pdx.z, -pdx.y}; dy = V3{pdy.x, -pdy.z, -pdy.y}; break;
        case 2: U = V3{pos.x, -pos.y, -pos.z}; dx = V3{pdx.x, pdx.y, -pdx.z}; dy = V3{pdy.x, pdy.y, -pdy.z}; break;
        case 3: U = V3{pos.z, -pos.y, pos.x}; dx = V3{pdx.z, pdx.y, pdx.x}; dy = V3{pdy.z, pdy.y, pdy.x}; break;
        case 4: U = V3{pos.x, -pos.z, pos.y}; dx = V3{pdx.x, pdx.z, pdx.y}; dy = V3{pdy.x, pdy.z, pdy.y}; break;
        default: U = V3{-pos.x, -pos.y, pos.z}; dx = V3{-pdx.x, pdx.y, pdx.z}; dy = V3{-pdy.x, pdy.y, pdy.z}; break;
        }
        const MapTriplanarTexture& t = q.textures[U32(q.indices[f])];
        if (t.texture < 0.0f) {
            tx[f] = fatal;
            continue;
        }
        if (t.active == 0.0f) continue;
        const float* m = t.xform;
        // transformPoint / transformVector (row vectors: x vx + y vy + z vz, then p), then centred at (0.5, 0.5)
        const float s = (U.x * m[0] + (U.y * m[3] + U.z * m[6])) + m[9] + 0.5f;
        const float tt = (U.x * m[1] + (U.y * m[4] + U.z * m[7])) + m[10] + 0.5f;
        const float dsdx = dx.x * m[0] + (dx.y * m[3] + dx.z * m[6]), dtdx = dx.x * m[1] + (dx.y * m[4] + dx.z * m[7]);
        const float dsdy = dy.x * m[0] + (dy.y * m[3] + dy.z * m[6]), dtdy = dy.x * m[1] + (dy.y * m[4] + dy.z * m[7]);
        // tex::lookup flips T and negates the T derivatives as ImageMap does; BasicTexture does neither.
        const tex::Binding b{nullptr, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, U32(t.wrap), U32(t.gamma), textureMode(st)};
        const tex::Footprint fp{s, 1.0f - tt, dsdx, -dtdx, dsdy, -dtdy};
        V3 texel;
        const U32 status = tex.lookup(U32(t.texture), b, fp, texel);
        if (status != tex::Ready) return status;
        tx[f] = Color{texel.x, texel.y, texel.z};
    }
    if (!normalMap) {
        out = (n.x > 0.0f ? tx[0] : tx[3]) * blend.x + (n.y > 0.0f ? tx[1] : tx[4]) * blend.y + (n.z > 0.0f ? tx[2] : tx[5]) * blend.z;
        return tex::Ready;
    }
    V3 normals[6];
    float lengths[6];
    // each face's tangent in the projector's space
    const V3 tangents[6] = {{0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}};
    for (U32 f = 0; f < 6; ++f) {
        normals[f] = n;
        lengths[f] = 1.0f;
        if (!read[f]) continue;
        V3 t = tx[f];
        lengths[f] = length(t);
        if (q.normalEncoding == 0.0f) {
            t = 2.0f * t - rgb(1.0f);
            lengths[f] = length(t);
            t = t / lengths[f];
        }
        normals[f] = localToGlobal(frame(n, tangents[f]), t);
    }
    const V3 b = (n.x > 0.0f ? normals[0] : normals[3]) * blend.x + (n.y > 0.0f ? normals[1] : normals[4]) * blend.y +
                 (n.z > 0.0f ? normals[2] : normals[5]) * blend.z;
    const float bl = (n.x > 0.0f ? lengths[0] : lengths[3]) * blend.x + (n.y > 0.0f ? lengths[1] : lengths[4]) * blend.y +
                     (n.z > 0.0f ? lengths[2] : lengths[5]) * blend.z;
    // transformNormal from the projector's space: its world-to-object transposed
    const float* m = q.toObject;
    const V3 r{m[0] * b.x + m[4] * b.y + m[8] * b.z, m[1] * b.x + m[5] * b.y + m[9] * b.z, m[2] * b.x + m[6] * b.y + m[10] * b.z};
    out = normalize(r) * bl;
    return tex::Ready;
}

// ProjectPlanarNormalMap's sampleNormal: the texel at the hit on the projector's +Z face (N outside it unless it
// wraps), as a tangent-space normal in each cube face's frame about the projector-space normal (X flipped on faces 0,
// 4 and 5), blended as triplanar, in render space at the texel's length times the weights' sum.
template <class Tex>
X2_CALL U32 sampleProjectPlanarNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    const MapProjectPlanarNormalParams& q = *reinterpret_cast<const MapProjectPlanarNormalParams*>(P.p);
    out = st.N;
    if (!h.object || h.texture == MapNoTexture) return tex::Ready;
    const V3 pos = xformPoint(q.toObject, st.P);
    const V3 pdx = xformPoint(q.toObject, st.P + statedPdx(st)) - pos, pdy = xformPoint(q.toObject, st.P + statedPdy(st)) - pos;
    const V3 n = normalize(xformVector(q.normalToObject, st.N));
    // swizzleTriplanarFace onto face 2 (+Z)
    const V3 U{pos.x, -pos.y, -pos.z}, dx{pdx.x, pdx.y, -pdx.z}, dy{pdy.x, pdy.y, -pdy.z};
    const float s = U.x + 0.5f, t = U.y + 0.5f;
    if (q.wrap == 0.0f && (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f)) return tex::Ready;
    const tex::Binding b{nullptr, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, h.wrap, 0u, textureMode(st)};
    const tex::Footprint fp{s, 1.0f - t, dx.x, -dx.y, dy.x, -dy.y};
    V3 texel;
    const U32 status = tex.lookup(h.texture, b, fp, texel);
    if (status != tex::Ready) return status;
    V3 tn = texel;
    float len = length(tn);
    if (q.normalEncoding == 0.0f) {
        tn = 2.0f * tn - rgb(1.0f);
        len = length(tn);
        tn = tn / len;
    }
    const V3 flipped{-tn.x, tn.y, tn.z};
    const float twp = q.transitionWidthPower;
    V3 blend{powf(fabsf(n.x), twp), powf(fabsf(n.y), twp), powf(fabsf(n.z), twp)};
    blend = blend / (blend.x + blend.y + blend.z);
    const V3 tangents[6] = {{0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}};
    V3 o[6];
    for (U32 f = 0; f < 6; ++f) {
        o[f] = n;
        const V3 c = cross(n, tangents[f]);
        if (isZero(c.x) && isZero(c.y) && isZero(c.z)) continue;
        o[f] = localToGlobal(frame(n, tangents[f]), (f == 0 || f == 4 || f == 5) ? flipped : tn);
    }
    const V3 bn = (n.x > 0.0f ? o[0] : o[3]) * blend.x + (n.y > 0.0f ? o[1] : o[4]) * blend.y + (n.z > 0.0f ? o[2] : o[5]) * blend.z;
    const float* m = q.toObject;  // transformNormal from the projector's space: its world-to-object transposed
    const V3 r{m[0] * bn.x + m[4] * bn.y + m[8] * bn.z, m[1] * bn.x + m[5] * bn.y + m[9] * bn.z, m[2] * bn.x + m[6] * bn.y + m[10] * bn.z};
    out = normalize(r) * (len * (blend.x + blend.y + blend.z));
    return tex::Ready;
}

// ProjectCameraMap_v2's sample: the hit in the projector's screen space; black outside [-1, 1] when asked, or on a
// face turned from the projector unless back faces take it; else its texture at the screen position mapped to
// [0, 1], as BasicTexture looks it up.
template <class Tex>
X2_CALL U32 sampleProjectCamera(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    const MapProjectCameraParams& q = *reinterpret_cast<const MapProjectCameraParams*>(P.p);
    const bool normalMap = h.kind == MapProjectCameraNormal;
    out = normalMap ? st.N : rgb(0.0f);
    if (!h.object) return tex::Ready;
    const float* m = q.worldToScreen;
    auto screen = [m](V3 p) {  // transformH
        const float x = p.x * m[0] + p.y * m[4] + p.z * m[8] + m[12], y = p.x * m[1] + p.y * m[5] + p.z * m[9] + m[13],
                    z = p.x * m[2] + p.y * m[6] + p.z * m[10] + m[14], w = p.x * m[3] + p.y * m[7] + p.z * m[11] + m[15];
        return V3{x / w, y / w, z / w};
    };
    const V3 ps = screen(st.P), dx = screen(st.P + statedPdx(st)) - ps, dy = screen(st.P + statedPdy(st)) - ps;
    if (q.blackOutsideProjection != 0.0f && (ps.x < -1.f || ps.x > 1.f || ps.y < -1.f || ps.y > 1.f)) return tex::Ready;
    V3 nc{0.0f, 0.0f, -1.0f};
    if (q.projectOnBackFaces == 0.0f) nc = normalize(xformVector(q.normalToCamera, st.N));
    if (normalMap) {
        // ProjectCameraNormalMap: the texel, decoded and taken in the frame of N and dPds. On a face turned away MoonRay
        // decodes the shading normal itself as if it were a texel: its render-space components.
        V3 normal = worldToRenderNormal(st.worldToRender, st.N);
        if (q.projectOnBackFaces != 0.0f || nc.z > 0.0f) {
            const float s = ps.x * 0.5f + 0.5f, t = ps.y * 0.5f + 0.5f;
            const tex::Binding b{nullptr, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0u, 0u, textureMode(st)};
            const tex::Footprint fp{s, 1.0f - t, dx.x * 0.5f, -(dx.y * 0.5f), dy.x * 0.5f, -(dy.y * 0.5f)};
            const U32 status = tex.lookup(h.texture, b, fp, normal);
            if (status != tex::Ready) return status;
        }
        if (!finite3(normal) || (isZero(normal.x) && isZero(normal.y) && isZero(normal.z))) {
            out = st.N;
            return tex::Ready;
        }
        if (q.normalEncoding == 0.0f) normal = 2.0f * normal - rgb(1.0f);
        out = localToGlobal(frame(st.N, normalize(st.dPds)), normal);
        return tex::Ready;
    }
    if (q.projectOnBackFaces == 0.0f && !(nc.z > 0.0f)) return tex::Ready;
    if (h.texture == MapNoTexture) return tex::Ready;
    // s2uv: [-1, 1] to [0, 1]
    const float s = ps.x * 0.5f + 0.5f, t = ps.y * 0.5f + 0.5f;
    const tex::Binding b{nullptr, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0u, h.gamma, textureMode(st)};
    const tex::Footprint fp{s, 1.0f - t, dx.x * 0.5f, -(dx.y * 0.5f), dy.x * 0.5f, -(dy.y * 0.5f)};
    V3 texel;
    const U32 status = tex.lookup(h.texture, b, fp, texel);
    if (status != tex::Ready) return status;
    out = Color{texel.x, texel.y, texel.z};
    return tex::Ready;
}

// UVTransformMap's sample: the hit's texture coordinates, or P in world, render (also camera: MoonRay's render
// space is the camera's) or its object's space; turned by the prepared rotation (about the centre in the plane, about
// the axis in 3D spaces) when the angle is not zero; then scaled and offset.
X2_HD inline Color sampleUVTransform(const MapParams& P, const State& st) {
    using Q = MapUVTransformParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const float* rot = P.p + sizeof(Q) / sizeof(float);  // Xform3f rows vx, vy, vz, p
    const int space = int(q.space);
    V3 pos;
    if (space == 6) pos = V3{st.s, st.t, 0.0f};
    else if (space == 2) pos = st.P;
    else if (space == 4) pos = st.worldToObject ? xformPoint(st.worldToObject, st.P) : st.P;
    else pos = st.worldToRender ? xformPoint(st.worldToRender, st.P) : st.P;
    if (!isZero(q.rotation_angle))
        pos = V3{pos.x * rot[0] + (pos.y * rot[3] + pos.z * rot[6]) + rot[9], pos.x * rot[1] + (pos.y * rot[4] + pos.z * rot[7]) + rot[10],
                 pos.x * rot[2] + (pos.y * rot[5] + pos.z * rot[8]) + rot[11]};
    float scale[2], offset[2];
    P.v2(X2_OFF(Q, scale), scale);
    P.v2(X2_OFF(Q, offset), offset);
    return Color{scale[0] * pos.x + offset[0], scale[1] * pos.y + offset[1], 0.0f};
}

// ColorCorrectLegacyMap's sample: brightness, contrast about mid grey, hue shift, saturation by the legacy luminance,
// inversion, monochrome, the multiplier and a clamp, then the input mixed back by the mask.
X2_HD inline Color sampleColorCorrectLegacy(const MapParams& P) {
    using Q = MapColorCorrectLegacyParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    Color r = input;
    if (q.on == 0 || !(P.f(X2_OFF(Q, mask)) > 0.0f)) return r;
    auto lum = [](Color c) { return 0.3f * c.x + 0.59f * c.y + 0.11f * c.z; };
    if (!isZero(q.brightness)) r = r + rgb(q.brightness / 255.0f);
    if (!isZero(q.contrast)) {
        float contrast = q.contrast / 100.0f;
        const Color mid = rgb(0.5f);
        if (contrast < 0) {
            contrast *= -1.0f;
            r = lerp(r, mid, contrast);
        } else {
            if (isEqual(contrast, 1.0f)) contrast = 0.999f;
            contrast = 1.0f / (1.0f - contrast);
            r = ((r - mid) * contrast) + mid;
        }
    }
    if (!isZero(q.hue)) {
        Color hsv = rgbToHsv(r);
        hsv.x = fmodf(hsv.x + (q.hue / 360.0f), 1.0f);
        r = hsvToRgb(hsv);
    }
    if (!isZero(q.saturation)) r = lerp(rgb(lum(r)), r, (q.saturation + 100.0f) / 100.0f);
    if (q.invert != 0) r = Color{1.0f - r.x, 1.0f - r.y, 1.0f - r.z};
    switch (int(q.monochrome)) {
    case 1: r = rgb(lum(r)); break;
    case 2: r = rgb((r.x + r.y + r.z) / 3.0f); break;
    case 3: r = rgb(fminf(r.x, fminf(r.y, r.z))); break;
    case 4: r = rgb(fmaxf(r.x, fmaxf(r.y, r.z))); break;
    case 5: r = rgb(r.x); break;
    case 6: r = rgb(r.y); break;
    case 7: r = rgb(r.z); break;
    default: break;
    }
    r = r * q.multiplier;
    if (q.clamp != 0) r = Color{saturate(r.x), saturate(r.y), saturate(r.z)};
    return lerp(r, input, 1.0f - P.f(X2_OFF(Q, mask)));
}

// CombineNormalMap's conditionNormal: a render-space normal into the frame's tangent space, its length eased toward
// one by the dial and the direction toward the unbent normal.
X2_HD inline V3 conditionNormal(V3 input, float dial, const Frame& frame, float& length) {
    const V3 N{0.0f, 0.0f, 1.0f};
    dial = clampf(dial, 0.0f, 1.0f);
    V3 n = globalToLocal(frame, input);
    length = (1.0f - dial) * 1.0f + dial * shade::length(n);
    if (isZero(length)) {
        length = 1.0f;
        return N;
    }
    n = n / length;
    if (!isOne(dial)) n = normalize(N + (n - N) * dial);
    return n;
}
// CombineNormalMap's sampleNormal: its two normal maps conditioned in the frame of N and dPds, joined by reoriented
// normal mapping at their mean length, kept above the unbent hemisphere. Without both inputs, two zero normals.
X2_HD inline Color sampleCombineNormal(const MapParams& P, const State& st) {
    const float r1 = P.p[2], r2 = P.p[3];
    const bool valid = r1 >= 0.0f && r2 >= 0.0f;
    const V3 s1 = valid ? P.regs[U32(r1)] : V3{0, 0, 0}, s2 = valid ? P.regs[U32(r2)] : V3{0, 0, 0};
    const Frame frame = shade::frame(st.N, normalize(st.dPds));
    float l1, l2;
    const V3 n1 = conditionNormal(s1, P.f(0), frame, l1), n2 = conditionNormal(s2, P.f(1), frame, l2);
    const V3 t{n1.x, n1.y, n1.z + 1.0f}, u{-1.0f * n2.x, -1.0f * n2.y, n2.z};
    V3 result = normalize(t * dot(t, u) - u * t.z) * ((l1 + l2) * 0.5f);
    if (result.z <= 0.0f) {
        result.z = Epsilon;
        result = normalize(result);
    }
    return localToGlobal(frame, result);
}

// NoiseMap_v2's sample: P in its space, scaled by the frequency and transformed; distorted by a 3D noise; then a
// fractal Perlin or simplex noise per channel, remapped to [0, 1], biased, gained, inverted, smoothstepped, scaled.
X2_HD inline NoiseTable noiseTableAt(const float* p, const float* prep, int slot) {
    const U32 offset = U32(prep[24 + slot]);
    NoiseTable t;
    if (offset) {
        t.perm = p + offset;
        t.g = t.perm + NoiseTableSize;
    }
    return t;
}
X2_HD inline Color sampleNoiseMap(const MapNodeHeader& h, const MapParams& P, const State& st) {
    using Q = MapNoiseParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const float* prep = P.p + sizeof(Q) / sizeof(float);
    const int space = int(q.space);
    V3 pos;
    if (space == int(SpaceTexture)) pos = V3{st.s, st.t, 0.0f};
    else if (space == 7) pos = P.c(X2_OFF(Q, input_texture_coordinates));  // input texture coordinates
    else pos = spacePoint(st, U32(space), h.object ? prep + 12 : nullptr);
    pos = pos * P.f(X2_OFF(Q, frequency_multiplier));
    pos = xformPoint(prep, pos);
    const float lacunarity = P.f(X2_OFF(Q, lacunarity)), persistence = P.f(X2_OFF(Q, persistence));
    const float amplitude = P.f(X2_OFF(Q, amplitude)), distortion = P.f(X2_OFF(Q, distortion)), maxLevel = P.f(X2_OFF(Q, max_level));
    const float biasVal = P.f(X2_OFF(Q, bias)), gainVal = P.f(X2_OFF(Q, gain)), flowAngle = P.f(X2_OFF(Q, flow_angle));
    const float advectionRate = P.f(X2_OFF(Q, flow_advection_rate)), time = P.f(X2_OFF(Q, time));
    float smooth[2];
    P.v2(X2_OFF(Q, smoothstep), smooth);
    float p4[4] = {pos.x, pos.y, pos.z, time};
    if (!isZero(distortion)) {
        const NoiseTable d = noiseTableAt(P.p, prep, 3);
        float derivatives[3];
        const float dn = int(q.distortion_noise_type) == 1 ? simplex3D(d, p4, 0.0f, derivatives) : perlin3D(d, p4);
        for (int c = 0; c < 3; ++c) p4[c] += dn * distortion;
    }
    const bool fourD = q.use_4D_noise != 0, simplex = int(q.noise_type) == 1, color = q.color != 0;
    float v[3] = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < (color ? 3 : 1); ++c) {
        const NoiseTable t = noiseTableAt(P.p, prep, c);
        float n = simplex ? simplexFractal(t, p4, fourD, flowAngle, advectionRate, maxLevel, persistence, lacunarity)
                          : perlinFractal(t, p4, fourD, maxLevel, persistence, lacunarity);
        n = (n + 1.0f) * 0.5f;
        n = gainf(biasf(n, biasVal), gainVal);
        if (q.invert != 0) n = 1.0f - n;
        if (q.use_smoothstep != 0) n = noiseSmoothStep(n, smooth[0], smooth[1]);
        v[c] = n * amplitude;
    }
    if (color) return Color{v[0], v[1], v[2]};
    return lerp(P.c(X2_OFF(Q, color_A)), P.c(X2_OFF(Q, color_B)), v[0]);
}

// NoiseWorleyMap's adjust: remap, bias and gain, smoothstep, the colours, inversion.
X2_HD inline Color worleyAdjust(float d, const float remap[2], float biasVal, float gainVal, bool useSmoothstep, const float smooth[2],
                                bool invert, Color a, Color b) {
    const float denom = remap[1] - remap[0];
    d = (d - remap[0]) / (denom != 0.f ? denom : 0.000001f);
    d = gainf(biasf(d, biasVal), gainVal);
    if (useSmoothstep) d = noiseSmoothStep(d, smooth[0], smooth[1]);
    Color n{a.x * (1.0f - d) + b.x * d, a.y * (1.0f - d) + b.y * d, a.z * (1.0f - d) + b.z * d};
    if (invert) n = Color{1.0f - n.x, 1.0f - n.y, 1.0f - n.z};
    return n;
}
X2_HD inline Color sampleWorleyMap(const MapNodeHeader& h, const MapParams& P, const State& st) {
    using Q = MapWorley3Params;  // _v2's layout is the same
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const float* prep = P.p + sizeof(Q) / sizeof(float);
    const int space = int(q.space);
    V3 pos;
    if (space == int(SpaceTexture)) pos = V3{st.s, st.t, 0.0f};
    else if (space == 7) pos = P.c(X2_OFF(Q, input_texture_coordinates));
    else pos = spacePoint(st, U32(space), h.object ? prep + 12 : nullptr);
    pos = pos * P.f(X2_OFF(Q, frequency));
    pos = xformPoint(prep, pos);
    const float f0 = P.f(X2_OFF(Q, F1)), f1 = P.f(X2_OFF(Q, F2)), f2 = P.f(X2_OFF(Q, F3)), f3 = P.f(X2_OFF(Q, F4));
    const float jitter = clampf(P.f(X2_OFF(Q, jitter)), 0.f, 1.f), maxLevel = P.f(X2_OFF(Q, max_level));
    const float mn = P.f(X2_OFF(Q, minkowski_number)), biasVal = P.f(X2_OFF(Q, bias)), gainVal = P.f(X2_OFF(Q, gain));
    float remap[2], smooth[2];
    P.v2(X2_OFF(Q, remap), remap);
    P.v2(X2_OFF(Q, smoothstep), smooth);
    const Color colorA = P.c(X2_OFF(Q, color_A)), colorB = P.c(X2_OFF(Q, color_B));
    WorleyTables w;
    const U32 offset = U32(prep[44]);
    w.perm.perm = P.p + offset;
    w.x = w.perm.perm + NoiseTableSize;
    w.y = w.x + NoiseTableSize;
    w.z = w.y + NoiseTableSize;
    w.probLow = prep + 24;
    w.probHigh = prep + 34;
    WorleyPoint wp[4];
    worleyFractal(w, h.kind == MapWorley3, jitter, mn, maxLevel, int(q.distance_method), pos, wp);
    const int cell = int(q.cell_id);
    switch (int(q.output_mode)) {
    case 0: {  // distance
        const float d = f0 * wp[0].dist + f1 * wp[1].dist + f2 * wp[2].dist + f3 * wp[3].dist;
        return worleyAdjust(d, remap, biasVal, gainVal, q.use_smoothstep != 0, smooth, q.invert != 0, colorA, colorB);
    }
    case 1:  // gradient
        return f0 * wp[0].gradient + f1 * wp[1].gradient + f2 * wp[2].gradient + f3 * wp[3].gradient;
    case 2: return worleyCellColor(w, wp[cell].id);  // cell id
    case 3: {  // cell edges: points equidistant from two cell centres
        const float d = noiseSmoothStep(wp[1].dist - wp[0].dist, 0.0f, 0.05f);
        return worleyAdjust(d, remap, biasVal, gainVal, q.use_smoothstep != 0, smooth, q.invert != 0, colorA, colorB);
    }
    case 4: {  // points
        const float ps = fmaxf(Epsilon, 0.1f * q.point_size);
        const float d = 1.0f - noiseSmoothStep(wp[cell].dist, ps - Epsilon, ps);
        return rgb(d);
    }
    default: return rgb(0.0f);
    }
}

}}}
#include "shade_mapnodes_more.h"
namespace xpu2 { namespace interactive { namespace shade {
// One map's result. Status as tex::lookup's: Ready, Miss (a texture tile is not resident yet) or Invalid.
// sampleMap's cases, one function each (X2_CALL: out of line on the device, so OptiX compiles them apart).
template <class Tex>
X2_CALL U32 sampleMapCaseAxisAngle(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleAxisAngle(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseTransformNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleTransformNormal(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRgbToLab(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleRgbToLab(P.c(X2_OFF(MapRgbToLabParams, input_color))); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRandom(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleRandom(P); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseMultiChannelToFloat(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleMultiChannelToFloat(P); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseLayer(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleLayerMap(P, false); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseLayer2(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleLayerMap(P, true); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseHair(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleHairMap(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseHairColumn(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleHairColumn(st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseHairColorPresets(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleHairColorPresets(int(reinterpret_cast<const MapHairColorPresetsParams*>(P.p)->color)); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseToon(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleToon(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCurvature(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleCurvature(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseWireframe(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleWireframe(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseDeformation(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleDeformation(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseDirectional(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleDirectional(P, st, h); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseTransformSpace(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleTransformSpace(P, st, h); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectTriplanarUdim(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleTriplanarUdim(P, st, h); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseExtraAov(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = P.c(X2_OFF(MapExtraAovParams, color)); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseList(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = rgb(1.0f); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseDebug(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleDebug(P, st, h); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseDistortNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleDistortNormal(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRandomNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleRandomNormal(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseConstantColor(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = reinterpret_cast<const MapConstantColorParams*>(P.p)->color_value; return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseConstantScalar(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = rgb(reinterpret_cast<const MapConstantScalarParams*>(P.p)->scalar_value); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCheckerboard(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCheckerboardParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    float s = st.s, t = st.t;
    if (int(q.texture_coordinates) == 1) {
        const Color uvw = P.c(X2_OFF(Q, input_texture_coordinates));
        s = uvw.x;
        t = uvw.y;
    }
    float smod = fmodf(s * float(int(q.num_u_tiles)) / 2.0f, 1.0f), tmod = fmodf(t * float(int(q.num_v_tiles)) / 2.0f, 1.0f);
    if (smod < 0.0f) smod += 1.0f;
    if (tmod < 0.0f) tmod += 1.0f;
    out = (smod < 0.5f) == (tmod < 0.5f) ? q.color_A : q.color_B;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseClamp(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapClampParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    Color c = P.c(X2_OFF(Q, input));
    if (q.clamp != 0) c = clampColor(c, q.clamp_min, q.clamp_max);
    out = c;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseFloatToRgb(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapFloatToRgbParams;
    out = Color{P.f(X2_OFF(Q, R)), P.f(X2_OFF(Q, G)), P.f(X2_OFF(Q, B))};
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRgbToFloat(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapRgbToFloatParams;
    const Color c = P.c(X2_OFF(Q, input));
    float v = 0.0f;
    switch (int(reinterpret_cast<const Q*>(P.p)->mode)) {
    case 0: v = c.x; break;
    case 1: v = c.y; break;
    case 2: v = c.z; break;
    case 3: v = fminf(fminf(c.x, c.y), c.z); break;
    case 4: v = fmaxf(fmaxf(c.x, c.y), c.z); break;
    case 5: v = (c.x + c.y + c.z) / 3.0f; break;
    case 6: v = c.x + c.y + c.z; break;
    case 7: v = luminance(c); break;
    default: out = rgb(0); return tex::Ready;  // MoonRay leaves the sample unset
    }
    out = rgb(v);
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseHsvToRgb(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = hsvToRgb(P.c(X2_OFF(MapHsvToRgbParams, input))); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRgbToHsv(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = rgbToHsv(P.c(X2_OFF(MapRgbToHsvParams, input))); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRemap(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapRemapParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float inMin = P.f(X2_OFF(Q, input_min)), inMax = P.f(X2_OFF(Q, input_max)), outMin = P.f(X2_OFF(Q, output_min)),
                outMax = P.f(X2_OFF(Q, output_max)), biasAmount = P.f(X2_OFF(Q, midpoint_bias));
    const Color inMinRGB = P.c(X2_OFF(Q, input_min_RGB)), inMaxRGB = P.c(X2_OFF(Q, input_max_RGB)),
                outMinRGB = P.c(X2_OFF(Q, output_min_RGB)), outMaxRGB = P.c(X2_OFF(Q, output_max_RGB)),
                biasRGB = P.c(X2_OFF(Q, midpoint_bias_RGB));
    const bool applyBias = !isEqual(biasAmount, 0.5f);
    Color r = input;
    if (int(q.remap_method) == 0) {
        if (!isEqual(inMin, outMin) || !isEqual(inMax, outMax)) r = remapUniform(r, inMin, inMax, outMin, outMax, applyBias, biasAmount);
        else if (applyBias) r = Color{biasf(r.x, biasAmount), biasf(r.y, biasAmount), biasf(r.z, biasAmount)};
        if (q.clamp != 0) r = clampColor(r, q.clamp_min, q.clamp_max);
    } else if (int(q.remap_method) == 1) {
        if (!isEqualColor(inMinRGB, outMinRGB) || !isEqualColor(inMaxRGB, outMaxRGB)) {
            r = remapRGB(r, inMinRGB, inMaxRGB, outMinRGB, outMaxRGB, biasRGB);
        } else {
            r.x = !isEqual(biasRGB.x, 0.5f) ? biasf(r.x, biasRGB.x) : r.x;
            r.y = !isEqual(biasRGB.y, 0.5f) ? biasf(r.y, biasRGB.y) : r.y;
            r.z = !isEqual(biasRGB.z, 0.5f) ? biasf(r.z, biasRGB.z) : r.z;
        }
        if (q.clamp_RGB != 0)
            r = Color{clampf(r.x, q.clamp_min_RGB.x, q.clamp_max_RGB.x), clampf(r.y, q.clamp_min_RGB.y, q.clamp_max_RGB.y),
                      clampf(r.z, q.clamp_min_RGB.z, q.clamp_max_RGB.z)};
    }
    out = r;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseBlend(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapBlendParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const float amount = P.f(X2_OFF(Q, blend_amount)), tMin = P.f(X2_OFF(Q, threshold_min)), tMax = P.f(X2_OFF(Q, threshold_max));
    if (amount <= tMin) {
        out = P.c(X2_OFF(Q, color_A));
    } else if (amount >= tMax) {
        out = P.c(X2_OFF(Q, color_B));
    } else {
        const Color a = P.c(X2_OFF(Q, color_A)), b = P.c(X2_OFF(Q, color_B));
        const float divisor = tMax - tMin;
        float t = 0.0f;
        if (divisor > 0.0f) t = (amount - tMin) / divisor;
        if (int(q.blend_type) == 1) t = t * t * (3 - 2 * t);
        out = lerp(a, b, t);
    }
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseColorCorrect(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapColorCorrectParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    Color s = input;
    const float mix = P.f(X2_OFF(Q, mix));
    if (q.on == 0 || isZero(mix)) {
        out = s;
        return tex::Ready;
    }
    if (q.hue_shift_enabled != 0) applyHueShift(P.f(X2_OFF(Q, hue_shift)), s);
    if (q.saturation_enabled != 0) {
        const Color sat = q.use_per_channel_saturation != 0
                              ? Color{P.f(X2_OFF(Q, saturation_r)), P.f(X2_OFF(Q, saturation_g)), P.f(X2_OFF(Q, saturation_b))}
                              : rgb(P.f(X2_OFF(Q, saturation)));
        applySaturation(sat.x, sat.y, sat.z, s.x, s.y, s.z);
    }
    if (q.contrast_enabled != 0) {
        const Color con = q.use_per_channel_contrast != 0
                              ? Color{P.f(X2_OFF(Q, contrast_r)), P.f(X2_OFF(Q, contrast_g)), P.f(X2_OFF(Q, contrast_b))}
                              : rgb(P.f(X2_OFF(Q, contrast)));
        applyContrast(con, s);
    }
    if (q.gamma_enabled != 0) {
        const Color g = q.use_per_channel_gamma != 0 ? Color{P.f(X2_OFF(Q, gamma_r)), P.f(X2_OFF(Q, gamma_g)), P.f(X2_OFF(Q, gamma_b))}
                                                     : rgb(P.f(X2_OFF(Q, gamma)));
        applyGamma(1.0f / fmaxf(Epsilon, g.x), 1.0f / fmaxf(Epsilon, g.y), 1.0f / fmaxf(Epsilon, g.z), s.x, s.y, s.z);
    }
    if (q.gain_offset_enabled != 0) {
        Color gain, offset;
        if (q.use_per_channel_gain_offset != 0) {
            gain = Color{P.f(X2_OFF(Q, gain_r)), P.f(X2_OFF(Q, gain_g)), P.f(X2_OFF(Q, gain_b))};
            offset = Color{P.f(X2_OFF(Q, offset_r)), P.f(X2_OFF(Q, offset_g)), P.f(X2_OFF(Q, offset_b))};
        } else {
            gain = rgb(P.f(X2_OFF(Q, gain)));
            offset = rgb(P.f(X2_OFF(Q, offset)));
        }
        applyGainAndOffset(gain.x, gain.y, gain.z, offset.x, offset.y, offset.z, s.x, s.y, s.z);
    }
    if (q.TMI_enabled != 0) {
        const Color tmi = P.c(X2_OFF(Q, TMI));
        applyTMI(tmi.x, tmi.y, tmi.z, s.x, s.y, s.z);
    }
    s = mixWithInput(input, s, mix);
    if (q.clamp != 0) s = clampColor(s, q.clamp_min, q.clamp_max);
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCContrast(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCContrastParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        const Color con = q.use_per_channel_contrast != 0
                              ? Color{P.f(X2_OFF(Q, contrast_r)), P.f(X2_OFF(Q, contrast_g)), P.f(X2_OFF(Q, contrast_b))}
                              : rgb(P.f(X2_OFF(Q, contrast)));
        applyContrast(con, s);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCGainOffset(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCGainOffsetParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        Color gain, offset;
        if (q.use_per_channel_gain_offset != 0) {
            gain = Color{P.f(X2_OFF(Q, gain_r)), P.f(X2_OFF(Q, gain_g)), P.f(X2_OFF(Q, gain_b))};
            offset = Color{P.f(X2_OFF(Q, offset_r)), P.f(X2_OFF(Q, offset_g)), P.f(X2_OFF(Q, offset_b))};
        } else {
            gain = rgb(P.f(X2_OFF(Q, gain)));
            offset = rgb(P.f(X2_OFF(Q, offset)));
        }
        applyGainAndOffset(gain.x, gain.y, gain.z, offset.x, offset.y, offset.z, s.x, s.y, s.z);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCGamma(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCGammaParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        const Color g = q.use_per_channel_gamma != 0 ? Color{P.f(X2_OFF(Q, gamma_r)), P.f(X2_OFF(Q, gamma_g)), P.f(X2_OFF(Q, gamma_b))}
                                                     : rgb(P.f(X2_OFF(Q, gamma)));
        applyGamma(1.0f / fmaxf(Epsilon, g.x), 1.0f / fmaxf(Epsilon, g.y), 1.0f / fmaxf(Epsilon, g.z), s.x, s.y, s.z);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCHsv(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCHsvParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    if (q.on == 0) {
        out = input;
        return tex::Ready;
    }
    Color hsv = rgbToHsv(input);
    float h = hsv.x, s = hsv.y, v = hsv.z;
    const float hueShift = P.f(X2_OFF(Q, hue_shift));
    if (!isZero(hueShift)) h = fmodf(h + hueShift / 360.f, 1.f);
    const float sf = P.f(X2_OFF(Q, saturation_factor));
    if (!isOne(sf)) s = s * sf;
    const float vf = P.f(X2_OFF(Q, value_factor));
    if (!isOne(vf)) v = v * vf;
    hsvContrast(clampf(q.saturation_contrast, -1.f, 1.f), s);
    hsvContrast(clampf(q.value_contrast, -1.f, 1.f), v);
    if (!isZero(q.saturation_shift)) s = s + q.saturation_shift;
    if (!isZero(q.value_shift)) v = v + q.value_shift;
    Color r = hsvToRgb(Color{h, s, v});
    if (q.clamp != 0) r = clampColor(r, 0.0f, 1.0f);
    out = r;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCHueShift(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCHueShiftParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        applyHueShift(P.f(X2_OFF(Q, hue_shift)), s);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCSaturation(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCSaturationParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        const Color sat = q.use_per_channel_saturation != 0
                              ? Color{P.f(X2_OFF(Q, saturation_r)), P.f(X2_OFF(Q, saturation_g)), P.f(X2_OFF(Q, saturation_b))}
                              : rgb(P.f(X2_OFF(Q, saturation)));
        applySaturation(sat.x, sat.y, sat.z, s.x, s.y, s.z);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCCTMI(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapCCTMIParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    const Color input = P.c(X2_OFF(Q, input));
    const float mix = P.f(X2_OFF(Q, mix));
    Color s = input;
    if (q.on != 0 && !isZero(mix)) {
        const Color tmi = P.c(X2_OFF(Q, TMI));
        applyTMI(tmi.x, tmi.y, tmi.z, s.x, s.y, s.z);
        s = mixWithInput(input, s, mix);
    }
    out = s;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseSwitchColor(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    int choice = int(P.f(0)) % 64;
    if (choice < 0) choice = -choice;  // MoonRay's % of a negative index reads before the inputs; refused at admission
    out = P.c(1 + 3 * U32(choice));
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseSwitchFloat(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    int choice = int(P.f(0)) % 64;
    if (choice < 0) choice = -choice;
    out = rgb(P.f(1 + U32(choice)));
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseImage(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleImageMap(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectTriplanar(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleTriplanar(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectCamera(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleProjectCamera(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectPlanarNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleProjectPlanarNormal(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRgbToNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = renderToWorldNormal(st.worldToRender, P.c(0)); return tex::Ready;  // its colour, a render-space normal
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseColorCorrectLegacy(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleColorCorrectLegacy(P); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseUVTransform(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleUVTransform(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseSwitchNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    // the chosen input's normal, N where there is none
    int choice = int(P.f(0));
    if (choice < 0) choice = 0;
    const float r = P.p[1 + U32(choice % 64)];
    out = r >= 0.0f ? P.regs[U32(r)] : st.N;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseCombineNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleCombineNormal(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseImageNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleImageNormalMap(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseModoTexture(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleModoTexture(P, st, nullptr); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseOp(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleOpMap(P); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseGradient(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleGradientMap(h, P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseRamp(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleRampMap(h, P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseAttribute(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleAttributeMap(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseUsdUVTexture(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    return sampleUsdUVTexture(h, P, st, tex, out);
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseUsdTransform2d(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleUsdTransform2d(P); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseUsdPrimvar(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleUsdPrimvar(P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectPlanar(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleProjection(h, P, st, sizeof(MapProjectPlanarParams) / sizeof(float)); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectSpherical(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleProjection(h, P, st, sizeof(MapProjectSphericalParams) / sizeof(float)); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseProjectCylindrical(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleProjection(h, P, st, sizeof(MapProjectCylindricalParams) / sizeof(float));
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseNoise(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleNoiseMap(h, P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseWorley2(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = sampleWorleyMap(h, P, st); return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseTwoSided(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = st.entering ? P.c(X2_OFF(MapTwoSidedParams, front_color)) : P.c(X2_OFF(MapTwoSidedParams, back_color));
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseOpSqrt(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    out = rgb(0.0f); return tex::Ready;  // MoonRay's OpSqrtMap returns black
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseLOD(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    using Q = MapLODParams;
    const Q& q = *reinterpret_cast<const Q*>(P.p);
    // Camera distance (feature width needs the hit's ray differentials: refused at admission).
    const float distance = length(st.worldToRender ? xformPoint(st.worldToRender, st.P) : st.P);
    const float value = saturate((distance - q.start) / (q.stop - q.start));
    out = lerp(P.c(X2_OFF(Q, near_value)), P.c(X2_OFF(Q, far_value)), value);
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseNormalToRgb(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    // The input normal map's render-space normal (here world space: turned into MoonRay's render space).
    const int reg = int(reinterpret_cast<const MapNormalToRgbParams*>(P.p)->input_register);
    const V3 n = reg < 0 ? V3{0, 0, 0} : P.regs[reg];
    out = st.worldToRender ? xformVector(st.worldToRender, n) : n;
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMapCaseModoNormal(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    (void)h; (void)P; (void)st; (void)tex; (void)out;
    {
    if (isZero(length(st.dPds))) {
        out = st.N;
        return tex::Ready;
    }
    out = normalize(localToGlobal(frame(st.N, normalize(st.dPds)), P.c(X2_OFF(MapModoNormalParams, input))));
    return tex::Ready;
    }
    return tex::Invalid;
}
template <class Tex>
X2_CALL U32 sampleMap(const MapNodeHeader& h, const MapParams& P, const State& st, const Tex& tex, Color& out) {
    switch (h.kind) {
    case MapAxisAngle: return sampleMapCaseAxisAngle(h, P, st, tex, out);
    case MapTransformNormal: return sampleMapCaseTransformNormal(h, P, st, tex, out);
    case MapRgbToLab: return sampleMapCaseRgbToLab(h, P, st, tex, out);
    case MapRandom: return sampleMapCaseRandom(h, P, st, tex, out);
    case MapMultiChannelToFloat: return sampleMapCaseMultiChannelToFloat(h, P, st, tex, out);
    case MapLayer: return sampleMapCaseLayer(h, P, st, tex, out);
    case MapLayer2: return sampleMapCaseLayer2(h, P, st, tex, out);
    case MapHair: return sampleMapCaseHair(h, P, st, tex, out);
    case MapHairColumn: return sampleMapCaseHairColumn(h, P, st, tex, out);
    case MapHairColorPresets: return sampleMapCaseHairColorPresets(h, P, st, tex, out);
    case MapToon: return sampleMapCaseToon(h, P, st, tex, out);
    case MapCurvature: return sampleMapCaseCurvature(h, P, st, tex, out);
    case MapWireframe: return sampleMapCaseWireframe(h, P, st, tex, out);
    case MapDeformation: return sampleMapCaseDeformation(h, P, st, tex, out);
    case MapDirectional: return sampleMapCaseDirectional(h, P, st, tex, out);
    case MapTransformSpace: return sampleMapCaseTransformSpace(h, P, st, tex, out);
    case MapProjectTriplanarUdim: return sampleMapCaseProjectTriplanarUdim(h, P, st, tex, out);
    case MapExtraAov: return sampleMapCaseExtraAov(h, P, st, tex, out);
    case MapList: return sampleMapCaseList(h, P, st, tex, out);
    case MapDebug: return sampleMapCaseDebug(h, P, st, tex, out);
    case MapDistortNormal: return sampleMapCaseDistortNormal(h, P, st, tex, out);
    case MapRandomNormal: return sampleMapCaseRandomNormal(h, P, st, tex, out);
    case MapConstantColor: return sampleMapCaseConstantColor(h, P, st, tex, out);
    case MapConstantScalar: return sampleMapCaseConstantScalar(h, P, st, tex, out);
    case MapCheckerboard: return sampleMapCaseCheckerboard(h, P, st, tex, out);
    case MapClamp: return sampleMapCaseClamp(h, P, st, tex, out);
    case MapFloatToRgb: return sampleMapCaseFloatToRgb(h, P, st, tex, out);
    case MapRgbToFloat: return sampleMapCaseRgbToFloat(h, P, st, tex, out);
    case MapHsvToRgb: return sampleMapCaseHsvToRgb(h, P, st, tex, out);
    case MapRgbToHsv: return sampleMapCaseRgbToHsv(h, P, st, tex, out);
    case MapRemap: return sampleMapCaseRemap(h, P, st, tex, out);
    case MapBlend: return sampleMapCaseBlend(h, P, st, tex, out);
    case MapColorCorrect: return sampleMapCaseColorCorrect(h, P, st, tex, out);
    case MapCCContrast: return sampleMapCaseCCContrast(h, P, st, tex, out);
    case MapCCGainOffset: return sampleMapCaseCCGainOffset(h, P, st, tex, out);
    case MapCCGamma: return sampleMapCaseCCGamma(h, P, st, tex, out);
    case MapCCHsv: return sampleMapCaseCCHsv(h, P, st, tex, out);
    case MapCCHueShift: return sampleMapCaseCCHueShift(h, P, st, tex, out);
    case MapCCSaturation: return sampleMapCaseCCSaturation(h, P, st, tex, out);
    case MapCCTMI: return sampleMapCaseCCTMI(h, P, st, tex, out);
    case MapSwitchColor: return sampleMapCaseSwitchColor(h, P, st, tex, out);
    case MapSwitchFloat: return sampleMapCaseSwitchFloat(h, P, st, tex, out);
    case MapImage: return sampleMapCaseImage(h, P, st, tex, out);
    case MapProjectTriplanar: return sampleMapCaseProjectTriplanar(h, P, st, tex, out);
    case MapProjectTriplanarNormal: return sampleMapCaseProjectTriplanar(h, P, st, tex, out);
    case MapProjectCamera: return sampleMapCaseProjectCamera(h, P, st, tex, out);
    case MapProjectCameraNormal: return sampleMapCaseProjectCamera(h, P, st, tex, out);
    case MapProjectPlanarNormal: return sampleMapCaseProjectPlanarNormal(h, P, st, tex, out);
    case MapRgbToNormal: return sampleMapCaseRgbToNormal(h, P, st, tex, out);
    case MapColorCorrectLegacy: return sampleMapCaseColorCorrectLegacy(h, P, st, tex, out);
    case MapUVTransform: return sampleMapCaseUVTransform(h, P, st, tex, out);
    case MapSwitchNormal: return sampleMapCaseSwitchNormal(h, P, st, tex, out);
    case MapCombineNormal: return sampleMapCaseCombineNormal(h, P, st, tex, out);
    case MapImageNormal: return sampleMapCaseImageNormal(h, P, st, tex, out);
    case MapModoTexture: return sampleMapCaseModoTexture(h, P, st, tex, out);
    case MapOp: return sampleMapCaseOp(h, P, st, tex, out);
    case MapGradient: return sampleMapCaseGradient(h, P, st, tex, out);
    case MapRamp: return sampleMapCaseRamp(h, P, st, tex, out);
    case MapAttribute: return sampleMapCaseAttribute(h, P, st, tex, out);
    case MapUsdUVTexture: return sampleMapCaseUsdUVTexture(h, P, st, tex, out);
    case MapUsdTransform2d: return sampleMapCaseUsdTransform2d(h, P, st, tex, out);
    case MapUsdPrimvar: return sampleMapCaseUsdPrimvar(h, P, st, tex, out);
    case MapProjectPlanar: return sampleMapCaseProjectPlanar(h, P, st, tex, out);
    case MapProjectSpherical: return sampleMapCaseProjectSpherical(h, P, st, tex, out);
    case MapProjectCylindrical: return sampleMapCaseProjectCylindrical(h, P, st, tex, out);
    case MapNoise: return sampleMapCaseNoise(h, P, st, tex, out);
    case MapWorley2: return sampleMapCaseWorley2(h, P, st, tex, out);
    case MapWorley3: return sampleMapCaseWorley2(h, P, st, tex, out);
    case MapTwoSided: return sampleMapCaseTwoSided(h, P, st, tex, out);
    case MapOpSqrt: return sampleMapCaseOpSqrt(h, P, st, tex, out);
    case MapLOD: return sampleMapCaseLOD(h, P, st, tex, out);
    case MapNormalToRgb: return sampleMapCaseNormalToRgb(h, P, st, tex, out);
    case MapModoNormal: return sampleMapCaseModoNormal(h, P, st, tex, out);
    default: return tex::Invalid;
    }
}

// Runs maps [0, end) of a program at st, each map's result in its register. A bump map
// (ModoTextureMap's bump mode) is refused: its heights come from runMapProgram only.
template <class Tex>
X2_CALL U32 runMapPrefix(const float* pool, U32 program, U32 end, const State& st, const Tex& tex, Color* regs) {
    for (U32 i = 0; i < end; ++i) {
        const U32 node = *reinterpret_cast<const U32*>(pool + program + 1 + i);
        const MapNodeHeader& h = *reinterpret_cast<const MapNodeHeader*>(pool + node);
        if (h.reg >= MapMaxRegisters) return tex::Invalid;
        const U32 headerFloats = sizeof(MapNodeHeader) / sizeof(float);
        const MapParams P{pool + node + headerFloats,
                          reinterpret_cast<const MapBinding*>(pool + node + headerFloats + h.paramCount), h.bindingCount, regs};
        if (modoNeedsBump(h, P.p)) return tex::Invalid;
        Color out;
        const U32 status = sampleMap(h, P, st, tex, out);
        if (status != tex::Ready) return status;
        regs[h.reg] = out;
    }
    return tex::Ready;
}
// A bump map's list (after the program's required keys: the bump count, then each bump map's node, its list's
// length and the nodes its inputs need, in program order); null when the node has none.
X2_HD inline const U32* bumpList(const float* pool, U32 program, U32 node, U32& n) {
    const U32* w = reinterpret_cast<const U32*>(pool + program);
    const U32 count = w[0], required = w[1 + count];
    const U32* t = w + 2 + count + required;
    const U32 bumps = t[0];
    ++t;
    for (U32 b = 0; b < bumps; ++b) {
        const U32 at = t[0], len = t[1];
        if (at == node) {
            n = len;
            return t + 2;
        }
        t += 2 + len;
    }
    n = 0;
    return nullptr;
}
// Runs the listed maps of a program at st, each map's result in its register (none of them a bump map).
template <class Tex>
X2_CALL U32 runMapList(const float* pool, U32 program, const U32* list, U32 n, const State& st, const Tex& tex, Color* regs) {
    for (U32 k = 0; k < n; ++k) {
        const U32 node = *reinterpret_cast<const U32*>(pool + program + 1 + list[k]);
        const MapNodeHeader& h = *reinterpret_cast<const MapNodeHeader*>(pool + node);
        if (h.reg >= MapMaxRegisters) return tex::Invalid;
        const U32 headerFloats = sizeof(MapNodeHeader) / sizeof(float);
        const MapParams P{pool + node + headerFloats,
                          reinterpret_cast<const MapBinding*>(pool + node + headerFloats + h.paramCount), h.bindingCount, regs};
        if (modoNeedsBump(h, P.p)) return tex::Invalid;
        Color out;
        const U32 status = sampleMap(h, P, st, tex, out);
        if (status != tex::Ready) return status;
        regs[h.reg] = out;
    }
    return tex::Ready;
}
// Whether the hit has every primitive attribute the program's maps require (after its nodes: a count, the keys).
X2_HD inline bool mapRequirementsMet(const float* pool, U32 program, const State& st) {
    const U32 count = *reinterpret_cast<const U32*>(pool + program);
    const U32 n = *reinterpret_cast<const U32*>(pool + program + 1 + count);
    for (U32 i = 0; i < n; ++i)
        if (!primProvided(st, *reinterpret_cast<const U32*>(pool + program + 2 + count + i))) return false;
    return true;
}
// The map module's callable's place in the GPU program's binding table (async_gpu_device.h's MaterialCall).
// (An enumerator, not a constexpr: no global in every relocatable module.)
enum : U32 { MapsCallable = 2 };
// Runs a material's map program: each map in dependency order, its result in its register.
template <class Tex>
X2_CALL U32 runMapProgram(const float* pool, U32 program, const State& st, const Tex& tex, Color* regs) {
    const U32 count = *reinterpret_cast<const U32*>(pool + program);
    for (U32 i = 0; i < count; ++i) {
        const U32 node = *reinterpret_cast<const U32*>(pool + program + 1 + i);
        const MapNodeHeader& h = *reinterpret_cast<const MapNodeHeader*>(pool + node);
        if (h.reg >= MapMaxRegisters) return tex::Invalid;
        const U32 headerFloats = sizeof(MapNodeHeader) / sizeof(float);
        const MapParams P{pool + node + headerFloats,
                          reinterpret_cast<const MapBinding*>(pool + node + headerFloats + h.paramCount), h.bindingCount, regs};
        Color out;
        if (modoNeedsBump(h, P.p)) {
            // evalAttrHeight at st shifted by the bump step: the maps its inputs need at each shifted state.
            using Q = MapModoTextureParams;
            const float e = fmaxf(1.e-7f, reinterpret_cast<const Q*>(P.p)->bump_step);
            const float ds[4] = {e, -e, 0, 0}, dt[4] = {0, 0, e, -e};
            U32 listLength = 0;
            const U32* list = bumpList(pool, program, i, listLength);
            if (!list) return tex::Invalid;
            ModoBumpHeights heights;
            for (int k = 0; k < 4; ++k) {
                State other = st;
                other.s = st.s + ds[k];
                other.t = st.t + dt[k];
                Color shifted[MapMaxRegisters];
                const U32 status = runMapList(pool, program, list, listLength, other, tex, shifted);
                if (status != tex::Ready) return status;
                const MapParams Q2{P.p, P.b, P.nb, shifted};
                heights.v[k] = Q2.f(X2_OFF(Q, height));
            }
            out = sampleModoTexture(P, st, &heights);
        } else {
            const U32 status = sampleMap(h, P, st, tex, out);
            if (status != tex::Ready) return status;
        }
        regs[h.reg] = out;
    }
    return tex::Ready;
}
// The map interpreter as materials call it: on the GPU, in a program built in modules (XPU2_MAP_MODULE), the map
// module's callable (async_maps.cu); elsewhere the interpreter itself.
template <class Tex>
X2_HD inline U32 runMaps(const float* pool, U32 program, const State& st, const Tex& tex, Color* regs) {
#if defined(__CUDACC__) && defined(XPU2_MAP_MODULE)
    return optixDirectCall<U32, const float*, U32, const State*, const Tex*, Color*>(MapsCallable, pool, program, &st, &tex, regs);
#else
    return runMapProgram(pool, program, st, tex, regs);
#endif
}
}}}
