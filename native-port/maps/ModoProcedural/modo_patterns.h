// Modo's procedural textures as patterns in three dimensions, written once for MoonRay's C++ and ISPC map code and
// for MoonLightIPR's CUDA kernel, so that all three give the same numbers. Each pattern takes a point in the texture
// locator's space and gives how much of the first colour there is, 0 to 1, before the layer's bias and gain.
//
// They are after Modo's patterns and their settings, not Modo's own code: a texture the plugin's tests compare with
// Modo's renders is tuned here. Plain C that ISPC also reads: no references, no structs, no arrays; ints and floats.

#ifndef MODO_PATTERNS_H
#define MODO_PATTERNS_H

#if defined(ISPC)
#define MP_FUNC static inline
#define MP_HEAVY static inline
#define MP_FLOOR(x) floor(x)
#define MP_SQRT(x) sqrt(x)
#define MP_ABS(x) abs(x)
#define MP_COS(x) cos(x)
#define MP_SIN(x) sin(x)
#define MP_EXP(x) exp(x)
#define MP_ATAN2(y, x) atan2(y, x)
#define MP_POW(x, y) pow(x, y)
#define MP_MIN(x, y) min(x, y)
#define MP_MAX(x, y) max(x, y)
#elif defined(__CUDACC__)
#define MP_FUNC static __device__ __forceinline__
// Called rather than copied into every caller: copied, the patterns made each MoonLightIPR kernel nine times its size,
// which the graphics driver then took minutes to compile before the first picture.
#define MP_HEAVY static __device__ __noinline__
#define MP_FLOOR(x) floorf(x)
#define MP_SQRT(x) sqrtf(x)
#define MP_ABS(x) fabsf(x)
#define MP_COS(x) cosf(x)
#define MP_SIN(x) sinf(x)
#define MP_EXP(x) expf(x)
#define MP_ATAN2(y, x) atan2f(y, x)
#define MP_POW(x, y) powf(x, y)
#define MP_MIN(x, y) fminf(x, y)
#define MP_MAX(x, y) fmaxf(x, y)
#else
#include <cmath>
#define MP_FUNC static inline
#define MP_HEAVY static inline
#define MP_FLOOR(x) std::floor(x)
#define MP_SQRT(x) std::sqrt(x)
#define MP_ABS(x) std::fabs(x)
#define MP_COS(x) std::cos(x)
#define MP_SIN(x) std::sin(x)
#define MP_EXP(x) std::exp(x)
#define MP_ATAN2(y, x) std::atan2(y, x)
#define MP_POW(x, y) std::pow(x, y)
#define MP_MIN(x, y) std::fmin(x, y)
#define MP_MAX(x, y) std::fmax(x, y)
#endif

// The patterns, and the choices of those that have them, as Modo numbers its own.
#define MP_NOISE 0
#define MP_CELLULAR 1
#define MP_CHECKER 2
#define MP_GRID 3
#define MP_DOTS 4
#define MP_WOOD 5
#define MP_WEAVE 6
#define MP_RIPPLES 7
#define MP_GABOR 8
#define MP_POISSON 9
#define MP_PATTERNS 10

#define MP_PI 3.14159265358979f

MP_FUNC float mp_clamp01(float x) { return MP_MIN(1.0f, MP_MAX(0.0f, x)); }
MP_FUNC float mp_fract(float x) { return x - MP_FLOOR(x); }
MP_FUNC float mp_lerp(float a, float b, float t) { return a + (b - a) * t; }
MP_FUNC float mp_smooth(float t) { t = mp_clamp01(t); return t * t * (3.0f - 2.0f * t); }

// 0 below an edge and 1 above it, easing between over soft either side.
MP_FUNC float mp_step(float value, float edge, float soft) {
    if (soft <= 1e-6f) return value >= edge ? 1.0f : 0.0f;
    return mp_smooth((value - edge + soft) / (2.0f * soft));
}

// A whole number from whole numbers, always the same for the same ones; and from it a number from 0 up to 1.
MP_FUNC unsigned int mp_hash(int x, int y, int z, int seed) {
    unsigned int h = ((unsigned int)x * 73856093u) ^ ((unsigned int)y * 19349663u) ^ ((unsigned int)z * 83492791u)
                     ^ ((unsigned int)seed * 2654435761u);
    h = h ^ (h >> 16);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16);
    return h;
}
MP_FUNC float mp_random(int x, int y, int z, int seed) { return (float)(mp_hash(x, y, z, seed) & 0xffffffu) / 16777216.0f; }

// Smooth gradient noise, about -1 to 1, as Perlin's improved noise.
MP_FUNC float mp_fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
MP_FUNC float mp_gradient(unsigned int h, float x, float y, float z) {
    int k = (int)(h & 15u);
    float u = k < 8 ? x : y;
    float v = k < 4 ? y : ((k == 12 || k == 14) ? x : z);
    return (((k & 1) != 0) ? -u : u) + (((k & 2) != 0) ? -v : v);
}
MP_HEAVY float mp_noise(float x, float y, float z, int seed) {
    float fx = MP_FLOOR(x), fy = MP_FLOOR(y), fz = MP_FLOOR(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    x = x - fx; y = y - fy; z = z - fz;
    float u = mp_fade(x), v = mp_fade(y), w = mp_fade(z);
    float a = mp_lerp(mp_gradient(mp_hash(ix, iy, iz, seed), x, y, z), mp_gradient(mp_hash(ix + 1, iy, iz, seed), x - 1.0f, y, z), u);
    float b = mp_lerp(mp_gradient(mp_hash(ix, iy + 1, iz, seed), x, y - 1.0f, z), mp_gradient(mp_hash(ix + 1, iy + 1, iz, seed), x - 1.0f, y - 1.0f, z), u);
    float c = mp_lerp(mp_gradient(mp_hash(ix, iy, iz + 1, seed), x, y, z - 1.0f), mp_gradient(mp_hash(ix + 1, iy, iz + 1, seed), x - 1.0f, y, z - 1.0f), u);
    float d = mp_lerp(mp_gradient(mp_hash(ix, iy + 1, iz + 1, seed), x, y - 1.0f, z - 1.0f),
                      mp_gradient(mp_hash(ix + 1, iy + 1, iz + 1, seed), x - 1.0f, y - 1.0f, z - 1.0f), u);
    return mp_lerp(mp_lerp(a, b, v), mp_lerp(c, d, v), w);
}

// Noise over several frequencies: each ratio times finer and ampRatio times weaker. turbulent adds their sizes.
MP_HEAVY float mp_fractal(float x, float y, float z, int octaves, float ratio, float ampRatio, int turbulent, int seed) {
    float sum = 0.0f, amplitude = 1.0f, weight = 0.0f, frequency = 1.0f;
    for (int i = 0; i < octaves; i++) {
        float n = mp_noise(x * frequency, y * frequency, z * frequency, seed + i * 7);
        sum = sum + amplitude * (turbulent != 0 ? MP_ABS(n) : n);
        weight = weight + amplitude;
        amplitude = amplitude * ampRatio;
        frequency = frequency * ratio;
    }
    return weight > 0.0f ? sum / weight : 0.0f;
}

// Modo's Noise: simple (type 0), fractal (1) or turbulence (2).
MP_HEAVY float mp_noise_pattern(int type, float x, float y, float z, float freqs, float freqRatio, float ampRatio) {
    int octaves = type == 0 ? 1 : (int)MP_MAX(1.0f, MP_MIN(16.0f, freqs));
    float ratio = MP_MAX(1.0f, freqRatio), weaker = MP_MAX(0.0f, ampRatio);
    if (type == 2) return mp_clamp01(1.6f * mp_fractal(x, y, z, octaves, ratio, weaker, 1, 11));
    return mp_clamp01(0.5f + 0.5f * 1.4f * mp_fractal(x, y, z, octaves, ratio, weaker, 0, 11));
}

// One frequency of Modo's Cellular, a cell a unit across: a point in each cell, round (type 0) within the cell's
// width of its point, angular (1) further than the gap from the cell's edge, each cell darkened by chance by variation.
MP_HEAVY float mp_cells(float x, float y, float z, int round, float width, float soft, float variation, int seed) {
    float cx = MP_FLOOR(x), cy = MP_FLOOR(y), cz = MP_FLOOR(z);
    float first = 9.0f, second = 9.0f;
    int nx = 0, ny = 0, nz = 0;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i, b = (int)cy + j, c = (int)cz + k;
                float px = (float)a + 0.15f + 0.7f * mp_random(a, b, c, seed);
                float py = (float)b + 0.15f + 0.7f * mp_random(a, b, c, seed + 1);
                float pz = (float)c + 0.15f + 0.7f * mp_random(a, b, c, seed + 2);
                float d = MP_SQRT((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
                if (d < first) { second = first; first = d; nx = a; ny = b; nz = c; }
                else if (d < second) second = d;
            }
        }
    }
    float inside = round != 0 ? mp_step(0.5f * width - first, 0.0f, soft) : mp_step(0.5f * (second - first), 0.5f * (1.0f - width), soft);
    return inside * (1.0f - mp_clamp01(variation) * mp_random(nx, ny, nz, seed + 13));
}
MP_HEAVY float mp_cellular(int type, float x, float y, float z, float cellWidth, float transWidth, float freqs, float freqRatio,
                          float ampRatio, float variation) {
    float width = mp_clamp01(cellWidth), soft = 0.25f * MP_MAX(0.0f, transWidth);
    float total = 0.0f, weight = 0.0f, amplitude = 1.0f, frequency = 1.0f;
    int octaves = (int)MP_MAX(1.0f, MP_MIN(8.0f, freqs));
    for (int i = 0; i < octaves; i++) {
        total = total + amplitude * mp_cells(x * frequency, y * frequency, z * frequency, type == 0 ? 1 : 0, width, soft, variation, 7 * i);
        weight = weight + amplitude;
        amplitude = amplitude * MP_MAX(0.0f, ampRatio);
        frequency = frequency * MP_MAX(1.0f, freqRatio);
    }
    return weight > 0.0f ? total / weight : 0.0f;
}

// Modo's Checker, a square a unit across: squares across the locator's X and Y (type 0) or cubes (1); its colours
// meet over transWidth of a square.
MP_FUNC float mp_checker(int type, float x, float y, float z, float transWidth) {
    float fx = mp_fract(x), fy = mp_fract(y), fz = mp_fract(z);
    int parity = ((int)MP_FLOOR(x) + (int)MP_FLOOR(y) + (type == 1 ? (int)MP_FLOOR(z) : 0)) & 1;
    float edge = MP_MIN(MP_MIN(fx, 1.0f - fx), MP_MIN(fy, 1.0f - fy));
    if (type == 1) edge = MP_MIN(edge, MP_MIN(fz, 1.0f - fz));
    float soft = 0.5f * MP_MAX(0.0f, transWidth);
    float sure = soft > 1e-6f ? mp_smooth(edge / soft) : 1.0f;
    return 0.5f + (parity != 0 ? 0.5f : -0.5f) * sure;
}

// How far a point is from the lines of Modo's Grid (line 0, triangle 1, square 2, hexagon 3, cube 4), in cells.
MP_FUNC float mp_grid_distance(int type, float x, float y, float z) {
    float fx = mp_fract(x), fy = mp_fract(y), fz = mp_fract(z);
    if (type == 0) return MP_MIN(fx, 1.0f - fx);
    if (type == 2) return MP_MIN(MP_MIN(fx, 1.0f - fx), MP_MIN(fy, 1.0f - fy));
    if (type == 4) return MP_MIN(MP_MIN(MP_MIN(fx, 1.0f - fx), MP_MIN(fy, 1.0f - fy)), MP_MIN(fz, 1.0f - fz));
    // Rows on a triangular lattice, each half a cell over from the last.
    float ry = y * 2.0f;
    float row = MP_FLOOR(ry);
    float ty = ry - row, tx = mp_fract(x - 0.5f * mp_fract(row * 0.5f) * 2.0f);
    if (type == 1)
        return MP_MIN(MP_MIN(ty, 1.0f - ty) * 0.5f, MP_MIN(MP_ABS(mp_fract(tx + ty * 0.5f) - 0.5f), MP_ABS(mp_fract(tx - ty * 0.5f) - 0.5f)));
    float dx = MP_ABS(tx - 0.5f), dy = MP_ABS(ty - 0.5f);
    return MP_ABS(MP_MAX(dx, dx * 0.5f + dy * 0.75f) - 0.5f);
}
MP_HEAVY float mp_grid(int type, float x, float y, float z, float lineWidth, float transWidth) {
    float half = 0.5f * mp_clamp01(lineWidth), soft = 0.5f * MP_MAX(0.0f, transWidth);
    return 1.0f - mp_step(mp_grid_distance(type, x, y, z), half, soft);
}

// Modo's Dots: on squares (type 0), a hexagonal lattice (1) or cubes (2), dotWidth of a cell across.
MP_HEAVY float mp_dots(int type, float x, float y, float z, float dotWidth, float transWidth) {
    float d;
    if (type == 1) {
        float ry = y * 2.0f;
    float row = MP_FLOOR(ry);
        float ty = ry - row, tx = mp_fract(x - 0.5f * mp_fract(row * 0.5f) * 2.0f);
        d = MP_SQRT((tx - 0.5f) * (tx - 0.5f) + (ty - 0.5f) * (ty - 0.5f));
    } else {
        float dx = mp_fract(x) - 0.5f, dy = mp_fract(y) - 0.5f, dz = type == 2 ? mp_fract(z) - 0.5f : 0.0f;
        d = MP_SQRT(dx * dx + dy * dy + dz * dz);
    }
    float radius = 0.5f * mp_clamp01(dotWidth), soft = 0.5f * MP_MAX(0.0f, transWidth);
    return 1.0f - mp_step(d, radius, soft);
}

// Modo's Wood: rings round the locator's Y axis, ringScale apart, bent by a slow wander and roughened by a grain
// that runs along Y. Dark at the start of a ring, easing to light across it, as a year's growth does.
MP_HEAVY float mp_wood(float x, float y, float z, float ringScale, float wave, float waveScale, float grain, float grainScale) {
    float scale = MP_MAX(0.001f, waveScale), fine = MP_MAX(0.001f, grainScale);
    float wander = mp_noise(x / scale, y / scale, z / scale, 3) * MP_MAX(0.0f, wave);
    float ring = mp_fract(MP_SQRT(x * x + z * z) / MP_MAX(0.001f, ringScale) + wander);
    float t = 1.0f - MP_POW(ring, 0.6f);
    t = t + mp_noise(x / fine, y / (fine * 8.0f), z / fine, 5) * MP_MAX(0.0f, grain) * 0.6f;
    return mp_clamp01(t);
}

// Modo's Weave: yarns a unit apart across the locator's X and Y, over and under by turns, highest along their middle
// and where they pass over.
MP_HEAVY float mp_weave(float x, float y, float yarnWidth, float roundness) {
    float cx = MP_FLOOR(x), cy = MP_FLOOR(y);
    float fx = x - cx, fy = y - cy;
    float width = MP_MAX(0.05f, mp_clamp01(yarnWidth));
    int across = (((int)cx + (int)cy) & 1) == 0 ? 1 : 0;
    float along = across != 0 ? fx : fy, side = across != 0 ? fy : fx;
    float off = MP_ABS(side - 0.5f);
    if (off > 0.5f * width) return 0.0f;
    float profile = MP_COS(MP_PI * off / width), arch = MP_COS(MP_PI * (along - 0.5f));
    float round = mp_clamp01(roundness);
    return mp_clamp01((1.0f - round) + round * profile * (0.5f + 0.5f * arch));
}

// Modo's Ripples: rings spreading from sources scattered through each unit cube, their crests added up.
MP_HEAVY float mp_ripples(float x, float y, float z, float sources, float wavelength, float phase) {
    int count = (int)MP_MAX(1.0f, MP_MIN(64.0f, sources));
    float length = MP_MAX(0.01f, wavelength), total = 0.0f;
    float fx = mp_fract(x), fy = mp_fract(y), fz = mp_fract(z);
    for (int i = 0; i < count; i++) {
        float dx = MP_ABS(fx - mp_random(i, 17, 0, 0)), dy = MP_ABS(fy - mp_random(i, 29, 0, 0)), dz = MP_ABS(fz - mp_random(i, 43, 0, 0));
        // The nearer way round, so that the rings meet themselves across the cube's faces.
        dx = MP_MIN(dx, 1.0f - dx); dy = MP_MIN(dy, 1.0f - dy); dz = MP_MIN(dz, 1.0f - dz);
        total = total + MP_COS(2.0f * MP_PI * (MP_SQRT(dx * dx + dy * dy + dz * dz) / length - phase));
    }
    return mp_clamp01(0.5f + 0.5f * total / MP_SQRT((float)count));
}

// A Modo texture layer's bias and gain, 0.5 leaving the value as it is.
MP_FUNC float mp_bias(float x, float b) {
    b = MP_MIN(0.99999f, MP_MAX(0.00001f, b));
    x = mp_clamp01(x);
    return x / ((1.0f / b - 2.0f) * (1.0f - x) + 1.0f);
}
MP_FUNC float mp_gain(float t, float g) {
    return t < 0.5f ? 0.5f * mp_bias(2.0f * t, 1.0f - g) : 1.0f - 0.5f * mp_bias(2.0f - 2.0f * t, 1.0f - g);
}

// ---- Enhance: Modo's patterns ----------------------------------------------------------------------------------
// Their Noise Type: Perlin 0, EnhancedPerlin 1, Gradient 2, Value 3, GradientValue 4, Impulse 5, Lattice 6, Bubble 7.

// Smooth value noise, about -1 to 1: a random number at each whole point, eased between.
MP_HEAVY float mp_value_noise(float x, float y, float z, int seed) {
    float fx = MP_FLOOR(x), fy = MP_FLOOR(y), fz = MP_FLOOR(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    float u = mp_smooth(x - fx), v = mp_smooth(y - fy), w = mp_smooth(z - fz);
    float a = mp_lerp(mp_random(ix, iy, iz, seed), mp_random(ix + 1, iy, iz, seed), u);
    float b = mp_lerp(mp_random(ix, iy + 1, iz, seed), mp_random(ix + 1, iy + 1, iz, seed), u);
    float c = mp_lerp(mp_random(ix, iy, iz + 1, seed), mp_random(ix + 1, iy, iz + 1, seed), u);
    float d = mp_lerp(mp_random(ix, iy + 1, iz + 1, seed), mp_random(ix + 1, iy + 1, iz + 1, seed), u);
    return 2.0f * mp_lerp(mp_lerp(a, b, v), mp_lerp(c, d, v), w) - 1.0f;
}

// Sparse convolution: a few impulses of random strength scattered through each cell, each a smooth bump.
MP_HEAVY float mp_impulse_noise(float x, float y, float z, int seed) {
    float cx = MP_FLOOR(x), cy = MP_FLOOR(y), cz = MP_FLOOR(z), total = 0.0f;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i, b = (int)cy + j, c = (int)cz + k;
                float dx = x - ((float)a + mp_random(a, b, c, seed));
                float dy = y - ((float)b + mp_random(a, b, c, seed + 1));
                float dz = z - ((float)c + mp_random(a, b, c, seed + 2));
                float d = dx * dx + dy * dy + dz * dz;
                if (d < 1.0f) total = total + (2.0f * mp_random(a, b, c, seed + 3) - 1.0f) * (1.0f - d) * (1.0f - d);
            }
        }
    }
    return MP_MAX(-1.0f, MP_MIN(1.0f, total));
}

// Bubbles: high at the middle of each cell's point, falling to the cells' edges.
MP_HEAVY float mp_bubble_noise(float x, float y, float z, int seed) {
    float cx = MP_FLOOR(x), cy = MP_FLOOR(y), cz = MP_FLOOR(z), first = 9.0f;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i, b = (int)cy + j, c = (int)cz + k;
                float dx = x - ((float)a + mp_random(a, b, c, seed)), dy = y - ((float)b + mp_random(a, b, c, seed + 1));
                float dz = z - ((float)c + mp_random(a, b, c, seed + 2));
                first = MP_MIN(first, dx * dx + dy * dy + dz * dz);
            }
        }
    }
    return 1.0f - 2.0f * mp_clamp01(MP_SQRT(first));
}

// One of Enhance: Modo's noise bases, about -1 to 1.
MP_HEAVY float mp_basis(int type, float x, float y, float z, int seed) {
    if (type == 3 || type == 6) return mp_value_noise(x, y, z, seed);
    if (type == 4) return 0.5f * (mp_noise(x, y, z, seed) + mp_value_noise(x, y, z, seed + 101));
    if (type == 5) return mp_impulse_noise(x, y, z, seed);
    if (type == 7) return mp_bubble_noise(x, y, z, seed);
    if (type == 1) return 1.15f * mp_noise(x * 1.07f, y * 1.07f, z * 1.07f, seed + 31);
    return mp_noise(x, y, z, seed);
}

// Fractal sums, as Musgrave writes them: octaves of the basis, each lacunarity times finer and lacunarity to the -h
// weaker. absolute adds the bases' sizes. Returned about -1 to 1 (0 to 1 with absolute).
MP_HEAVY float mp_fbm(int type, float x, float y, float z, int seed, float octaves, float lacunarity, float h, int absolute) {
    float sum = 0.0f, weight = 0.0f, frequency = 1.0f;
    float lac = MP_MAX(1.01f, lacunarity);
    int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, octaves));
    for (int i = 0; i < count; i++) {
        float amplitude = MP_POW(frequency, -h);
        float n = mp_basis(type, x * frequency, y * frequency, z * frequency, seed + 13 * i);
        sum = sum + amplitude * (absolute != 0 ? MP_ABS(n) : n);
        weight = weight + amplitude;
        frequency = frequency * lac;
    }
    return weight > 0.0f ? sum / weight : 0.0f;
}
MP_FUNC float mp_unit(float signedValue) { return mp_clamp01(0.5f + 0.5f * signedValue); }
// A value brought from lower .. upper to 0 .. 1, as Enhance: Modo's Lower and Upper do.
MP_FUNC float mp_range(float v, float lower, float upper) {
    float span = upper - lower;
    if (MP_ABS(span) < 1e-6f) return v >= lower ? 1.0f : 0.0f;
    return mp_clamp01((v - lower) / span);
}

typedef struct {
    float v[24];    // a pattern's settings from v[0], in the order procedurals.py gives them; v[22] bias, v[23] gain
} MP_Settings;

// Enhance: Modo's Noise patterns, 100 on, in the order of their names. Each reads its settings as listed beside it
// (procedurals.ENHANCE has the same lists), its Noise Type being type.
MP_HEAVY float mp_enhance_noise(int pattern, int type, float x, float y, float z, MP_Settings s) {
    int seed = (int)s.v[0];
    float t = 0.0f;
    if (pattern == 100) {
        // Agate: seed, absolute, lower, upper. Bands round the z axis, troubled by turbulence.
        float turb = mp_fbm(type, x, y, z, seed, 6.0f, 2.0f, 0.5f, 1);
        t = mp_range(0.5f + 0.5f * MP_COS(2.0f * MP_PI * (z * 1.5f + 3.0f * turb)), s.v[2], s.v[3]);
        if (s.v[1] != 0.0f) t = MP_ABS(2.0f * t - 1.0f);
    } else if (pattern == 101 || pattern == 121) {
        // Bozo and fBm: seed, octaves, frequency, increment, absolute, lower, upper.
        float f = mp_fbm(type, x, y, z, seed, s.v[1], s.v[2], s.v[3], s.v[4] != 0.0f ? 1 : 0);
        t = mp_range(s.v[4] != 0.0f ? f : mp_unit(f), s.v[5], s.v[6]);
    } else if (pattern == 102) {
        // Cruddy: seed, frequency, magnitude, lower, upper. Noise moved about by noise.
        float m = MP_MAX(0.0f, s.v[2]), f = MP_MAX(0.01f, s.v[1]);
        float w = mp_basis(type, x * f, y * f, z * f, seed + 7) * m;
        t = mp_range(mp_unit(mp_fbm(type, x * f + w, y * f - w, z * f + w, seed, 4.0f, 2.0f, 0.5f, 0)), s.v[3], s.v[4]);
    } else if (pattern == 103 || pattern == 104) {
        // Dented and Etched: seed, octaves, dentedness or corrosion, lower, upper. Pits of the sizes of the octaves.
        float d = mp_fbm(type, x, y, z, seed, s.v[1], 2.0f, 1.0f, 1);
        float power = MP_MAX(0.1f, s.v[2]) * (pattern == 104 ? 1.5f : 1.0f);
        t = mp_range(1.0f - MP_POW(mp_clamp01(d * 1.6f), power), s.v[3], s.v[4]);
    } else if (pattern == 105) {
        // FlowBozo (type is its Flow Type): seed, octaves, advection, time, absolute, lower, upper. Each octave
        // carried along by the coarser ones as time goes on.
        float sum = 0.0f, weight = 0.0f, frequency = 1.0f, ax = 0.0f, ay = 0.0f, az = 0.0f;
        int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, s.v[1]));
        for (int i = 0; i < count; i++) {
            float amplitude = 1.0f / frequency;
            float n = mp_noise(x * frequency + ax + s.v[3], y * frequency + ay, z * frequency + az - s.v[3], seed + 13 * i);
            if (type == 1) n = n * MP_ABS(n);
            if (type == 2) n = -n;
            sum = sum + amplitude * (s.v[4] != 0.0f ? MP_ABS(n) : n);
            weight = weight + amplitude;
            ax = ax + s.v[2] * amplitude * n; ay = ay - s.v[2] * amplitude * n; az = az + 0.5f * s.v[2] * amplitude * n;
            frequency = frequency * 2.0f;
        }
        float f = weight > 0.0f ? sum / weight : 0.0f;
        t = mp_range(s.v[4] != 0.0f ? f : mp_unit(f), s.v[5], s.v[6]);
    } else if (pattern == 106) {
        // Granite: seed, octaves, frequency, fold, lower, upper. Gritz's: |0.5 - noise| over the octaves.
        float sum = 0.0f, frequency = 1.0f;
        int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, s.v[1]));
        for (int i = 0; i < count; i++) {
            sum = sum + MP_ABS(0.5f - mp_unit(mp_basis(type, x * frequency * 4.0f, y * frequency * 4.0f, z * frequency * 4.0f, seed + 13 * i))) / frequency;
            frequency = frequency * MP_MAX(1.01f, s.v[2]);
        }
        if (s.v[3] != 0.0f) sum = MP_ABS(2.0f * mp_fract(sum) - 1.0f);
        t = mp_range(mp_clamp01(sum), s.v[4], s.v[5]);
    } else if (pattern == 107 || pattern == 113) {
        // Hybrid: seed, octaves, frequency, increment, offset, heterogeneous, lower, upper. PuffyClouds: seed,
        // octaves, frequency, increment, coverage, heterogeneous, lower, upper. Musgrave's hybrid multifractal.
        float lac = MP_MAX(1.01f, s.v[2]), offset = pattern == 107 ? s.v[4] : 0.7f, frequency = 1.0f;
        float result = mp_basis(type, x, y, z, seed) + offset;
        float weight = result;
        int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, s.v[1]));
        for (int i = 1; i < count; i++) {
            frequency = frequency * lac;
            weight = MP_MIN(1.0f, weight);
            float signal = (mp_basis(type, x * frequency, y * frequency, z * frequency, seed + 13 * i) + offset) * MP_POW(frequency, -s.v[3]);
            result = result + (s.v[5] != 0.0f ? 1.0f : weight) * signal;
            weight = weight * signal;
        }
        float v = mp_clamp01(result / (1.0f + offset + 0.5f * (float)(count - 1)));
        if (pattern == 113) v = mp_clamp01((v - (1.0f - mp_clamp01(s.v[4]))) / MP_MAX(0.05f, mp_clamp01(s.v[4])));
        t = mp_range(v, s.v[6], s.v[7]);
    } else if (pattern == 108) {
        // Lump: seed, octaves, frequency, lower, upper. Rounded lumps: the sizes of the octaves.
        t = mp_range(mp_clamp01(1.5f * mp_fbm(type, x, y, z, seed, s.v[1], s.v[2], 0.6f, 1)), s.v[3], s.v[4]);
    } else if (pattern == 109 || pattern == 110) {
        // MarbleNoise: seed, octaves, frequency, increment, lower, upper. MarbleVein: those and sharpness before
        // lower and upper. Bands across x, bent by turbulence; the veins are where the bands turn.
        float turb = mp_fbm(type, x, y, z, seed, s.v[1], s.v[2], s.v[3], 1);
        float band = 0.5f + 0.5f * MP_COS(MP_PI * (x * 2.0f + 6.0f * turb));
        if (pattern == 110) {
            float sharp = MP_MIN(0.98f, MP_MAX(0.0f, s.v[4]));
            band = MP_POW(1.0f - MP_ABS(2.0f * band - 1.0f), 1.0f / (1.0f - sharp));
            t = mp_range(band, s.v[5], s.v[6]);
        } else t = mp_range(band, s.v[4], s.v[5]);
    } else if (pattern == 111) {
        // MultiFractal: seed, octaves, frequency, increment, offset, lower, upper. Musgrave's multiplicative cascade.
        float value = 1.0f, frequency = 1.0f;
        int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, s.v[1]));
        for (int i = 0; i < count; i++) {
            value = value * (s.v[4] + mp_basis(type, x * frequency, y * frequency, z * frequency, seed + 13 * i) * MP_POW(frequency, -s.v[3]));
            frequency = frequency * MP_MAX(1.01f, s.v[2]);
        }
        t = mp_range(mp_clamp01(value), s.v[5], s.v[6]);
    } else if (pattern == 112) {
        // Pebbles: seed, min freq, max freq, steps, magnitude, frequency, lower, upper. Stones of sizes between
        // the two frequencies, rounded and jostled by noise.
        float jostle = mp_basis(type, x * s.v[5], y * s.v[5], z * s.v[5], seed + 3) * 0.3f * MP_MAX(0.0f, s.v[4]);
        float low = MP_MAX(0.01f, s.v[1]);
        float high = MP_MAX(low, s.v[2]), best = 0.0f;
        int steps = (int)MP_MAX(1.0f, MP_MIN(6.0f, s.v[3]));
        for (int i = 0; i < steps; i++) {
            float f = steps > 1 ? low + (high - low) * (float)i / (float)(steps - 1) : low;
            float edge = 0.5f + 0.5f * mp_bubble_noise((x + jostle) * f, (y - jostle) * f, (z + jostle) * f, seed + 17 * i);
            best = MP_MAX(best, MP_POW(mp_clamp01(edge), 0.5f));
        }
        t = mp_range(best, s.v[6], s.v[7]);
    } else if (pattern == 114) {
        // Ridged: seed, octaves, frequency, increment, offset, threshold, lower, upper. Musgrave's ridged multifractal.
        float frequency = 1.0f, signal = s.v[4] - MP_ABS(mp_basis(type, x, y, z, seed));
        signal = signal * signal;
        float result = signal;
        int count = (int)MP_MAX(1.0f, MP_MIN(12.0f, s.v[1]));
        for (int i = 1; i < count; i++) {
            frequency = frequency * MP_MAX(1.01f, s.v[2]);
            float weight = mp_clamp01(signal * s.v[5]);
            signal = s.v[4] - MP_ABS(mp_basis(type, x * frequency, y * frequency, z * frequency, seed + 13 * i));
            signal = signal * signal * weight;
            result = result + signal * MP_POW(frequency, -s.v[3]);
        }
        t = mp_range(mp_clamp01(result / MP_MAX(0.1f, s.v[4] * s.v[4] * 1.5f)), s.v[6], s.v[7]);
    } else if (pattern == 115) {
        // Scar: seed, octaves, frequency, increment, lower, upper. Thin lines where the noise crosses its middle.
        float f = mp_fbm(type, x, y, z, seed, s.v[1], MP_MAX(1.01f, s.v[2]) * 1.0f + 1.0f, s.v[3], 0);
        t = mp_range(MP_POW(1.0f - mp_clamp01(MP_ABS(f) * 4.0f), 6.0f), s.v[4], s.v[5]);
    } else if (pattern == 116) {
        // Scruffed: seed, length, coverage, axis (X 0, Y 1, Z 2), lower, upper. Scratches drawn out along the axis.
        float stretch = 1.0f / MP_MAX(1.0f, s.v[1]);
        int axis = (int)s.v[3];
        float sx = axis == 0 ? x * stretch : x * 8.0f, sy = axis == 1 ? y * stretch : y * 8.0f, sz = axis == 2 ? z * stretch : z * 8.0f;
        float n = mp_unit(mp_fbm(type, sx, sy, sz, seed, 3.0f, 2.0f, 0.5f, 0));
        t = mp_range(mp_step(n, 1.0f - 0.5f * mp_clamp01(s.v[2]), 0.02f), s.v[4], s.v[5]);
    } else if (pattern == 117) {
        // Strata: seed, octaves, frequency, increment, width, magnitude, axis, lower, upper. Layers across the axis,
        // bent by noise, width of each band filled.
        int axis = (int)s.v[6];
        float along = axis == 0 ? x : (axis == 1 ? y : z);
        float bend = mp_fbm(type, x, y, z, seed, s.v[1], s.v[2], s.v[3], 0) * s.v[5];
        float layer = mp_fract(along * 4.0f + bend);
        t = mp_range(mp_step(mp_clamp01(s.v[4]), layer, 0.05f), s.v[7], s.v[8]);
    } else if (pattern == 118) {
        // Stucco: seed, octaves, frequency, increment, power, lower, upper. Sharp plaster bumps.
        float f = mp_unit(mp_fbm(type, x * 4.0f, y * 4.0f, z * 4.0f, seed, s.v[1], s.v[2], s.v[3], 0));
        t = mp_range(MP_POW(f, MP_MAX(0.1f, s.v[4]) * 0.25f), s.v[5], s.v[6]);
    } else if (pattern == 119 || pattern == 120) {
        // VectorBozo: seed, octaves, frequency, magnitude, fractalize, lower, upper. WrappedfBm: seed, octaves,
        // frequency, magnitude, lower, upper. Noise looked up where noise moves the point.
        float m = MP_MAX(0.0f, s.v[3]);
        int fractal = pattern == 120 || s.v[4] != 0.0f ? 1 : 0;
        float wx = mp_fbm(type, x + 5.2f, y + 1.3f, z, seed + 1, fractal != 0 ? s.v[1] : 1.0f, MP_MAX(1.01f, s.v[2]), 0.5f, 0) * m;
        float wy = mp_fbm(type, x, y + 9.2f, z + 2.8f, seed + 2, fractal != 0 ? s.v[1] : 1.0f, MP_MAX(1.01f, s.v[2]), 0.5f, 0) * m;
        float wz = mp_fbm(type, x + 3.7f, y, z + 7.1f, seed + 3, fractal != 0 ? s.v[1] : 1.0f, MP_MAX(1.01f, s.v[2]), 0.5f, 0) * m;
        float f = mp_fbm(type, x + wx, y + wy, z + wz, seed, pattern == 120 ? s.v[1] : 1.0f, MP_MAX(1.01f, s.v[2]), 0.5f, 0);
        t = mp_range(mp_unit(f), pattern == 120 ? s.v[4] : s.v[5], pattern == 120 ? s.v[5] : s.v[6]);
    }
    return t;
}

// Cells: a point scattered in each unit cube, jitter of the cube from its middle. what is 0 to 3 for the distance to
// the nearest to the fourth nearest point, 4 for the second less the first (0 on the cells' edges), 5 for the nearest
// cell's own random number.
MP_HEAVY float mp_worley(float x, float y, float z, int seed, float jitter, int what) {
    float cx = MP_FLOOR(x), cy = MP_FLOOR(y), cz = MP_FLOOR(z);
    float d1 = 9.0f, d2 = 9.0f, d3 = 9.0f, d4 = 9.0f;
    int nx = 0, ny = 0, nz = 0;
    float spread = mp_clamp01(jitter);
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i, b = (int)cy + j, c = (int)cz + k;
                float px = (float)a + 0.5f + spread * (mp_random(a, b, c, seed) - 0.5f);
                float py = (float)b + 0.5f + spread * (mp_random(a, b, c, seed + 1) - 0.5f);
                float pz = (float)c + 0.5f + spread * (mp_random(a, b, c, seed + 2) - 0.5f);
                float d = MP_SQRT((x - px) * (x - px) + (y - py) * (y - py) + (z - pz) * (z - pz));
                if (d < d1) { d4 = d3; d3 = d2; d2 = d1; d1 = d; nx = a; ny = b; nz = c; }
                else if (d < d2) { d4 = d3; d3 = d2; d2 = d; }
                else if (d < d3) { d4 = d3; d3 = d; }
                else if (d < d4) d4 = d;
            }
        }
    }
    if (what == 0) return d1;
    if (what == 1) return d2;
    if (what == 2) return d3;
    if (what == 3) return d4;
    if (what == 4) return d2 - d1;
    return mp_random(nx, ny, nz, seed + 5);
}

// Cells over several frequencies, each lacunarity times finer and lacunarity to the -h weaker.
MP_HEAVY float mp_worley_fractal(float x, float y, float z, int seed, float jitter, int what, float octaves, float lacunarity, float h) {
    float sum = 0.0f, weight = 0.0f, frequency = 1.0f;
    int count = (int)MP_MAX(1.0f, MP_MIN(8.0f, octaves));
    for (int i = 0; i < count; i++) {
        float amplitude = MP_POW(frequency, -h);
        sum = sum + amplitude * mp_worley(x * frequency, y * frequency, z * frequency, seed + 19 * i, jitter, what);
        weight = weight + amplitude;
        frequency = frequency * MP_MAX(1.01f, lacunarity);
    }
    return weight > 0.0f ? sum / weight : 0.0f;
}

// Enhance: Modo's Organic patterns, 130 on, in the order of their names; the settings each reads are beside it
// (procedurals.ENHANCE). type is its Noise Type, Cell Order (F1 0 to F4 3) or Shape, as it has.
MP_HEAVY float mp_enhance_organic(int pattern, int type, float x, float y, float z, MP_Settings s) {
    int seed = (int)s.v[0];
    float t = 0.0f;
    if (pattern == 130) {
        // ArtDeco: seed, octaves, frequency, increment, power, lower, upper, regions. Cells in sweeping bands.
        float f = s.v[7] != 0.0f ? mp_worley(x, y, z, seed, 1.0f, 5) : mp_worley_fractal(x, y, z, seed, 1.0f, 0, MP_MAX(1.0f, s.v[1]), s.v[2], s.v[3]);
        t = mp_range(MP_POW(mp_clamp01(f), MP_MAX(0.05f, s.v[4])), s.v[5], s.v[6]);
    } else if (pattern == 131) {
        // Blister: levels, lower, upper. Raised bubbles of several sizes.
        float best = 0.0f, scale = 1.0f;
        int count = (int)MP_MAX(1.0f, MP_MIN(6.0f, s.v[0]));
        for (int i = 0; i < count; i++) {
            best = MP_MAX(best, 0.5f + 0.5f * mp_bubble_noise(x * scale, y * scale, z * scale, 41 + 7 * i));
            scale = scale * 2.0f;
        }
        t = mp_range(best, s.v[1], s.v[2]);
    } else if (pattern == 132) {
        // Branches: branches, width, seed, octaves, magnitude, scale, detail, full 3D, lower, upper. Branching lines
        // where the noise crosses its middle, as many as branches.
        int bseed = (int)s.v[2];
        float sc = MP_MAX(0.01f, s.v[5]);
        float bz = s.v[7] != 0.0f ? z * sc : 0.0f;
        float n = mp_fbm(type, x * sc * MP_MAX(1.0f, s.v[0]) * 0.5f, y * sc * MP_MAX(1.0f, s.v[0]) * 0.5f, bz, bseed, s.v[3], 2.0f, 1.0f - 0.5f * mp_clamp01(s.v[6]), 0);
        n = n + s.v[4] * 0.2f * mp_basis(type, x * sc * 4.0f, y * sc * 4.0f, bz * 4.0f, bseed + 9);
        t = mp_range(1.0f - mp_clamp01(MP_ABS(n) / MP_MAX(0.005f, s.v[1] * 0.25f)), s.v[8], s.v[9]);
    } else if (pattern == 133) {
        // Caustic: seed, octaves, frequency, increment, lower, upper. Bright light gathered into a net.
        float wx = mp_fbm(type, x, y, z, seed + 1, s.v[1], s.v[2], s.v[3], 0) * 0.4f;
        float e = mp_worley(x * 2.0f + wx, y * 2.0f - wx, z * 2.0f + wx, seed, 1.0f, 4);
        t = mp_range(MP_POW(1.0f - mp_clamp01(e * 3.0f), 4.0f), s.v[4], s.v[5]);
    } else if (pattern == 134 || pattern == 135 || pattern == 137) {
        // Cellular: seed, octaves, frequency, increment, fractalize, lower, upper, regions. Cheesy: seed, octaves,
        // frequency, increment, lower, upper, regions. Crackle: as Cellular. type is the Cell Order.
        int fractal = pattern == 135 ? 1 : (s.v[4] != 0.0f ? 1 : 0);
        float octaves = fractal != 0 ? s.v[1] : 1.0f;
        int last = pattern == 135 ? 6 : 7;
        float lower = pattern == 135 ? s.v[4] : s.v[5], upper = pattern == 135 ? s.v[5] : s.v[6];
        float v;
        if (s.v[last] != 0.0f) v = mp_worley(x, y, z, seed, 1.0f, 5);
        else if (pattern == 137) v = mp_clamp01(mp_worley_fractal(x, y, z, seed, 1.0f, 4, octaves, s.v[2], s.v[3]) * 2.0f);
        else if (pattern == 135) v = mp_step(mp_worley_fractal(x, y, z, seed, 1.0f, type, octaves, s.v[2], s.v[3]), 0.35f, 0.04f);
        else v = mp_clamp01(mp_worley_fractal(x, y, z, seed, 1.0f, type, octaves, s.v[2], s.v[3]));
        t = mp_range(v, lower, upper);
    } else if (pattern == 136) {
        // Concrete: seed, lower, upper. A cloudy grey with fine specks of grit.
        float cloud = mp_unit(mp_fbm(type, x * 2.0f, y * 2.0f, z * 2.0f, seed, 5.0f, 2.0f, 0.6f, 0));
        float grit = mp_step(mp_random((int)MP_FLOOR(x * 60.0f), (int)MP_FLOOR(y * 60.0f), (int)MP_FLOOR(z * 60.0f), seed + 3), 0.85f, 0.0f);
        t = mp_range(mp_clamp01(0.8f * cloud + 0.2f * grit), s.v[1], s.v[2]);
    } else if (pattern == 138) {
        // Dirt: seed, frequency 1, magnitude 1, frequency 2, magnitude 2, lower, upper. Grime at two sizes.
        float a = mp_fbm(type, x * s.v[1], y * s.v[1], z * s.v[1], seed, 4.0f, 2.0f, 0.5f, 0) * s.v[2];
        float b = mp_fbm(type, x * s.v[3], y * s.v[3], z * s.v[3], seed + 7, 3.0f, 2.0f, 0.5f, 0) * s.v[4];
        t = mp_range(mp_clamp01(0.5f + 0.5f * a + 0.25f * b), s.v[5], s.v[6]);
    } else if (pattern == 139) {
        // Disturbed: seed, distance, frequency, angle, lower, upper. Noise pushed aside and turned by noise.
        float push = mp_basis(type, x * s.v[2], y * s.v[2], z * s.v[2], seed + 5) * s.v[1] * 0.25f;
        float turn = s.v[3] * MP_PI * mp_basis(type, x, y, z, seed + 6);
        float rx = x * MP_COS(turn) - y * (MP_COS(turn - 0.5f * MP_PI));
        float ry = x * (MP_COS(turn - 0.5f * MP_PI)) + y * MP_COS(turn);
        t = mp_range(mp_unit(mp_basis(type, (rx + push) * s.v[2], (ry - push) * s.v[2], z * s.v[2], seed)), s.v[4], s.v[5]);
    } else if (pattern == 140 || pattern == 144) {
        // EasyWood: seed, octaves, magnitude, scale, detail, lower, upper. HardWood: seed, graininess, ringiness,
        // lower, upper. Rings round the locator's Y axis, wandering with noise.
        float ringScale = pattern == 140 ? MP_MAX(0.01f, s.v[3]) : MP_MAX(0.1f, s.v[2]);
        float wander = pattern == 140 ? s.v[2] * 0.3f * mp_fbm(type, x * s.v[4], y * s.v[4] * 0.2f, z * s.v[4], seed, s.v[1], 2.0f, 0.5f, 0)
                                      : 0.3f * mp_fbm(type, x, y * 0.2f, z, seed, 3.0f, 2.0f, 0.5f, 0);
        float r = MP_SQRT(x * x + z * z) * ringScale + wander;
        float grain = pattern == 144 ? s.v[1] * mp_basis(type, x * 40.0f, y * 2.0f, z * 40.0f, seed + 3) : 0.0f;
        float ring = pattern == 140 ? mp_fract(r) : 0.5f + 0.5f * MP_COS(2.0f * MP_PI * r);
        float lower = pattern == 140 ? s.v[5] : s.v[3], upper = pattern == 140 ? s.v[6] : s.v[4];
        t = mp_range(mp_clamp01(ring + grain), lower, upper);
    } else if (pattern == 141) {
        // Electric: seed, octaves, frequency, increment, width, lower, upper. Bright forks where the noise crosses 0.
        float n = mp_fbm(type, x, y, z, seed, MP_MAX(1.0f, s.v[1]), s.v[2], s.v[3], 0);
        t = mp_range(mp_clamp01(1.0f - MP_ABS(n) / MP_MAX(0.005f, s.v[4] * 0.25f)), s.v[5], s.v[6]);
    } else if (pattern == 142 || pattern == 143) {
        // Fire: seed, flame height, magnitude, fragmentation, detail, speed, time, lower, upper. FireWall: seed,
        // octaves, frequency, increment, height, lower, upper. Flames rising up the locator's Y axis, licked by
        // noise that moves up with time.
        float rise = pattern == 142 ? s.v[5] * s.v[6] : 0.0f;
        float flicker;
        float height;
        if (pattern == 142) {
            float f = MP_MAX(0.01f, s.v[3] * 0.1f);
            flicker = s.v[2] * mp_fbm(type, x * f, (y - rise) * f, z * f, seed, MP_MAX(1.0f, s.v[4] * 2.0f), 2.0f, 0.5f, 1);
            height = MP_MAX(0.01f, s.v[1]);
        } else {
            flicker = mp_fbm(type, x, y, z, seed, s.v[1], s.v[2], s.v[3], 1);
            height = MP_MAX(0.01f, s.v[4]);
        }
        float v = mp_clamp01(1.0f - y / height + flicker - 0.3f);
        float lower = pattern == 142 ? s.v[7] : s.v[5], upper = pattern == 142 ? s.v[8] : s.v[6];
        t = mp_range(v, lower, upper);
    } else if (pattern == 145 || pattern == 150) {
        // Membrane and Wires: seed, octaves, frequency, increment, jitter, grout, bevel, lower, upper. Cells parted
        // by grout and rounded by a bevel; Wires is the grout itself.
        float e = 0.5f * mp_worley_fractal(x, y, z, seed, s.v[4], 4, s.v[1], s.v[2], s.v[3]);
        float grout = MP_MAX(0.0f, s.v[5]) * 0.5f;
        float v = mp_smooth((e - grout) / MP_MAX(0.001f, s.v[6] * 0.5f));
        if (e < grout) v = 0.0f;
        if (pattern == 150) v = 1.0f - v;
        t = mp_range(v, s.v[7], s.v[8]);
    } else if (pattern == 146) {
        // Minky: seed, octaves, frequency, increment, power, lower, upper, regions. A soft pile, cell by cell.
        float f = s.v[7] != 0.0f ? mp_worley(x, y, z, seed, 1.0f, 5) : 1.0f - mp_clamp01(mp_worley_fractal(x, y, z, seed, 1.0f, type, s.v[1], s.v[2], s.v[3]));
        t = mp_range(MP_POW(mp_clamp01(f), MP_MAX(0.05f, s.v[4])), s.v[5], s.v[6]);
    } else if (pattern == 147) {
        // Scatter (type: Spheres 0, Cubes 1, Diamonds 2): seed, distribution, spread, scale, smooth, lower, upper.
        // A shape in some of the cells, as many as distribution says, moved about by spread.
        float cx = MP_FLOOR(x), cy = MP_FLOOR(y), cz = MP_FLOOR(z), best = 0.0f;
        float radius = 0.5f * MP_MAX(0.0f, s.v[3]);
        float soft = MP_MAX(0.001f, s.v[4] * radius);
        for (int k = -1; k <= 1; k++) {
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int a = (int)cx + i, b = (int)cy + j, c = (int)cz + k;
                    if (mp_random(a, b, c, seed + 7) > mp_clamp01(s.v[1])) continue;
                    float dx = MP_ABS(x - ((float)a + 0.5f + s.v[2] * (mp_random(a, b, c, seed) - 0.5f)));
                    float dy = MP_ABS(y - ((float)b + 0.5f + s.v[2] * (mp_random(a, b, c, seed + 1) - 0.5f)));
                    float dz = MP_ABS(z - ((float)c + 0.5f + s.v[2] * (mp_random(a, b, c, seed + 2) - 0.5f)));
                    float d = type == 1 ? MP_MAX(MP_MAX(dx, dy), dz) : (type == 2 ? (dx + dy + dz) * 0.7f : MP_SQRT(dx * dx + dy * dy + dz * dz));
                    best = MP_MAX(best, 1.0f - mp_smooth((d - radius + soft) / soft));
                }
            }
        }
        t = mp_range(best, s.v[5], s.v[6]);
    } else if (pattern == 148) {
        // SinBlob: seed, lower, upper. Waves bending through each other into soft blobs.
        float o = (float)(seed % 97) * 0.13f;
        float a = MP_COS(3.0f * x + o + 2.0f * MP_COS(2.0f * y - o));
        float b = MP_COS(3.0f * y - o + 2.0f * MP_COS(2.0f * z + o));
        float c = MP_COS(3.0f * z + o + 2.0f * MP_COS(2.0f * x));
        t = mp_range(mp_clamp01(0.5f + (a + b + c) / 6.0f), s.v[1], s.v[2]);
    } else if (pattern == 149) {
        // Veins: seed, octaves, veininess, lower, upper. Thin veins where the noise crosses its middle.
        float n = mp_fbm(type, x, y, z, seed, s.v[1], 2.0f, 0.5f, 0);
        t = mp_range(MP_POW(1.0f - mp_clamp01(MP_ABS(n) * 2.0f), MP_MAX(0.5f, s.v[2] * 8.0f)), s.v[3], s.v[4]);
    } else if (pattern == 151) {
        // WormVein: seed, lo, hi, lower, upper. Worm-like bands where the noise lies between lo and hi.
        float n = mp_unit(mp_fbm(type, x, y, z, seed, 4.0f, 2.0f, 0.5f, 0));
        float lo = MP_MIN(s.v[1], s.v[2]), hi = MP_MAX(s.v[1], s.v[2]);
        t = mp_range(mp_step(n, lo, 0.01f) * (1.0f - mp_step(n, hi, 0.01f)), s.v[3], s.v[4]);
    }
    return t;
}

// ---- Enhance: Modo's Tiles family, 160 on ----------------------------------------------------------------------------
// Tiles lie on the locator's XY plane, a tile a unit wide unless its settings say otherwise (the locator's size sets
// theirs). Each pattern's own settings come from s.v[0]; every one of them has the same edge noise in s.v[16] to
// s.v[20] (noise seed, octaves, magnitude, scale, detail), its Noise Type being type. 1 is a tile's face (the Fore
// colour), 0 the grout between (the Back colour).

MP_FUNC int mp_mod(int a, int n) { int m = a % n; return m < 0 ? m + n : m; }
MP_FUNC int mp_div(int a, int n) { return (a - mp_mod(a, n)) / n; }
MP_FUNC int mp_iabs(int a) { return a < 0 ? -a : a; }

// A tile's face from how far a point is inside its edge: 0 in the grout, rising over the bevel to 1, and darker at
// random, tile by tile, by as much as distrib.
MP_FUNC float mp_tile_face(float e, float grout, float bevel, float r, float distrib) {
    float g = MP_MAX(0.0f, grout) * 0.5f;
    if (e <= g) return 0.0f;
    float b = MP_MAX(0.0f, bevel) * 0.5f;
    float v = b > 1e-4f ? mp_smooth((e - g) / b) : 1.0f;
    return v * (1.0f - mp_clamp01(distrib) * r);
}

// How far a point at fx, fy of a w by h tile is inside its edge, its corners rounded by radius.
MP_FUNC float mp_round_edge(float fx, float fy, float w, float h, float radius) {
    float dx = MP_MIN(fx, w - fx);
    float dy = MP_MIN(fy, h - fy);
    float rr = MP_MAX(0.0f, MP_MIN(radius, 0.5f * MP_MIN(w, h)));
    if (rr > 0.0f && dx < rr && dy < rr) return rr - MP_SQRT((rr - dx) * (rr - dx) + (rr - dy) * (rr - dy));
    return MP_MIN(dx, dy);
}

// The edge noise: how far a point is pushed along one axis before the tiles are laid.
MP_HEAVY float mp_tile_warp(int type, float x, float y, float z, MP_Settings s, int axis) {
    if (s.v[18] == 0.0f) return 0.0f;
    float f = 1.0f / MP_MAX(0.01f, s.v[19]);
    return 0.1f * s.v[18] * mp_fbm(type, x * f, y * f, z * f, (int)s.v[16] + 17 * axis, MP_MAX(1.0f, s.v[17]), 2.0f,
                                   1.0f - 0.5f * mp_clamp01(s.v[20]), 0);
}

// A tile found for a point: how far inside its edge the point is, and two whole numbers naming the tile.
typedef struct { float e; int a; int b; } MP_Tile;

MP_FUNC MP_Tile mp_tile_of(float e, int a, int b) {
    MP_Tile tile;
    tile.e = e;
    tile.a = a;
    tile.b = b;
    return tile;
}

// Rows of w by h tiles, each row shifted by stagger of a tile from the one below.
MP_FUNC MP_Tile mp_rows(float x, float y, float w, float h, float stagger, float radius) {
    float bw = MP_MAX(0.01f, w);
    float bh = MP_MAX(0.01f, h);
    float row = MP_FLOOR(y / bh);
    float xs = x + stagger * bw * row;
    float col = MP_FLOOR(xs / bw);
    return mp_tile_of(mp_round_edge(xs - col * bw, y - row * bh, bw, bh, radius), (int)col, (int)row);
}

// Hexagons a unit across their flat sides, pointed at top and bottom: how far a point is inside its hexagon's edge,
// and where it is from the hexagon's middle.
typedef struct { float dx; float dy; int a; int b; } MP_Hex;
MP_FUNC MP_Hex mp_hex(float x, float y) {
    float h = 1.7320508f;
    float ia = MP_FLOOR(x + 0.5f);
    float ja = MP_FLOOR(y / h + 0.5f);
    float ax = x - ia;
    float ay = y - ja * h;
    float ib = MP_FLOOR(x);
    float jb = MP_FLOOR(y / h);
    float bx = x - (ib + 0.5f);
    float by = y - (jb + 0.5f) * h;
    MP_Hex hex;
    if (ax * ax + ay * ay <= bx * bx + by * by) { hex.dx = ax; hex.dy = ay; hex.a = 2 * (int)ia; hex.b = 2 * (int)ja; }
    else { hex.dx = bx; hex.dy = by; hex.a = 2 * (int)ib + 1; hex.b = 2 * (int)jb + 1; }
    return hex;
}
MP_FUNC float mp_hex_edge(float dx, float dy) {
    float m = MP_MAX(MP_ABS(dx), MP_MAX(MP_ABS(0.5f * dx + 0.8660254f * dy), MP_ABS(0.5f * dx - 0.8660254f * dy)));
    return 0.5f - m;
}

// Equilateral triangles a unit along each side, as many pointing up as down: how far a point is inside its
// triangle's edge, and the triangle (b odd for those pointing down).
MP_FUNC MP_Tile mp_triangle(float x, float y) {
    float u = x - y * 0.57735027f;
    float v = y * 1.1547005f;
    float cu = MP_FLOOR(u);
    float cv = MP_FLOOR(v);
    float fu = u - cu;
    float fv = v - cv;
    float rest = 1.0f - fu - fv;
    int down = rest < 0.0f ? 1 : 0;
    float e = MP_MIN(MP_MIN(fu, fv), MP_ABS(rest));
    if (down != 0) e = MP_MIN(MP_MIN(1.0f - fu, 1.0f - fv), -rest);
    return mp_tile_of(e * 0.8660254f, (int)cu, 2 * (int)cv + down);
}

// A run along a row: where a tile begins and how long it is, cut at whole numbers where chance says so (at least one
// unit, at most five).
typedef struct { float start; float length; } MP_Run;
MP_FUNC MP_Run mp_run(float x, int row, int seed, float chance) {
    int cell = (int)MP_FLOOR(x);
    int first = cell;
    int last = cell + 1;
    int open = 1;
    for (int k = 0; k < 4; k++) {
        if (open != 0 && mp_random(first, row, 3, seed) >= chance) first = first - 1;
        else open = 0;
    }
    open = 1;
    for (int k = 0; k < 4; k++) {
        if (open != 0 && mp_random(last, row, 3, seed) >= chance) last = last + 1;
        else open = 0;
    }
    MP_Run run;
    run.start = (float)first;
    run.length = (float)(last - first);
    return run;
}

// Whether thread A lies over thread B where column i of B crosses row j of A, for Basket's Pattern Type.
MP_FUNC int mp_weave_over(int kind, int i, int j, int seed) {
    if (kind == 1) return mp_mod(i + 2 * j, 5) != 0 ? 1 : 0;
    if (kind == 2) return mp_mod(mp_div(i, 2) + mp_div(j, 2), 2);
    if (kind == 3) return mp_mod(i + j, 4) < 2 ? 1 : 0;
    if (kind == 4) return (mp_mod(i, 4) == 0 && mp_mod(j, 4) == 0) ? 0 : 1;
    if (kind == 5) return mp_mod(mp_div(i, 3) + mp_div(j, 3) + i + j, 2);
    if (kind == 6) return mp_mod(i - j, 4) < 2 ? 1 : 0;
    if (kind == 7) return (mp_mod(i, 3) == 1 && mp_mod(j, 3) == 1) ? 0 : 1;
    if (kind == 8) return mp_mod(mp_div(i, 4) + mp_div(j, 2), 2);
    if (kind == 9) return mp_mod(i + 2 * mp_div(j, 2), 4) < 2 ? 1 : 0;
    if (kind == 10) return mp_mod(j, 2);
    if (kind == 11) return mp_mod(i + (mp_mod(mp_div(j, 4), 2) != 0 ? -j : j), 4) < 2 ? 1 : 0;
    if (kind == 12) return mp_mod(mp_div(i, 2) + j, 2);
    if (kind == 13) return mp_mod(mp_div(i, 4) + mp_div(j, 4), 2);
    if (kind == 14) return mp_mod(mp_div(i, 2) + mp_div(j, 4), 2);
    if (kind == 15 || kind == 25) return mp_mod(i, 2);
    if (kind == 16) return mp_mod(i + mp_iabs(mp_mod(j, 8) - 4), 4) < 2 ? 1 : 0;
    if (kind == 17) return mp_mod(j + mp_iabs(mp_mod(i, 8) - 4), 4) < 2 ? 1 : 0;
    if (kind == 18) return mp_mod(i + mp_iabs(mp_mod(j, 12) - 6), 6) < 3 ? 1 : 0;
    if (kind == 19) return mp_mod(mp_div(i, 2) + mp_iabs(mp_mod(j, 8) - 4), 2);
    if (kind == 20) return mp_mod(i + 2 * j, 3) == 0 ? 0 : 1;
    if (kind == 21) return mp_mod(2 * i + j, 3) != 0 ? 0 : 1;
    if (kind == 22) {
        int ci = mp_mod(i, 16) - 8;
        int cj = mp_mod(j, 16) - 8;
        return mp_mod((int)MP_SQRT((float)(ci * ci + cj * cj)), 2);
    }
    if (kind == 23) return (mp_mod(i, 6) == 3 || mp_mod(j, 6) == 3) ? 1 - mp_mod(i + j, 2) : mp_mod(i + j, 2);
    if (kind == 24) return mp_random(i, j, 9, seed) < 0.5f ? 1 : 0;
    return mp_mod(i + j, 2);
}

// Tiler's layouts (its Tile Type), in units of the smallest tile.
MP_HEAVY MP_Tile mp_tiler(int kind, float x, float y, float radius, int seed) {
    int a = (int)MP_FLOOR(x);
    int b = (int)MP_FLOOR(y);
    float fx = x - (float)a;
    float fy = y - (float)b;
    if (kind == 0 || kind == 14) return mp_rows(x, y, 2.0f, 1.0f, 0.5f, radius);
    if (kind == 15) return mp_rows(x, y, 2.0f, 1.0f, 1.0f / 3.0f, radius);
    if (kind == 16) return mp_rows(x, y, 2.0f, 1.0f, 0.25f, radius);
    if (kind == 17) return mp_mod(b, 2) == 0 ? mp_rows(x, y, 2.0f, 1.0f, 0.0f, radius) : mp_rows(x + 0.5f, y, 1.0f, 1.0f, 0.0f, radius);
    if (kind == 2) return mp_rows(x, y, 1.0f, 1.0f, 0.0f, radius);
    if (kind == 20) return mp_rows(x, y, 2.0f, 1.0f, 0.0f, radius);
    if (kind == 21) return mp_rows(y, x, 2.0f, 1.0f, 0.0f, radius);
    if (kind == 1 || kind == 24 || kind == 25) {
        // Basket weave: two by two blocks, their tiles lying one way and then the other (BasketV and BasketH in
        // threes, all one way).
        int planks = kind == 1 ? 2 : 3;
        float size = (float)planks;
        int ba = (int)MP_FLOOR(x / size);
        int bb = (int)MP_FLOOR(y / size);
        float u = x - (float)ba * size;
        float v = y - (float)bb * size;
        int across = kind == 24 ? 0 : (kind == 25 ? 1 : mp_mod(ba + bb, 2));
        if (across != 0) {
            int k = (int)MP_FLOOR(v);
            return mp_tile_of(mp_round_edge(u, v - (float)k, size, 1.0f, radius), ba * 8 + k, bb);
        }
        int k = (int)MP_FLOOR(u);
        return mp_tile_of(mp_round_edge(u - (float)k, v, 1.0f, size, radius), ba * 8 + k, bb + 4096);
    }
    if (kind == 3) {
        // Half weave: a long tile over two squares, block by block.
        int ba = mp_div(a, 2);
        float u = x - (float)ba * 2.0f;
        if (mp_mod(ba + b, 2) == 0) return mp_tile_of(mp_round_edge(u, fy, 2.0f, 1.0f, radius), ba, b);
        return mp_tile_of(mp_round_edge(fx, fy, 1.0f, 1.0f, radius), a, b + 4096);
    }
    if (kind == 4 || kind == 22 || kind == 23) {
        // Corridors: rows of squares between rows of long tiles (II: squares two wide; III: three rows of each).
        int group = kind == 23 ? 3 : 1;
        int band = mp_mod(mp_div(b, group), 2);
        if (band == 0) {
            float w = kind == 22 ? 2.0f : 1.0f;
            return mp_rows(x, y, w, 1.0f, 0.0f, radius);
        }
        return mp_rows(x, y, 3.0f, 1.0f, 0.5f, radius);
    }
    if (kind == 5) {
        // Morse code: rows of tiles of one to five units, cut at random.
        MP_Run run = mp_run(x, b, seed, 0.45f);
        return mp_tile_of(mp_round_edge(x - run.start, fy, run.length, 1.0f, radius), (int)run.start, b);
    }
    if (kind == 6 || kind == 7) {
        // Sliding: columns (rows) of tiles two long, each column slid along at random.
        float px = kind == 6 ? x : y;
        float py = kind == 6 ? y : x;
        int column = (int)MP_FLOOR(px);
        float slide = py + 2.0f * mp_random(column, 0, 5, seed);
        int k = (int)MP_FLOOR(slide / 2.0f);
        return mp_tile_of(mp_round_edge(px - (float)column, slide - (float)k * 2.0f, 1.0f, 2.0f, radius), column, k);
    }
    if (kind == 8) {
        // Herringbone: tiles two by one, lying and standing by turns along the diagonal.
        int sum = mp_mod(a + b, 4);
        if (sum == 0) return mp_tile_of(mp_round_edge(fx, fy, 2.0f, 1.0f, radius), a, b);
        if (sum == 1) return mp_tile_of(mp_round_edge(fx + 1.0f, fy, 2.0f, 1.0f, radius), a - 1, b);
        if (sum == 2) return mp_tile_of(mp_round_edge(fx, fy, 1.0f, 2.0f, radius), a, b + 4096);
        return mp_tile_of(mp_round_edge(fx, fy + 1.0f, 1.0f, 2.0f, radius), a, b - 1 + 4096);
    }
    if (kind == 9) {
        // Windmill: a square with four tiles two by one turning round it, in blocks of three.
        int ba = mp_div(a, 3);
        int bb = mp_div(b, 3);
        float u = x - (float)ba * 3.0f;
        float v = y - (float)bb * 3.0f;
        int id = ba * 8;
        if (u >= 1.0f && u < 2.0f && v >= 1.0f && v < 2.0f) return mp_tile_of(mp_round_edge(u - 1.0f, v - 1.0f, 1.0f, 1.0f, radius), id, bb);
        if (v < 1.0f && u < 2.0f) return mp_tile_of(mp_round_edge(u, v, 2.0f, 1.0f, radius), id + 1, bb);
        if (u >= 2.0f && v < 2.0f) return mp_tile_of(mp_round_edge(u - 2.0f, v, 1.0f, 2.0f, radius), id + 2, bb);
        if (v >= 2.0f && u >= 1.0f) return mp_tile_of(mp_round_edge(u - 1.0f, v - 2.0f, 2.0f, 1.0f, radius), id + 3, bb);
        return mp_tile_of(mp_round_edge(u, v - 1.0f, 1.0f, 2.0f, radius), id + 4, bb);
    }
    if (kind == 10) {
        // Trellis: squares with slim tiles along two sides and a small square at their corner.
        float size = 1.5f;
        int ba = (int)MP_FLOOR(x / size);
        int bb = (int)MP_FLOOR(y / size);
        float u = x - (float)ba * size;
        float v = y - (float)bb * size;
        if (u < 1.0f && v < 1.0f) return mp_tile_of(mp_round_edge(u, v, 1.0f, 1.0f, radius), ba * 4, bb);
        if (v < 1.0f) return mp_tile_of(mp_round_edge(u - 1.0f, v, 0.5f, 1.0f, radius), ba * 4 + 1, bb);
        if (u < 1.0f) return mp_tile_of(mp_round_edge(u, v - 1.0f, 1.0f, 0.5f, radius), ba * 4 + 2, bb);
        return mp_tile_of(mp_round_edge(u - 1.0f, v - 1.0f, 0.5f, 0.5f, radius), ba * 4 + 3, bb);
    }
    if (kind == 11 || kind == 18) {
        // Vectored: two by two blocks of two long tiles, lying one way or the other at random. Splits: the same,
        // cut in two or left whole at random.
        int ba = mp_div(a, 2);
        int bb = mp_div(b, 2);
        float u = x - (float)ba * 2.0f;
        float v = y - (float)bb * 2.0f;
        float r = mp_random(ba, bb, 11, seed);
        if (kind == 18 && r < 0.33f) return mp_tile_of(mp_round_edge(u, v, 2.0f, 2.0f, radius), ba * 4, bb);
        if (r < 0.66f) {
            int k = (int)MP_FLOOR(v);
            return mp_tile_of(mp_round_edge(u, v - (float)k, 2.0f, 1.0f, radius), ba * 4 + 1 + k, bb);
        }
        int k = (int)MP_FLOOR(u);
        return mp_tile_of(mp_round_edge(u - (float)k, v, 1.0f, 2.0f, radius), ba * 4 + 3 + k, bb + 4096);
    }
    if (kind == 12) {
        // Hopscotch: squares two wide and one wide, the small ones in the large ones' corners.
        float la = (2.0f * x + y) / 5.0f;
        float lb = (2.0f * y - x) / 5.0f;
        int ka = (int)MP_FLOOR(la);
        int kb = (int)MP_FLOOR(lb);
        MP_Tile found = mp_tile_of(-1.0f, ka, kb);
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int ca = ka + i;
                int cb = kb + j;
                float ox = 2.0f * (float)ca - (float)cb;
                float oy = (float)ca + 2.0f * (float)cb;
                float u = x - ox;
                float v = y - oy;
                if (u >= 0.0f && u < 2.0f && v >= 0.0f && v < 2.0f) found = mp_tile_of(mp_round_edge(u, v, 2.0f, 2.0f, radius), ca, cb);
                if (u >= 1.0f && u < 2.0f && v >= 2.0f && v < 3.0f) found = mp_tile_of(mp_round_edge(u - 1.0f, v - 2.0f, 1.0f, 1.0f, radius), ca, cb + 4096);
            }
        }
        return found;
    }
    if (kind == 13) {
        // Cobbled: stones of uneven sizes.
        float e = 0.5f * mp_worley(x, y, 0.5f, seed, 0.6f, 4);
        float r = mp_worley(x, y, 0.5f, seed, 0.6f, 5);
        return mp_tile_of(e, (int)(r * 65536.0f), 0);
    }
    if (kind == 19) {
        // Half basket: two long tiles lying beside one standing, in blocks three wide.
        int ba = mp_div(a, 3);
        int bb = mp_div(b, 2);
        float u = x - (float)ba * 3.0f;
        float v = y - (float)bb * 2.0f;
        if (u < 2.0f) {
            int k = (int)MP_FLOOR(v);
            return mp_tile_of(mp_round_edge(u, v - (float)k, 2.0f, 1.0f, radius), ba * 4 + k, bb);
        }
        return mp_tile_of(mp_round_edge(u - 2.0f, v, 1.0f, 2.0f, radius), ba * 4 + 3, bb);
    }
    return mp_rows(x, y, 1.0f, 1.0f, 0.0f, radius);
}

// Enhance: Modo's Tiles patterns, 160 to 175, and 176 on below.
// (Two functions, 160 to 175 and 176 on: ISPC's parser runs out of room in one chain of them all.)
MP_HEAVY float mp_tiles_first(int pattern, float x, float y, int seed, MP_Settings s) {
    float t = 0.0f;
    if (pattern == 160) {
        // Basket: thread A width, thread B width, thread, weave, thread 2, weave 2, thread A frequency and
        // amplitude, thread B frequency and amplitude, threads A, threads B, pattern type. Cloth: threads A along X,
        // B along Y, over and under as the pattern type says; the second colours on every other thread.
        float na = MP_MAX(1.0f, s.v[10]);
        float nb = MP_MAX(1.0f, s.v[11]);
        float ya = y * na + s.v[7] * MP_COS(2.0f * MP_PI * x * s.v[6]);
        float xb = x * nb + s.v[9] * MP_COS(2.0f * MP_PI * y * s.v[8]);
        int j = (int)MP_FLOOR(ya);
        int i = (int)MP_FLOOR(xb);
        float fa = ya - (float)j - 0.5f;
        float fb = xb - (float)i - 0.5f;
        float wa = MP_MAX(0.01f, mp_clamp01(s.v[0]) * 0.5f);
        float wb = MP_MAX(0.01f, mp_clamp01(s.v[1]) * 0.5f);
        float shadeA = MP_SQRT(mp_clamp01(1.0f - (fa / wa) * (fa / wa)));
        float shadeB = MP_SQRT(mp_clamp01(1.0f - (fb / wb) * (fb / wb)));
        int inA = MP_ABS(fa) < wa ? 1 : 0;
        int inB = MP_ABS(fb) < wb ? 1 : 0;
        float colourA = mp_mod(j, 2) != 0 ? s.v[4] : s.v[2];
        float colourB = mp_mod(i, 2) != 0 ? s.v[5] : s.v[3];
        int over = mp_weave_over((int)s.v[12], i, j, seed);
        if (inA != 0 && (over != 0 || inB == 0)) t = colourA * shadeA;
        else if (inB != 0) t = colourB * shadeB;
    } else if (pattern == 161 || pattern == 162 || pattern == 180) {
        // BathTile: seed, stagger, distribution, grout, bevel. Bricks: the same, then height and width.
        // RoundedTile: the same, then round. Rows of tiles, each row shifted along by stagger.
        float w = pattern == 162 ? s.v[6] : 1.0f;
        float h = pattern == 162 ? s.v[5] : 1.0f;
        float radius = pattern == 180 ? mp_clamp01(s.v[5]) * 0.5f * MP_MIN(w, h) : 0.0f;
        MP_Tile tile = mp_rows(x, y, w, h, s.v[1], radius);
        t = mp_tile_face(tile.e, s.v[3], s.v[4], mp_random(tile.a, tile.b, 0, seed), s.v[2]);
    } else if (pattern == 163) {
        // Checks: seed, stagger, distribution, softness. A checkerboard, rows shifted by stagger, softened at the
        // edges of its squares.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, s.v[1], 0.0f);
        float v = mp_mod(tile.a + tile.b, 2) != 0 ? 1.0f - mp_clamp01(s.v[2]) * mp_random(tile.a, tile.b, 0, seed) : 0.0f;
        float blend = s.v[3] > 1e-4f ? mp_smooth(tile.e / (s.v[3] * 0.5f)) : 1.0f;
        t = 0.5f + (v - 0.5f) * blend;
    } else if (pattern == 164) {
        // Cornerless: seed, distribution, grout, bevel. Squares with their corners cut away.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        float dx = MP_MIN(x - (float)tile.a, 1.0f - (x - (float)tile.a));
        float dy = MP_MIN(y - (float)tile.b, 1.0f - (y - (float)tile.b));
        float e = MP_MIN(tile.e, (dx + dy) * 0.70710678f - 0.15f);
        t = mp_tile_face(e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 165) {
        // Cubes: seed. Hexagons drawn as cubes seen from a corner, a lit top and two shaded sides; with a seed, each
        // turned at random.
        MP_Hex hex = mp_hex(x, y);
        int face = 2;
        if (0.8660254f * hex.dy - 0.5f * hex.dx >= 0.0f && 0.5f * hex.dx + 0.8660254f * hex.dy >= 0.0f) face = 0;
        else if (-0.8660254f * hex.dy - 0.5f * hex.dx >= 0.0f && hex.dx <= 0.0f) face = 1;
        if (seed != 0) face = mp_mod(face + (int)(mp_hash(hex.a, hex.b, 0, seed) % 3u), 3);
        t = face == 0 ? 1.0f : (face == 1 ? 0.6f : 0.3f);
    } else if (pattern == 166) {
        // DashLine: seed, pattern, width, smooth. Rows of dashed lines; the pattern sets the dashes' length, each
        // row starting at random.
        float row = MP_FLOOR(y);
        float across = MP_ABS(y - row - 0.5f);
        float hw = MP_MAX(0.005f, mp_clamp01(s.v[2]) * 0.5f);
        float soft = MP_MAX(0.0f, s.v[3]) * hw;
        float line = 1.0f - mp_step(across, hw, soft);
        float dash = 0.4f + 0.1f * (float)mp_mod((int)s.v[1], 5);
        float along = mp_fract(x + mp_random((int)row, 0, 2, seed));
        float on = 1.0f - mp_step(along, dash, soft);
        t = line * on;
    } else if (pattern == 167) {
        // DiamondDeck: seed, distribution, grout, bevel. Tread plate: long diamonds lying one way and the other
        // by turns.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        float dx = x - (float)tile.a - 0.5f;
        float dy = y - (float)tile.b - 0.5f;
        float flip = mp_mod(tile.a + tile.b, 2) != 0 ? -1.0f : 1.0f;
        float u = (dx + flip * dy) * 0.70710678f;
        float v = (dx - flip * dy) * 0.70710678f;
        float e = 0.12f * (1.0f - (MP_ABS(u) / 0.45f + MP_ABS(v) / 0.12f));
        t = mp_tile_face(e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 168) {
        // FishScales: seed, distribution, bevel. Rows of round scales, each row overlapping the one above.
        float row = MP_FLOOR(y / 0.5f);
        int found = 0;
        float e = 0.0f;
        int fa = 0;
        int fb = 0;
        for (int k = 1; k >= -1; k--) {
            int r = (int)row + k;
            float shift = 0.5f * (float)mp_mod(r, 2);
            float col = MP_FLOOR(x - shift + 0.5f);
            float cx = col + shift;
            float cy = (float)r * 0.5f;
            float d = MP_SQRT((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (found == 0 && d < 0.5f && y <= cy + 0.0001f) { found = 1; e = 0.5f - d; fa = (int)col; fb = r; }
        }
        if (found != 0) t = mp_tile_face(e, 0.02f, s.v[2], mp_random(fa, fb, 0, seed), s.v[1]);
    } else if (pattern == 169) {
        // HexTile: seed, distribution, grout, bevel.
        MP_Hex hex = mp_hex(x, y);
        t = mp_tile_face(mp_hex_edge(hex.dx, hex.dy), s.v[2], s.v[3], mp_random(hex.a, hex.b, 0, seed), s.v[1]);
    } else if (pattern == 170 || pattern == 171) {
        // Lattice1: thickness. Lattice2: seed. Bars crossing over and under (Lattice2's on the diagonal, each bar
        // its own shade).
        float u = x;
        float v = y;
        if (pattern == 171) { u = (x + y) * 0.70710678f; v = (y - x) * 0.70710678f; }
        float hw = pattern == 170 ? MP_MAX(0.01f, mp_clamp01(s.v[0]) * 0.5f) : 0.15f;
        int i = (int)MP_FLOOR(u + 0.5f);
        int j = (int)MP_FLOOR(v + 0.5f);
        float du = MP_ABS(u - (float)i);
        float dv = MP_ABS(v - (float)j);
        float shadeU = MP_SQRT(mp_clamp01(1.0f - (du / hw) * (du / hw)));
        float shadeV = MP_SQRT(mp_clamp01(1.0f - (dv / hw) * (dv / hw)));
        float barU = pattern == 171 ? 0.6f + 0.4f * mp_random(i, 0, 1, seed) : 1.0f;
        float barV = pattern == 171 ? 0.6f + 0.4f * mp_random(j, 0, 2, seed) : 1.0f;
        int over = mp_mod(i + j, 2);
        if (du < hw && (over != 0 || dv >= hw)) t = barU * shadeU;
        else if (dv < hw) t = barV * shadeV;
    } else if (pattern == 172) {
        // Lattice3: bars along the edges of hexagons.
        MP_Hex hex = mp_hex(x, y);
        t = 1.0f - mp_smooth(mp_hex_edge(hex.dx, hex.dy) / 0.12f);
    } else if (pattern == 173) {
        // Mosaic: seed, jitter, distribution, grout, bevel, X shift, X scale, Y shift, Y scale. Uneven pieces.
        float mx = (x + s.v[5]) * (s.v[6] != 0.0f ? s.v[6] : 1.0f);
        float my = (y + s.v[7]) * (s.v[8] != 0.0f ? s.v[8] : 1.0f);
        float e = 0.5f * mp_worley(mx, my, 0.5f, seed, s.v[1], 4);
        float r = mp_worley(mx, my, 0.5f, seed, s.v[1], 5);
        t = mp_tile_face(e, s.v[3], s.v[4], r, s.v[2]);
    } else if (pattern == 174) {
        // OctTile: seed, stagger, distribution, grout, bevel. Octagons with small squares between their corners.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, s.v[1], 0.0f);
        float xs = x + s.v[1] * MP_FLOOR(y);
        float dx = MP_ABS(xs - (float)tile.a - 0.5f);
        float dy = MP_ABS(y - (float)tile.b - 0.5f);
        float diagonal = (dx + dy) * 0.70710678f;
        float e = 0.5f - MP_MAX(MP_MAX(dx, dy), diagonal * 1.0823922f);
        int corner = e < 0.0f ? 1 : 0;
        if (corner != 0) e = MP_MIN(-e, MP_MIN(0.5f - dx, 0.5f - dy));
        t = mp_tile_face(e, s.v[3], s.v[4], mp_random(tile.a, tile.b, corner, seed), s.v[2]);
    } else if (pattern == 175) {
        // Parquet: seed, distribution, grout, bevel, blocks. Squares of parallel strips, each square lying across
        // its neighbours.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        float u = x - (float)tile.a;
        float v = y - (float)tile.b;
        if (mp_mod(tile.a + tile.b, 2) != 0) { float w = u; u = v; v = w; }
        float n = MP_MAX(1.0f, MP_FLOOR(s.v[4]));
        float k = MP_FLOOR(v * n);
        float e = MP_MIN(MP_MIN(u, 1.0f - u), MP_MIN(v - k / n, (k + 1.0f) / n - v));
        t = mp_tile_face(e, s.v[2], s.v[3], mp_random(tile.a, tile.b, (int)k, seed), s.v[1]);
    }
    return t;
}

MP_HEAVY float mp_tiles_second(int pattern, float x, float y, int seed, MP_Settings s) {
    float t = 0.0f;
    if (pattern == 176) {
        // Paving: seed, jitter, distribution, grout, bevel, paving shape, paving style. Squares, hexagons,
        // octagons, triangles or wedges laid straight; or crazy paving's uneven stones, ornate octagons, or uneven
        // rows.
        int shape = (int)s.v[5];
        int style = (int)s.v[6];
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        if (style == 1) {
            tile = mp_tile_of(0.5f * mp_worley(x, y, 0.5f, seed, MP_MAX(0.3f, s.v[1]), 4), 0, 0);
            tile.a = (int)(mp_worley(x, y, 0.5f, seed, MP_MAX(0.3f, s.v[1]), 5) * 65536.0f);
        } else if (style == 2 || shape == 2) {
            float dx = MP_ABS(x - (float)tile.a - 0.5f);
            float dy = MP_ABS(y - (float)tile.b - 0.5f);
            float e = 0.5f - MP_MAX(MP_MAX(dx, dy), (dx + dy) * 0.70710678f * 1.0823922f);
            if (e < 0.0f) e = MP_MIN(-e, MP_MIN(0.5f - dx, 0.5f - dy));
            tile.e = e;
        } else if (style == 3) {
            int row = (int)MP_FLOOR(y);
            tile = mp_rows(x + mp_random(row, 0, 4, seed) * 3.0f, y, 1.0f + mp_clamp01(s.v[1]) * mp_random(row, 1, 4, seed), 1.0f, 0.0f, 0.0f);
        } else if (shape == 1) {
            MP_Hex hex = mp_hex(x, y);
            tile = mp_tile_of(mp_hex_edge(hex.dx, hex.dy), hex.a, hex.b);
        } else if (shape == 3) {
            tile = mp_triangle(x, y);
        } else if (shape == 4) {
            float u = x - (float)tile.a;
            float v = y - (float)tile.b;
            float diagonal = (v - u) * 0.70710678f;
            tile.e = MP_MIN(tile.e, MP_ABS(diagonal));
            tile.b = 2 * tile.b + (diagonal > 0.0f ? 1 : 0);
        } else if (s.v[1] != 0.0f) {
            int row = (int)MP_FLOOR(y);
            tile = mp_rows(x + s.v[1] * mp_random(row, 0, 6, seed), y, 1.0f, 1.0f, 0.0f, 0.0f);
        }
        t = mp_tile_face(tile.e, s.v[3], s.v[4], mp_random(tile.a, tile.b, 0, seed), s.v[2]);
    } else if (pattern == 177) {
        // Plaid: bands of several widths across both ways, laid over each other.
        float bx = 0.5f * mp_step(mp_fract(x), 0.6f, 0.02f) + 0.3f * mp_step(mp_fract(x * 3.0f), 0.5f, 0.05f) + 0.2f * mp_step(mp_fract(x * 7.0f), 0.8f, 0.05f);
        float by = 0.5f * mp_step(mp_fract(y), 0.6f, 0.02f) + 0.3f * mp_step(mp_fract(y * 3.0f), 0.5f, 0.05f) + 0.2f * mp_step(mp_fract(y * 7.0f), 0.8f, 0.05f);
        t = mp_clamp01(0.5f * (bx + by));
    } else if (pattern == 178) {
        // Planks: seed, distribution, grout, bevel, length, stagger. Boards of the length along X, each row moved
        // along at random by as much as stagger.
        float length = MP_MAX(1.0f, s.v[4]);
        int row = (int)MP_FLOOR(y);
        MP_Tile tile = mp_rows(x + s.v[5] * length * mp_random(row, 0, 7, seed), y, length, 1.0f, 0.0f, 0.0f);
        t = mp_tile_face(tile.e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 179) {
        // Ribs: seed, distribution, grout, bevel, ribs, layout. Tiles with ribs across them: all one way, along X
        // (Positive), along Y (Negative), or both by turns.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        float u = x - (float)tile.a;
        float v = y - (float)tile.b;
        int layout = (int)s.v[5];
        int along = layout == 2 ? 1 : (layout == 3 ? mp_mod(tile.a + tile.b, 2) : 0);
        float across = along != 0 ? u : v;
        float rib = 0.5f + 0.5f * MP_COS(2.0f * MP_PI * across * MP_MAX(1.0f, MP_FLOOR(s.v[4])));
        t = mp_tile_face(tile.e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]) * (0.6f + 0.4f * rib);
    } else if (pattern == 181) {
        // Shingles: seed, distribution, bevel. Rows of shingles, shifted by hw, round at the foot and lying in
        // the shadow of the row above.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.5f, 0.0f);
        float u = x + 0.5f * MP_FLOOR(y) - (float)tile.a;
        float v = y - (float)tile.b;
        float foot = 0.5f - MP_SQRT((u - 0.5f) * (u - 0.5f) + MP_MAX(0.0f, 0.5f - v) * MP_MAX(0.0f, 0.5f - v));
        float e = MP_MIN(MP_MIN(u, 1.0f - u), foot);
        t = mp_tile_face(e, 0.02f, s.v[2], mp_random(tile.a, tile.b, 0, seed), s.v[1]) * (0.5f + 0.5f * v);
    } else if (pattern == 182) {
        // Spots: seed, stagger, distribution, bevel, radius. A round spot in each square.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, s.v[1], 0.0f);
        float xs = x + s.v[1] * MP_FLOOR(y);
        float dx = xs - (float)tile.a - 0.5f;
        float dy = y - (float)tile.b - 0.5f;
        t = mp_tile_face(MP_MAX(0.0f, s.v[4]) - MP_SQRT(dx * dx + dy * dy), 0.0f, s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[2]);
    } else if (pattern == 183) {
        // Stamped: seed, stagger, distribution, grout, bevel, stamps, stamp type. Tiles pressed with a few marks:
        // straight bars, curves or capsules.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, s.v[1], 0.0f);
        float u = x + s.v[1] * MP_FLOOR(y) - (float)tile.a;
        float v = y - (float)tile.b;
        float n = MP_MAX(1.0f, MP_FLOOR(s.v[5]));
        int kind = (int)s.v[6];
        float across = kind == 1 ? MP_SQRT(u * u + v * v) * 1.2f : v;
        float k = MP_FLOOR(across * n);
        float mark = 1.0f - mp_step(MP_ABS(across * n - k - 0.5f), 0.18f, 0.05f);
        if (kind == 2) mark = mark * (1.0f - mp_step(MP_ABS(u - 0.5f), 0.3f, 0.05f));
        else mark = mark * (1.0f - mp_step(MP_ABS(u - 0.5f), 0.4f, 0.02f));
        t = mp_tile_face(tile.e, s.v[3], s.v[4], mp_random(tile.a, tile.b, 0, seed), s.v[2]) * (1.0f - 0.5f * mark);
    } else if (pattern == 184) {
        // Tacos: seed, distribution, grout, bevel, layout. Half-round (or hw-square) tiles, all one way or
        // turned by turns.
        MP_Tile tile = mp_rows(x, y, 1.0f, 1.0f, 0.0f, 0.0f);
        float u = x - (float)tile.a;
        float v = y - (float)tile.b;
        int layout = (int)s.v[4];
        if ((layout == 1 || layout == 3) && mp_mod(tile.a + tile.b, 2) != 0) v = 1.0f - v;
        float e;
        if (layout >= 2) e = MP_MIN(0.5f - MP_SQRT((u - 0.5f) * (u - 0.5f) + v * v), v);
        else e = MP_MIN(MP_MIN(u, 1.0f - u), MP_MIN(v, 0.5f - v));
        t = mp_tile_face(e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 185) {
        // TarTan: width, height, thick, thread. Bands across and along, the width and height apart and thick
        // wide, with a fine twill of threads over them.
        float w = MP_MAX(0.01f, s.v[0]);
        float h = MP_MAX(0.01f, s.v[1]);
        float hw = mp_clamp01(s.v[2]) * 0.5f;
        float bx = 1.0f - mp_step(MP_ABS(mp_fract(x / w) - 0.5f), hw, 0.01f);
        float by = 1.0f - mp_step(MP_ABS(mp_fract(y / h) - 0.5f), hw, 0.01f);
        float twill = mp_mod((int)MP_FLOOR(x * 40.0f) + (int)MP_FLOOR(y * 40.0f), 4) < 2 ? 1.0f : 0.0f;
        t = mp_clamp01(0.45f * (bx + by) + 0.1f + mp_clamp01(s.v[3]) * 0.1f * (twill - 0.5f));
    } else if (pattern == 186) {
        // Tiler: seed, distribution, grout, bevel, round, tile type. Tiles laid in one of 26 layouts.
        MP_Tile tile = mp_tiler((int)s.v[5], x, y, mp_clamp01(s.v[4]) * 0.5f, seed);
        t = mp_tile_face(tile.e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 187) {
        // TriChecks: seed, distribution, softness. A checkerboard of triangles.
        MP_Tile tile = mp_triangle(x, y);
        float v = mp_mod(tile.b, 2) != 0 ? 1.0f - mp_clamp01(s.v[1]) * mp_random(tile.a, tile.b, 0, seed) : 0.0f;
        float blend = s.v[2] > 1e-4f ? mp_smooth(tile.e / (s.v[2] * 0.5f)) : 1.0f;
        t = 0.5f + (v - 0.5f) * blend;
    } else if (pattern == 188) {
        // TriHexes: hexagons meeting at their corners, triangles between them.
        MP_Hex hex = mp_hex(x, y);
        float m = MP_MAX(MP_ABS(hex.dy), MP_MAX(MP_ABS(0.8660254f * hex.dx + 0.5f * hex.dy), MP_ABS(0.8660254f * hex.dx - 0.5f * hex.dy)));
        float e = 0.4330127f - m;
        t = e > 0.0f ? 1.0f - 0.3f * (1.0f - mp_smooth(e / 0.03f)) : 0.4f;
    } else if (pattern == 189) {
        // TriTile: seed, distribution, grout, bevel. Triangular tiles.
        MP_Tile tile = mp_triangle(x, y);
        t = mp_tile_face(tile.e, s.v[2], s.v[3], mp_random(tile.a, tile.b, 0, seed), s.v[1]);
    } else if (pattern == 190) {
        // Wall: seed, stagger, distribution, grout, bevel, round, levels. Courses of stones of uneven lengths, the
        // courses shifted by stagger; levels lets stones fall short of their course's height.
        int row = (int)MP_FLOOR(y);
        float xs = x + s.v[1] * (float)row;
        MP_Run run = mp_run(xs * 0.75f, row, seed, 0.5f);
        float u = xs * 0.75f - run.start;
        float v = y - (float)row;
        float drop = mp_clamp01(s.v[6] * 0.1f) * mp_random((int)run.start, row, 8, seed);
        float h = 1.0f - drop;
        float radius = mp_clamp01(s.v[5]) * 0.5f * MP_MIN(run.length, h);
        float e = v < h ? mp_round_edge(u, v, run.length, h, radius) : -1.0f;
        t = mp_tile_face(e, s.v[3], s.v[4], mp_random((int)run.start, row, 0, seed), s.v[2]);
    }
    return t;
}

// Enhance: Modo's Tiles patterns, 160 on, in the order of their names; the settings each reads are beside it
// (procedurals.ENHANCE).
MP_HEAVY float mp_enhance_tiles(int pattern, int type, float px, float py, float pz, MP_Settings s) {
    float x = px + mp_tile_warp(type, px, py, pz, s, 0);
    float y = py + mp_tile_warp(type, px, py, pz, s, 1);
    if (pattern < 176) return mp_tiles_first(pattern, x, y, (int)s.v[0], s);
    return mp_tiles_second(pattern, x, y, (int)s.v[0], s);
}

// ---- Enhance: Modo's Geometric, Waveforms, Skins, Space, Panels, Water and Display families, 200 on ------------------
// procedurals.ENHANCE lays out their settings alike: a pattern's own from s.v[0], its Lower and Upper in s.v[14] and
// s.v[15], and where it has one, its noise (seed, octaves, magnitude, scale, detail) in s.v[16] to s.v[20]. Waveforms
// have their amplitude, frequency, phase and shift in s.v[10] to s.v[13]. Flat ones lie on the locator's XY plane;
// Water on its XZ plane, as water lies.

// Smoothed by as much as amount.
MP_FUNC float mp_soften(float t, float amount) { return mp_lerp(t, mp_smooth(t), mp_clamp01(amount)); }
// In as many even steps as levels (none under two).
MP_FUNC float mp_levels(float t, float levels) {
    float n = MP_FLOOR(levels);
    if (n < 2.0f) return t;
    return mp_clamp01(MP_FLOOR(mp_clamp01(t) * n * 0.9999f) / (n - 1.0f));
}
// Nothing below clip.
MP_FUNC float mp_clip(float t, float clip) { return t < clip ? 0.0f : t; }

// Enhance: Modo's Geometric patterns, 200 on: a shape in each unit cell of the locator's XY plane (Cubic in each
// cube), turned a quarter at random cell by cell where the region seed is set.
MP_HEAVY float mp_enhance_geometric(int pattern, int type, float px, float py, float pz, MP_Settings s) {
    float x = px + mp_tile_warp(type, px, py, pz, s, 0);
    float y = py + mp_tile_warp(type, px, py, pz, s, 1);
    float z = pz + mp_tile_warp(type, px, py, pz, s, 2);
    float cx = MP_FLOOR(x);
    float cy = MP_FLOOR(y);
    float cz = MP_FLOOR(z);
    float u = x - cx - 0.5f;
    float v = y - cy - 0.5f;
    float w = z - cz - 0.5f;
    int region = (int)s.v[0];
    if (region != 0) {
        int turn = (int)(mp_hash((int)cx, (int)cy, 0, region) & 3u);
        float ou = u;
        if (turn == 1) { u = -v; v = ou; }
        else if (turn == 2) { u = -u; v = -v; }
        else if (turn == 3) { u = v; v = -ou; }
    }
    float r = MP_SQRT(u * u + v * v);
    float a = MP_ATAN2(v, u);
    float t = 0.0f;
    if (pattern == 200) {
        // Box: region seed, smooth. Brightest in the middle, falling to the cell's sides.
        t = mp_soften(mp_clamp01(1.0f - 2.0f * MP_MAX(MP_ABS(u), MP_ABS(v))), s.v[1]);
    } else if (pattern == 201) {
        // Circular: region seed, smooth. Brightest in the middle, falling with the distance from it.
        t = mp_soften(mp_clamp01(1.0f - 2.0f * r), s.v[1]);
    } else if (pattern == 202) {
        // Corners: region seed, smooth. Brightest at the corners.
        float dx = 0.5f - MP_ABS(u);
        float dy = 0.5f - MP_ABS(v);
        t = mp_soften(mp_clamp01(1.0f - MP_SQRT(dx * dx + dy * dy) / 0.70710678f), s.v[1]);
    } else if (pattern == 203) {
        // Cubic: region seed, low, high. Brightest in the middle of each cube.
        t = mp_range(1.0f - 2.0f * MP_MAX(MP_MAX(MP_ABS(u), MP_ABS(v)), MP_ABS(w)), s.v[1], s.v[2]);
    } else if (pattern == 204) {
        // Dimples: region seed. A round dent in each cell.
        t = 1.0f - MP_SQRT(mp_clamp01(1.0f - 4.0f * r * r));
    } else if (pattern == 205) {
        // Grid: region seed. Lines along the cells' sides.
        t = mp_step(MP_MAX(MP_ABS(u), MP_ABS(v)), 0.45f, 0.03f);
    } else if (pattern == 206) {
        // Iris: region seed, rays, smooth. Rays round the middle, fading out to the sides.
        t = mp_soften((0.5f + 0.5f * MP_COS(MP_MAX(1.0f, s.v[1]) * a + 20.0f * r)) * mp_clamp01(1.0f - 2.0f * r), s.v[2]);
    } else if (pattern == 207) {
        // Linear: region seed, smooth. A ramp across the cell.
        t = mp_soften(mp_clamp01(u + 0.5f), s.v[1]);
    } else if (pattern == 208) {
        // Polygon: region seed, sides, smooth. A polygon of so many sides, brightest in the middle.
        float n = MP_MAX(3.0f, MP_FLOOR(s.v[1]));
        float sector = 2.0f * MP_PI / n;
        float around = a + MP_PI;
        float m = around - sector * MP_FLOOR(around / sector) - 0.5f * sector;
        float rho = 0.5f * MP_COS(MP_PI / n) / MP_COS(m);
        t = mp_soften(mp_clamp01(1.0f - r / rho), s.v[2]);
    } else if (pattern == 209) {
        // Radial: region seed, rays, smooth. Rays from the middle.
        t = mp_soften(0.5f + 0.5f * MP_COS(MP_MAX(1.0f, s.v[1]) * a), s.v[2]);
    } else if (pattern == 210) {
        // Ring: region seed, width, smooth. A ring round the middle.
        t = mp_soften(mp_clamp01(1.0f - MP_ABS(r - 0.3f) / MP_MAX(0.005f, s.v[1] * 0.5f)), s.v[2]);
    } else if (pattern == 211) {
        // RndLinear: region seed, random type. A ramp across each cell, every cell's its own way.
        float angle = 2.0f * MP_PI * mp_random((int)cx, (int)cy, 0, region + 1);
        t = mp_clamp01(0.5f + u * MP_COS(angle) + v * MP_SIN(angle));
    } else if (pattern == 212) {
        // Spiral: region seed, rays, smooth. Arms winding out from the middle.
        t = mp_soften(0.5f + 0.5f * MP_COS(MP_MAX(1.0f, s.v[1]) * a + 8.0f * MP_PI * r), s.v[2]);
    } else if (pattern == 213) {
        // Star: region seed, radius, points, extent, smooth. A star of so many points.
        float points = MP_MAX(2.0f, MP_FLOOR(s.v[2]));
        float spike = MP_POW(0.5f + 0.5f * MP_COS(points * a), 4.0f);
        float rho = MP_MAX(0.01f, s.v[1]) * 0.5f * (1.0f + MP_MAX(0.0f, s.v[3]) * spike);
        t = mp_soften(mp_clamp01(1.0f - r / rho), s.v[4]);
    }
    return mp_range(t, s.v[14], s.v[15]);
}

// Enhance: Modo's Waveforms, 220 on: a wave along the locator's X axis, amplitude times the waveform plus shift, the
// waveform repeating frequency times a unit and moved along by phase.
MP_HEAVY float mp_enhance_waveform(int pattern, float x, MP_Settings s) {
    float u = mp_fract(x * s.v[11] + s.v[12]);
    float w = u;
    if (pattern == 220) w = mp_gain(mp_bias(u, s.v[0]), s.v[1]);                       // BiasGain: bias, gain
    else if (pattern == 221) {                                                          // Fresnel: IOR
        float f0 = (MP_MAX(1.0001f, s.v[0]) - 1.0f) / (MP_MAX(1.0001f, s.v[0]) + 1.0f);
        w = f0 * f0 + (1.0f - f0 * f0) * MP_POW(u, 5.0f);
    } else if (pattern == 222) w = MP_POW(u, 1.0f / MP_MAX(0.01f, s.v[0]));            // Gamma: amount
    else if (pattern == 223) {                                                          // Gaussian: spread
        float d = (u - 0.5f) / MP_MAX(0.001f, s.v[0]);
        w = MP_EXP(-0.5f * d * d);
    } else if (pattern == 224) w = (u >= s.v[0] && u <= s.v[1]) ? 1.0f : 0.0f;          // Impulse: start, end
    else if (pattern == 225) {                                                          // Noise: seed, octaves, scale, increment
        float f = s.v[11] / MP_MAX(0.01f, s.v[2]);
        w = mp_unit(mp_fbm(0, x * f + s.v[12], 0.5f, 0.5f, (int)s.v[0], MP_MAX(1.0f, s.v[1]), 2.0f, s.v[3], 0));
    } else if (pattern == 227) w = MP_SQRT(mp_clamp01(1.0f - (2.0f * u - 1.0f) * (2.0f * u - 1.0f)));  // Rounded
    else if (pattern == 228) w = mp_smooth((u - s.v[0]) / MP_MAX(0.001f, s.v[1] - s.v[0])); // SCurve: start, end
    else if (pattern == 229) w = s.v[0] != 0.0f ? 1.0f - u : u;                         // SawTooth: falling
    else if (pattern == 230) w = 1.0f - MP_SQRT(mp_clamp01(1.0f - (2.0f * u - 1.0f) * (2.0f * u - 1.0f)));  // Scallop
    else if (pattern == 231) w = 0.5f + 0.5f * MP_SIN(2.0f * MP_PI * u);              // Sine
    else if (pattern == 232) {                                                          // Smooth: rising, falling
        float up = mp_smooth(u / MP_MAX(0.001f, s.v[0]));
        float down = 1.0f - mp_smooth((u - (1.0f - s.v[1])) / MP_MAX(0.001f, s.v[1]));
        w = MP_MIN(up, down);
    } else if (pattern == 233) w = mp_step(u, s.v[0], s.v[2]) * (1.0f - mp_step(u, s.v[1], s.v[2]));  // SmoothImpulse
    else if (pattern == 234) w = mp_step(u, s.v[0], s.v[1]);                            // SmoothStep: start, smooth
    else if (pattern == 235) {                                                          // Staircase: levels, slant
        float n = MP_MAX(1.0f, MP_FLOOR(s.v[0]));
        float k = MP_FLOOR(u * n);
        w = (k + mp_clamp01(s.v[1]) * (u * n - k)) / n;
    }
    return mp_range(s.v[13] + s.v[10] * w, s.v[14], s.v[15]);
}

// Enhance: Modo's Skins, 240 on. Their noise reads s.v[16] to s.v[20] (seed, octaves, magnitude, scale, detail).
MP_HEAVY float mp_enhance_skins(int pattern, int type, float x, float y, float z, MP_Settings s) {
    int seed = (int)s.v[0];
    float f = 1.0f / MP_MAX(0.01f, s.v[19]);
    float octaves = MP_MAX(1.0f, s.v[17]);
    float h = 1.0f - 0.5f * mp_clamp01(s.v[20]);
    float fx = x * f;
    float fy = y * f;
    float fz = z * f;
    float t = 0.0f;
    if (pattern == 240 || pattern == 248) {
        // Camo and Pastella: seed, clip, levels. Blotches of flat colour (Pastella's softer).
        float n = mp_unit(mp_fbm(type, fx, fy, fz, seed, octaves, 2.0f, h, 0) * (1.0f + 0.5f * s.v[18]));
        if (pattern == 248) n = mp_smooth(n);
        t = mp_levels(mp_clip(n, s.v[1]), s.v[2]);
    } else if (pattern == 241) {
        // Crumpled: seed, dispersion, clip, levels. Creases, as of crumpled paper.
        float push = s.v[1] * 0.3f * mp_basis(type, fx * 0.5f, fy * 0.5f, fz * 0.5f, seed + 3);
        float n = 1.0f - MP_ABS(mp_fbm(type, fx + push, fy - push, fz, seed, octaves, 2.0f, h, 0));
        t = mp_levels(mp_clip(n * n * (1.0f + 0.3f * s.v[18]), s.v[2]), s.v[3]);
    } else if (pattern == 242) {
        // DinoSkin: seed, dispersion, clip, levels. Rounded scales.
        float e = mp_worley(fx, fy, fz, seed, mp_clamp01(s.v[1]), 4);
        float bump = 0.7f + 0.3f * mp_unit(s.v[18] * mp_fbm(type, fx * 4.0f, fy * 4.0f, fz * 4.0f, seed + 1, octaves, 2.0f, h, 0));
        t = mp_levels(mp_clip(mp_smooth(e * 3.0f) * bump, s.v[2]), s.v[3]);
    } else if (pattern == 243) {
        // Disease: seed, octaves, frequency, increment. Blotches where the noise runs high.
        float n = mp_fbm(type, x, y, z, seed, MP_MAX(1.0f, s.v[1]), s.v[2], s.v[3], 0);
        t = mp_step(n, 0.15f, 0.08f);
    } else if (pattern == 244) {
        // FrogSkin: seed, dispersion, clip, levels, spot size, edge. Spots ringed by a darker edge.
        float d = mp_worley(fx, fy, fz, seed, mp_clamp01(s.v[1]), 0);
        float radius = MP_MAX(0.01f, s.v[4]) * 0.5f;
        float spot = 1.0f - mp_step(d, radius, 0.02f);
        float ring = 1.0f - mp_step(d, radius + MP_MAX(0.0f, s.v[5]) * 0.5f, 0.02f);
        float ground = 0.3f * mp_unit(mp_fbm(type, fx * 2.0f, fy * 2.0f, fz * 2.0f, seed + 1, octaves, 2.0f, h, 0) * s.v[18]);
        t = mp_levels(mp_clip(MP_MAX(spot, ground * (1.0f - ring) + 0.05f * ring), s.v[2]), s.v[3]);
    } else if (pattern == 245) {
        // GrainyWood: seed, ringiness, grain frequency, graininess. Rings round the Y axis, with grain along it.
        float wander = 0.3f * s.v[18] * mp_fbm(type, fx, fy * 0.2f, fz, seed, octaves, 2.0f, h, 0);
        float rings = 0.5f + 0.5f * MP_COS(2.0f * MP_PI * (MP_SQRT(x * x + z * z) * MP_MAX(0.1f, s.v[1]) + wander));
        float grain = mp_basis(type, x * MP_MAX(0.1f, s.v[2]) * 10.0f, y * 0.5f, z * MP_MAX(0.1f, s.v[2]) * 10.0f, seed + 5);
        t = mp_clamp01(0.85f * rings + 0.15f + mp_clamp01(s.v[3]) * 0.3f * grain);
    } else if (pattern == 246) {
        // Leather: seed, low, high, levels. Pebbled hide parted by creases.
        float e = mp_worley(fx, fy, fz, seed, 1.0f, 4) + 0.1f * s.v[18] * mp_basis(type, fx * 3.0f, fy * 3.0f, fz * 3.0f, seed + 2);
        t = mp_levels(mp_range(e, s.v[1], s.v[2] > s.v[1] ? s.v[2] : s.v[1] + 0.3f), s.v[3]);
    } else if (pattern == 247) {
        // Monster: seed, dispersion, clip, small scales. Big scales covered in small ones.
        float big = mp_smooth(mp_worley(fx, fy, fz, seed, mp_clamp01(s.v[1]), 4) * 3.0f);
        float little = mp_smooth(mp_worley(fx * 4.0f, fy * 4.0f, fz * 4.0f, seed + 1, mp_clamp01(s.v[1]), 4) * 3.0f);
        t = mp_clip(big * mp_lerp(1.0f, 0.4f + 0.6f * little, mp_clamp01(s.v[3])), s.v[2]);
    } else if (pattern == 249) {
        // Peened: seed, dispersion, clip, levels. Round dents, as of a hammered surface.
        float d = mp_clamp01(mp_worley(fx, fy, fz, seed, mp_clamp01(s.v[1]), 0) * 1.6f);
        t = mp_levels(mp_clip(d * d, s.v[2]), s.v[3]);
    } else if (pattern == 250) {
        // Scratches: seed, dispersion, clip, levels. Fine straight scratches at random, as many as dispersion.
        float cx = MP_FLOOR(fx);
        float cy = MP_FLOOR(fy);
        float best = 0.0f;
        float width = 0.012f * (1.0f + MP_MAX(0.0f, s.v[18]));
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i;
                int b = (int)cy + j;
                float ox = (float)a + mp_random(a, b, 0, seed);
                float oy = (float)b + mp_random(a, b, 1, seed);
                float angle = MP_PI * mp_random(a, b, 2, seed);
                float dx = MP_COS(angle);
                float dy = MP_SIN(angle);
                float along = MP_MAX(-0.6f, MP_MIN(0.6f, (fx - ox) * dx + (fy - oy) * dy));
                float qx = fx - (ox + along * dx);
                float qy = fy - (oy + along * dy);
                float present = mp_random(a, b, 7, seed) <= MP_MAX(0.05f, mp_clamp01(s.v[1])) ? 1.0f : 0.0f;
                best = MP_MAX(best, present * (0.5f + 0.5f * mp_random(a, b, 3, seed)) * (1.0f - mp_step(MP_SQRT(qx * qx + qy * qy), width, width * 0.5f)));
            }
        }
        t = mp_levels(mp_clip(best, s.v[2]), s.v[3]);
    }
    return mp_range(t, s.v[14], s.v[15]);
}

// Enhance: Modo's Space patterns, 260 on. Most read noise seed, octaves, frequency and increment from s.v[0]; their
// flares and glints lie on the locator's XY plane, about its middle, and planets and their rings about its Y axis.
MP_HEAVY float mp_enhance_space(int pattern, int type, float x, float y, float z, MP_Settings s) {
    int seed = (int)s.v[0];
    float octaves = MP_MAX(1.0f, s.v[1]);
    float r2 = MP_SQRT(x * x + y * y);
    float a2 = MP_ATAN2(y, x);
    float t = 0.0f;
    if (pattern == 260) {
        // Blast: rays, variance, disturbance. Rays bursting out from the middle.
        float n = mp_fbm(type, x, y, z, seed, octaves, s.v[2], s.v[3], 0);
        float rays = MP_POW(0.5f + 0.5f * MP_COS(MP_MAX(1.0f, s.v[4]) * a2 + 3.0f * s.v[5] * n), 6.0f);
        t = mp_clamp01((rays + 0.3f) * (1.0f - r2) + 0.3f * s.v[6] * n);
    } else if (pattern == 261 || pattern == 265) {
        // Coriolis: twist. Hurricane: twist, storm, eye. Clouds wound round the Y axis (a hurricane's about an eye).
        float r = MP_SQRT(x * x + z * z);
        float turn = pattern == 261 ? s.v[4] * y : s.v[4] * (1.0f - mp_clamp01(r));
        float rx = x * MP_COS(turn) - z * MP_SIN(turn);
        float rz = x * MP_SIN(turn) + z * MP_COS(turn);
        float clouds = mp_unit(mp_fbm(type, rx * 2.0f, y, rz * 2.0f, seed, octaves, s.v[2], s.v[3], 0));
        if (pattern == 265) clouds = clouds * mp_step(r, MP_MAX(0.0f, s.v[6]), 0.02f) * (1.0f - mp_step(r, MP_MAX(0.01f, s.v[5]), 0.1f));
        t = clouds;
    } else if (pattern == 262) {
        // Flare: rays. A glow with streaks.
        float glow = 1.0f / (1.0f + 8.0f * r2 * r2);
        float streaks = MP_POW(MP_ABS(MP_COS(MP_MAX(1.0f, s.v[4]) * 0.5f * a2)), 20.0f) * MP_EXP(-2.0f * r2);
        float n = 0.6f + 0.4f * mp_unit(mp_fbm(type, a2 * 3.0f, r2, 0.5f, seed, octaves, s.v[2], s.v[3], 0));
        t = mp_clamp01(glow + streaks * n);
    } else if (pattern == 263) {
        // GasGiant: noise seed, frequency. Bands along the Y axis, stirred by noise.
        float stir = 2.5f * mp_fbm(type, x * 2.0f, y * 0.5f, z * 2.0f, seed, 4.0f, 2.0f, 0.5f, 0);
        t = 0.5f + 0.5f * MP_COS(y * MP_MAX(0.1f, s.v[1]) * 6.0f + stir);
    } else if (pattern == 264) {
        // Glint: noise seed, rays, glare, falloff, radius. A star's glint.
        float core = 1.0f - mp_smooth(r2 / MP_MAX(0.001f, s.v[4]));
        float rays = MP_POW(MP_ABS(MP_COS(MP_MAX(1.0f, s.v[1]) * 0.5f * a2)), 1.0f + 40.0f * (1.0f - mp_clamp01(s.v[2]))) * MP_EXP(-MP_MAX(0.0f, s.v[3]) * r2);
        t = mp_clamp01(core + rays);
    } else if (pattern == 266) {
        // Nurnies (type: the cell order): noise seed, octaves, frequency, increment. Stepped cells, as of
        // machinery.
        t = MP_FLOOR(mp_clamp01(mp_worley_fractal(x, y, z, seed, 1.0f, type, octaves, s.v[2], s.v[3])) * 3.999f) / 3.0f;
    } else if (pattern == 267) {
        // Planet: latitude. Land and sea, ice towards the poles.
        float land = mp_unit(mp_fbm(type, x, y, z, seed, octaves, s.v[2], s.v[3], 0));
        float extent = MP_MAX(1e-4f, MP_SQRT(x * x + y * y + z * z));
        t = MP_MAX(land, mp_step(MP_ABS(y) / extent, mp_clamp01(s.v[4]), 0.03f));
    } else if (pattern == 268) {
        // PlanetClouds: global scale, circulation, threshold. Clouds stretched by the planet's turning.
        float g = MP_MAX(0.01f, s.v[4]);
        float turn = s.v[5] * y;
        float rx = x * MP_COS(turn) - z * MP_SIN(turn);
        float rz = x * MP_SIN(turn) + z * MP_COS(turn);
        float c = mp_unit(mp_fbm(type, rx * g, y * g * 2.0f, rz * g, seed, octaves, s.v[2], s.v[3], 0));
        t = mp_step(c, s.v[6], 0.05f);
    } else if (pattern == 269) {
        // Rings: size, intensity, inner radius, outer radius. A planet's rings about the Y axis.
        float r = MP_SQRT(x * x + z * z);
        float within = mp_step(r, s.v[6], 0.005f) * (1.0f - mp_step(r, s.v[7], 0.005f));
        float bands = mp_unit(mp_fbm(type, r * MP_MAX(0.01f, s.v[4]) * 10.0f, 0.5f, 0.5f, seed, octaves, s.v[2], s.v[3], 0));
        t = within * mp_clamp01(bands * s.v[5]);
    } else if (pattern == 270) {
        // StarField: noise seed, density. Stars at random.
        float k = 20.0f;
        float cx = MP_FLOOR(x * k);
        float cy = MP_FLOOR(y * k);
        float cz = MP_FLOOR(z * k);
        int a = (int)cx;
        int b = (int)cy;
        int c = (int)cz;
        if (mp_random(a, b, c, seed) < mp_clamp01(s.v[1]) * 0.1f) {
            float dx = x * k - cx - mp_random(a, b, c, seed + 1);
            float dy = y * k - cy - mp_random(a, b, c, seed + 2);
            float dz = z * k - cz - mp_random(a, b, c, seed + 3);
            t = mp_random(a, b, c, seed + 4) * (1.0f - mp_smooth(MP_SQRT(dx * dx + dy * dy + dz * dz) / 0.15f));
        }
    } else if (pattern == 271) {
        // Swirl: scale, magnitude. Noise stirred by noise.
        float g = MP_MAX(0.01f, s.v[4]);
        float push = s.v[5] * mp_fbm(type, x * g, y * g, z * g, seed + 7, octaves, s.v[2], s.v[3], 0);
        t = mp_unit(mp_fbm(type, x + push, y - push, z + push, seed, octaves, s.v[2], s.v[3], 0));
    } else if (pattern == 272) {
        // Terra: mountainous land, sharp ridges.
        float n = 1.0f - MP_ABS(mp_fbm(type, x, y, z, seed, octaves, s.v[2], s.v[3], 0));
        t = n * n;
    } else if (pattern == 273) {
        // Windows: seed, horizontal min and max, vertical min and max, windows on, smooth. A building's windows,
        // a window a unit, some lit.
        float cx = MP_FLOOR(x);
        float cy = MP_FLOOR(y);
        float u = x - cx;
        float v = y - cy;
        float soft = MP_MAX(0.0f, s.v[6]) * 0.05f;
        float pane = mp_step(u, s.v[1], soft) * (1.0f - mp_step(u, s.v[2], soft)) * mp_step(v, s.v[3], soft) * (1.0f - mp_step(v, s.v[4], soft));
        t = pane * (mp_random((int)cx, (int)cy, 0, seed) < mp_clamp01(s.v[5]) ? 1.0f : 0.2f);
    }
    return mp_range(t, s.v[14], s.v[15]);
}

// Enhance: Modo's Panels, 280 on: seed, disturb, distribution and grout, then their own; panels two units by one on
// the locator's XY plane, each row shifted at random by as much as disturb, with the Tiles' edge noise. 1 is a panel,
// 0 the seams.
MP_HEAVY float mp_enhance_panels(int pattern, int type, float px, float py, float pz, MP_Settings s) {
    float x = px + mp_tile_warp(type, px, py, pz, s, 0);
    float y = py + mp_tile_warp(type, px, py, pz, s, 1);
    int seed = (int)s.v[0];
    int row = (int)MP_FLOOR(y);
    float xs = x + 2.0f * s.v[1] * mp_random(row, 0, 9, seed);
    float col = MP_FLOOR(xs / 2.0f);
    float u = xs - col * 2.0f;
    float v = y - (float)row;
    float e = MP_MIN(MP_MIN(u, 2.0f - u), MP_MIN(v, 1.0f - v));
    float face = mp_tile_face(e, s.v[3], pattern == 281 ? s.v[4] : 0.02f, mp_random((int)col, row, 0, seed), s.v[2]);
    float t = face;
    if (pattern == 280) {
        // Peel: noise seed, peel width, peel magnitude, peel scale. Paint peeling off the panels, a bright lip at
        // the edge of each bare patch.
        float n = mp_unit(mp_fbm(type, x / MP_MAX(0.01f, s.v[7]), y / MP_MAX(0.01f, s.v[7]), pz, (int)s.v[4], 4.0f, 2.0f, 0.5f, 0));
        float edge = mp_clamp01(s.v[6]) * 0.6f;
        float bare = 1.0f - mp_step(n, edge, 0.01f);
        float lip = (1.0f - bare) * (1.0f - mp_step(n, edge + MP_MAX(0.001f, s.v[5]) * 0.1f, 0.005f));
        t = face * (bare != 0.0f ? 0.35f : (lip > 0.5f ? 1.0f : 0.75f));
    } else if (pattern == 282 || pattern == 283) {
        // RivetRust: rivet seed, rivet radius, rust radius, rust scale, rivet distribution, rivets. Rivets: rivet
        // seed, rivet radius, rivet bevel, slot width, rivet distribution, rivets. Rivets along each panel's top
        // and bottom, rust round them, or a slot across them.
        float n = MP_MAX(1.0f, MP_FLOOR(s.v[9]));
        float spacing = 2.0f / n;
        float ru = (MP_FLOOR(u / spacing) + 0.5f) * spacing;
        float rv = v < 0.5f ? 0.12f : 0.88f;
        float du = u - ru;
        float dv = v - rv;
        float d = MP_SQRT(du * du + dv * dv);
        float radius = MP_MAX(0.005f, s.v[5]);
        int id = (int)MP_FLOOR(u / spacing) + (v < 0.5f ? 0 : 64);
        float shade = 1.0f - mp_clamp01(s.v[8]) * mp_random((int)col * 128 + id, row, 1, (int)s.v[4]);
        if (pattern == 282) {
            float rust = (1.0f - mp_smooth(d / MP_MAX(0.001f, s.v[6]))) * mp_unit(mp_fbm(type, x / MP_MAX(0.01f, s.v[7]), y / MP_MAX(0.01f, s.v[7]), pz, (int)s.v[4], 3.0f, 2.0f, 0.5f, 0));
            t = MP_MAX(face * (1.0f - 0.6f * rust), (1.0f - mp_step(d, radius, 0.01f)) * shade);
        } else {
            float dome = d < radius ? mp_smooth((radius - d) / MP_MAX(0.001f, s.v[6] * radius)) * shade : 0.0f;
            if (d < radius && MP_ABS(du) < MP_MAX(0.0f, s.v[7]) * 0.5f * radius) dome = dome * 0.3f;
            t = MP_MAX(face, dome);
        }
    } else if (pattern == 284) {
        // Rust: noise seed, intensity, rust scale. Rust over the panels.
        float rust = mp_smooth(mp_unit(mp_fbm(type, x / MP_MAX(0.01f, s.v[6]), y / MP_MAX(0.01f, s.v[6]), pz, (int)s.v[4], 5.0f, 2.0f, 0.6f, 0)) * 1.5f - 0.25f);
        t = face * (1.0f - mp_clamp01(s.v[5]) * rust);
    } else if (pattern == 285) {
        // Smear: noise seed, intensity, smear scale, direction (up, down, right, left). Streaks run from the seams.
        int direction = (int)s.v[7];
        float across = direction >= 2 ? v : u;
        float from = direction == 0 ? v : (direction == 1 ? 1.0f - v : (direction == 2 ? u * 0.5f : 1.0f - u * 0.5f));
        float g = 1.0f / MP_MAX(0.01f, s.v[6]);
        float streak = mp_unit(mp_fbm(type, across * g * 8.0f, from * 0.3f, pz, (int)s.v[4], 4.0f, 2.0f, 0.5f, 0));
        t = face * (1.0f - 0.7f * mp_clamp01(s.v[5]) * streak * (1.0f - from));
    }
    return t;
}

// Ripples spreading from a point: how high the water is at r from it, the wave front at front, a band of ripples
// ripple wide behind it, wave long.
MP_FUNC float mp_ripple(float r, float front, float ripple, float wave) {
    float behind = front - r;
    if (behind < 0.0f || behind > MP_MAX(0.001f, ripple)) return 0.0f;
    float envelope = MP_SIN(MP_PI * behind / MP_MAX(0.001f, ripple));
    return envelope * MP_COS(2.0f * MP_PI * behind / MP_MAX(0.001f, wave));
}

// Enhance: Modo's Water, 290 on, on the locator's XZ plane: the water's height, 0.5 still.
MP_HEAVY float mp_enhance_water(int pattern, int type, float x, float y, float z, MP_Settings s) {
    float t = 0.5f;
    if (pattern == 290) {
        // DripDrop: min and max wave, min and max radius, min and max ripple, time. A drop's ripples at the
        // middle, spreading out each second.
        float phase = mp_fract(s.v[6]);
        float r = MP_SQRT(x * x + z * z);
        float front = mp_lerp(s.v[2], s.v[3], phase);
        t = 0.5f + 0.5f * (1.0f - phase) * mp_ripple(r, front, mp_lerp(s.v[4], s.v[5], phase), mp_lerp(s.v[0], s.v[1], phase));
    } else if (pattern == 291) {
        // Rain: seed, drops, spread, min and max wave, radius and ripple, min and max life, time. Drops falling
        // all over, each spreading its ripples for its life.
        int seed = (int)s.v[0];
        float cell = MP_MAX(0.01f, s.v[2]) / MP_SQRT(MP_MAX(1.0f, s.v[1]));
        float cx = MP_FLOOR(x / cell);
        float cz = MP_FLOOR(z / cell);
        float sum = 0.0f;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i;
                int b = (int)cz + j;
                float life = MP_MAX(0.01f, mp_lerp(s.v[9], s.v[10], mp_random(a, b, 0, seed)));
                float phase = mp_fract(s.v[11] / life + mp_random(a, b, 1, seed));
                float dx = x - ((float)a + mp_random(a, b, 2, seed)) * cell;
                float dz = z - ((float)b + mp_random(a, b, 3, seed)) * cell;
                float r = MP_SQRT(dx * dx + dz * dz);
                float front = mp_lerp(s.v[5], s.v[6], phase);
                sum = sum + (1.0f - phase) * mp_ripple(r, front, mp_lerp(s.v[7], s.v[8], phase), mp_lerp(s.v[3], s.v[4], mp_random(a, b, 4, seed)));
            }
        }
        t = mp_clamp01(0.5f + 0.5f * sum);
    } else if (pattern == 292) {
        // Ripples: seed, waves, spread, extent, time. Rings spreading from several points.
        int seed = (int)s.v[0];
        int count = (int)MP_MAX(1.0f, MP_MIN(8.0f, s.v[1]));
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            if (k < count) {
                float dx = x - s.v[2] * (mp_random(k, 0, 0, seed) - 0.5f);
                float dz = z - s.v[2] * (mp_random(k, 0, 1, seed) - 0.5f);
                sum = sum + MP_COS(2.0f * MP_PI * (MP_SQRT(dx * dx + dz * dz) - s.v[4]) / MP_MAX(0.001f, s.v[3]));
            }
        }
        t = 0.5f + 0.5f * sum / (float)count;
    } else if (pattern == 293) {
        // Surf: falloff, spread, extent, time. Breakers rolling in along X.
        float phase = (x + s.v[3]) / MP_MAX(0.001f, s.v[2]) + s.v[1] * 0.2f * mp_basis(type, z * 0.5f, 0.5f, 0.5f, 3);
        float crest = MP_POW(0.5f + 0.5f * MP_COS(2.0f * MP_PI * phase), 3.0f);
        t = crest * MP_EXP(-MP_MAX(0.0f, s.v[0]) * MP_ABS(x));
    } else if (pattern == 294) {
        // Waves: seed, extent, time. Waves from four ways at once.
        int seed = (int)s.v[0];
        float sum = 0.0f;
        for (int k = 0; k < 4; k++) {
            float angle = 2.0f * MP_PI * mp_random(k, 1, 0, seed);
            float extent = MP_MAX(0.001f, s.v[1]) * (0.6f + 0.8f * mp_random(k, 2, 0, seed));
            sum = sum + MP_COS(2.0f * MP_PI * ((x * MP_COS(angle) + z * MP_SIN(angle)) - s.v[2]) / extent);
        }
        t = 0.5f + 0.125f * sum;
    } else if (pattern == 295) {
        // WindyWaves (type: the waves' noise type): wave noise seed, octaves, frequency, increment, amplitude;
        // wind noise type, seed, octaves, frequency, increment, amplitude. Swell, with ripples raised by the wind.
        float swell = s.v[4] * mp_fbm(type, x, 0.5f, z, (int)s.v[0], MP_MAX(1.0f, s.v[1]), s.v[2], s.v[3], 0);
        float wind = s.v[10] * mp_fbm((int)s.v[5], x * 4.0f, 0.5f, z * 4.0f, (int)s.v[6], MP_MAX(1.0f, s.v[7]), s.v[8], s.v[9], 0);
        t = 0.5f + 0.5f * (swell + wind);
    }
    return mp_range(t, s.v[14], s.v[15]);
}

// The segments of a seven-segment digit, 0 to F: bit 0 the top, 1 top right, 2 bottom right, 3 the bottom, 4 bottom
// left, 5 top left, 6 the middle.
MP_FUNC int mp_segments(int digit) {
    if (digit == 0) return 0x3F;
    if (digit == 1) return 0x06;
    if (digit == 2) return 0x5B;
    if (digit == 3) return 0x4F;
    if (digit == 4) return 0x66;
    if (digit == 5) return 0x6D;
    if (digit == 6) return 0x7D;
    if (digit == 7) return 0x07;
    if (digit == 8) return 0x7F;
    if (digit == 9) return 0x6F;
    if (digit == 10) return 0x77;
    if (digit == 11) return 0x7C;
    if (digit == 12) return 0x39;
    if (digit == 13) return 0x5E;
    if (digit == 14) return 0x79;
    return 0x71;
}

// Enhance: Modo's Display patterns, 300 on, on the locator's XY plane.
MP_HEAVY float mp_enhance_display(int pattern, float x, float y, MP_Settings s) {
    float t = 0.0f;
    if (pattern == 300 || pattern == 301) {
        // Counter1 and Counter2 (slanted): characters, character gap, scroll, segment thickness, segment gap,
        // leading zeros, decimal point, base, format, then the scene's time and frame rate. Seven-segment digits a
        // unit wide and two high, counting frames, seconds or SMPTE time code.
        int chars = (int)MP_MAX(1.0f, MP_MIN(16.0f, s.v[0]));
        float cellw = 1.0f + MP_MAX(0.0f, s.v[1]);
        int kind = (int)s.v[7];
        int base = kind == 1 ? 16 : (kind == 2 ? 8 : (kind == 3 ? 2 : 10));
        int format = (int)s.v[8];
        float fps = MP_MAX(1.0f, s.v[10]);
        int frame = (int)MP_FLOOR(s.v[9] * fps + 0.5f);
        int number = frame;
        int point = (int)s.v[6];
        int padded = s.v[5] != 0.0f ? 1 : 0;
        if (format == 1) { number = (int)MP_FLOOR(s.v[9] * 100.0f); point = 2; base = 10; }
        else if (format >= 2) {
            int rate = format == 2 ? 24 : (format == 3 ? 25 : 30);
            int seconds = frame / rate;
            number = ((seconds / 3600) * 10000 + ((seconds / 60) % 60) * 100 + seconds % 60) * 100 + frame % rate;
            base = 10;
            padded = 1;
        }
        float u = x + s.v[2];
        float v = y;
        if (pattern == 301) u = u - 0.15f * (v - 1.0f);
        int ci = (int)MP_FLOOR(u / cellw);
        if (ci >= 0 && ci < chars && v >= 0.0f && v <= 2.0f) {
            int k = chars - 1 - ci;
            int quotient = number < 0 ? -number : number;
            for (int i = 0; i < 16; i++) {
                if (i < k) quotient = quotient / base;
            }
            int digit = quotient % base;
            float cu = u - (float)ci * cellw;
            float th = MP_MAX(0.02f, s.v[3]) * 0.5f;
            float sg = MP_MAX(0.0f, s.v[4]);
            int shown = (padded != 0 || k == 0 || quotient != 0 || (point > 0 && k <= point)) ? 1 : 0;
            if (shown != 0 && cu <= 1.0f) {
                int bits = mp_segments(digit);
                float lit = 0.0f;
                int across = (cu >= th + sg && cu <= 1.0f - th - sg) ? 1 : 0;
                if (across != 0 && MP_ABS(v - 2.0f) < th && (bits & 1) != 0) lit = 1.0f;
                if (across != 0 && MP_ABS(v - 1.0f) < th && (bits & 64) != 0) lit = 1.0f;
                if (across != 0 && MP_ABS(v) < th && (bits & 8) != 0) lit = 1.0f;
                int upper = (v >= 1.0f + th + sg && v <= 2.0f - th - sg) ? 1 : 0;
                int lower = (v >= th + sg && v <= 1.0f - th - sg) ? 1 : 0;
                if (MP_ABS(cu) < th && upper != 0 && (bits & 32) != 0) lit = 1.0f;
                if (MP_ABS(cu) < th && lower != 0 && (bits & 16) != 0) lit = 1.0f;
                if (MP_ABS(cu - 1.0f) < th && upper != 0 && (bits & 2) != 0) lit = 1.0f;
                if (MP_ABS(cu - 1.0f) < th && lower != 0 && (bits & 4) != 0) lit = 1.0f;
                t = lit;
            }
            if (point > 0 && k == point) {
                float px = cu - (1.0f + 0.5f * (cellw - 1.0f));
                if (MP_SQRT(px * px + v * v) < th * 1.2f) t = 1.0f;
            }
        }
    } else if (pattern == 302) {
        // UVLEDs: seed, gap, width, LEDs on, time, LEDs, shape (round or square). A board of lights, as many as
        // LEDs a unit, some on, changing four times a second.
        float n = MP_MAX(1.0f, s.v[5]);
        float cx = MP_FLOOR(x * n);
        float cy = MP_FLOOR(y * n);
        float du = x * n - cx - 0.5f;
        float dv = y * n - cy - 0.5f;
        float size = 0.5f * mp_clamp01(1.0f - s.v[1]) * (s.v[2] > 0.0f ? mp_clamp01(s.v[2]) : 1.0f);
        float d = (int)s.v[6] == 1 ? MP_MAX(MP_ABS(du), MP_ABS(dv)) : MP_SQRT(du * du + dv * dv);
        int tick = (int)MP_FLOOR(s.v[4] * 4.0f);
        float on = mp_random((int)cx, (int)cy, tick, (int)s.v[0]) < mp_clamp01(s.v[3]) ? 1.0f : 0.0f;
        t = on * (1.0f - mp_step(d, size, 0.02f));
    }
    return t;
}

// ---- Modo's Gabor and Poisson noise, and Enhance: Modo's RegionalHSV -------------------------------------------------

// Modo's Gabor noise: orientation, frequency, bandwidth and offset, read as Lagae's sparse Gabor convolution has them,
// in 256ths of a unit of the locator's space: impulses scattered at random, each a wave of the frequency along the
// orientation (turned about the locator's Z axis), fading over about 1 / bandwidth.
MP_HEAVY float mp_gabor(float x, float y, float z, float orientation, float frequency, float bandwidth, int seed) {
    float k = 256.0f;
    float a = MP_MAX(0.005f, bandwidth);
    float radius = 1.0f / a;
    float px = x * k / radius;
    float py = y * k / radius;
    float pz = z * k / radius;
    float cx = MP_FLOOR(px);
    float cy = MP_FLOOR(py);
    float cz = MP_FLOOR(pz);
    float ox = MP_COS(orientation);
    float oy = MP_SIN(orientation);
    float sum = 0.0f;
    for (int kz = -1; kz <= 1; kz++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int ca = (int)cx + kx;
                int cb = (int)cy + ky;
                int cc = (int)cz + kz;
                for (int i = 0; i < 4; i++) {
                    int s = seed * 7 + i * 4;
                    float dx = (px - ((float)ca + mp_random(ca, cb, cc, s))) * radius;
                    float dy = (py - ((float)cb + mp_random(ca, cb, cc, s + 1))) * radius;
                    float dz = (pz - ((float)cc + mp_random(ca, cb, cc, s + 2))) * radius;
                    float d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 < radius * radius) {
                        float weight = mp_random(ca, cb, cc, s + 3) < 0.5f ? -1.0f : 1.0f;
                        sum = sum + weight * MP_EXP(-MP_PI * a * a * d2) * MP_COS(2.0f * MP_PI * frequency * (dx * ox + dy * oy));
                    }
                }
            }
        }
    }
    return mp_clamp01(0.5f + 0.2f * sum);
}

// Modo's Poisson noise: cells about points scattered at random, level of them in each unit cube, each there as often as
// density says; round (type 0) or angular cells as Modo's Cellular has them, over several frequencies.
MP_HEAVY float mp_poisson_cells(float x, float y, float z, int round, float width, float soft, float variation, float density, int level, int seed) {
    float cx = MP_FLOOR(x);
    float cy = MP_FLOOR(y);
    float cz = MP_FLOOR(z);
    float first = 9.0f;
    float second = 9.0f;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int nk = 0;
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                int a = (int)cx + i;
                int b = (int)cy + j;
                int c = (int)cz + k;
                for (int p = 0; p < 4; p++) {
                    if (p < level && mp_random(a, b, c, seed + 40 + p) < density) {
                        float qx = (float)a + mp_random(a, b, c, seed + 4 * p);
                        float qy = (float)b + mp_random(a, b, c, seed + 4 * p + 1);
                        float qz = (float)c + mp_random(a, b, c, seed + 4 * p + 2);
                        float d = MP_SQRT((x - qx) * (x - qx) + (y - qy) * (y - qy) + (z - qz) * (z - qz));
                        if (d < first) { second = first; first = d; nx = a; ny = b; nz = c; nk = p; }
                        else if (d < second) second = d;
                    }
                }
            }
        }
    }
    if (first > 8.0f) return 0.0f;
    float spacing = 1.0f / MP_POW((float)(level < 1 ? 1 : level), 0.3333333f);
    float inside = round != 0 ? mp_step(0.5f * width * spacing - first, 0.0f, soft * spacing)
                              : mp_step(0.5f * (second - first), 0.5f * (1.0f - width) * spacing, soft * spacing);
    return inside * (1.0f - mp_clamp01(variation) * mp_random(nx, ny, nz, seed + 13 + nk));
}
MP_HEAVY float mp_poisson(int type, float x, float y, float z, MP_Settings s) {
    // level, density, cell width, transition width, frequencies, frequency ratio, amplitude ratio, value variation.
    int level = (int)MP_MAX(1.0f, MP_MIN(4.0f, s.v[0]));
    float width = mp_clamp01(s.v[2]);
    float soft = 0.25f * MP_MAX(0.0f, s.v[3]);
    float total = 0.0f;
    float weight = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    int octaves = (int)MP_MAX(1.0f, MP_MIN(8.0f, s.v[4]));
    for (int i = 0; i < octaves; i++) {
        total = total + amplitude * mp_poisson_cells(x * frequency, y * frequency, z * frequency, type == 0 ? 1 : 0, width, soft, s.v[7],
                                                     mp_clamp01(s.v[1]), level, 7 * i);
        weight = weight + amplitude;
        amplitude = amplitude * MP_MAX(0.0f, s.v[6]);
        frequency = frequency * MP_MAX(1.0f, s.v[5]);
    }
    return weight > 0.0f ? total / weight : 0.0f;
}

// Brightness and contrast, as Modo's Gabor and Poisson noise have them.
MP_FUNC float mp_brighten(float t, float brightness, float contrast) {
    return mp_clamp01((t - 0.5f) * (1.0f + contrast) + 0.5f + brightness);
}

// A colour.
typedef struct { float r; float g; float b; } MP_Rgb;

MP_FUNC MP_Rgb mp_rgb(float r, float g, float b) {
    MP_Rgb c;
    c.r = r;
    c.g = g;
    c.b = b;
    return c;
}

// A colour's hue, 0 to 1 round the wheel from red.
MP_FUNC float mp_hue(float r, float g, float b) {
    float high = MP_MAX(r, MP_MAX(g, b));
    float low = MP_MIN(r, MP_MIN(g, b));
    float span = high - low;
    if (span <= 1e-6f) return 0.0f;
    float h = 0.0f;
    if (high == r) h = (g - b) / span;
    else if (high == g) h = 2.0f + (b - r) / span;
    else h = 4.0f + (r - g) / span;
    return mp_fract(h / 6.0f);
}

// A colour from hue, saturation and value.
MP_FUNC MP_Rgb mp_from_hsv(float h, float s, float v) {
    float k = mp_fract(h) * 6.0f;
    float f = k - MP_FLOOR(k);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float u = v * (1.0f - s * (1.0f - f));
    int sector = (int)MP_FLOOR(k);
    if (sector == 0) return mp_rgb(v, u, p);
    if (sector == 1) return mp_rgb(q, v, p);
    if (sector == 2) return mp_rgb(p, v, u);
    if (sector == 3) return mp_rgb(p, q, v);
    if (sector == 4) return mp_rgb(u, p, v);
    return mp_rgb(v, p, q);
}

// Enhance: Modo's RegionalHSV (400): each region of what is beneath (each colour of it, as a pattern's Regions give
// each cell its own grey) given a colour at random, its hue between those of hue min and hue max, its saturation and
// value between their mins and maxes. Settings: seed, hue min (three), hue max (three), saturation min and max, value
// min and max.
MP_HEAVY MP_Rgb mp_regional_hsv(float r, float g, float b, MP_Settings s) {
    int seed = (int)s.v[0];
    int kr = (int)MP_FLOOR(mp_clamp01(r) * 255.0f + 0.5f);
    int kg = (int)MP_FLOOR(mp_clamp01(g) * 255.0f + 0.5f);
    int kb = (int)MP_FLOOR(mp_clamp01(b) * 255.0f + 0.5f);
    float low = mp_hue(s.v[1], s.v[2], s.v[3]);
    float high = mp_hue(s.v[4], s.v[5], s.v[6]);
    float h = mp_lerp(low, high, mp_random(kr, kg, kb, seed));
    float sat = mp_lerp(s.v[7], s.v[8], mp_random(kr, kg, kb, seed + 1));
    float val = mp_lerp(s.v[9], s.v[10], mp_random(kr, kg, kb, seed + 2));
    return mp_from_hsv(h, mp_clamp01(sat), MP_MAX(0.0f, val));
}

// A process: a colour made from the colour beneath it. RegionalHSV is the one so far.
MP_HEAVY MP_Rgb mp_process(int pattern, float r, float g, float b, MP_Settings s) {
    if (pattern == 400) return mp_regional_hsv(r, g, b, s);
    return mp_rgb(r, g, b);
}

// A pattern at a point of the locator's space, with its bias and gain: how much of the first colour there is.
// Modo's own patterns read s.v[0] to s.v[5] (procedurals.MAP_SETTINGS); Enhance: Modo's (100 on) read theirs.
MP_HEAVY float mp_pattern(int pattern, int type, float x, float y, float z, MP_Settings s) {
    float t = 0.0f;
    if (pattern == MP_NOISE) t = mp_noise_pattern(type, x, y, z, s.v[0], s.v[1], s.v[2]);
    else if (pattern == MP_CELLULAR) t = mp_cellular(type, x, y, z, s.v[0], s.v[1], s.v[2], s.v[3], s.v[4], s.v[5]);
    else if (pattern == MP_CHECKER) t = mp_checker(type, x, y, z, s.v[0]);
    else if (pattern == MP_GRID) t = mp_grid(type, x, y, z, s.v[0], s.v[1]);
    else if (pattern == MP_DOTS) t = mp_dots(type, x, y, z, s.v[0], s.v[1]);
    else if (pattern == MP_WOOD) t = mp_wood(x, y, z, s.v[0], s.v[1], s.v[2], s.v[3], s.v[4]);
    else if (pattern == MP_WEAVE) t = mp_weave(x, y, s.v[0], s.v[1]);
    else if (pattern == MP_RIPPLES) t = mp_ripples(x, y, z, s.v[0], s.v[1], s.v[2]);
    else if (pattern == MP_GABOR) t = mp_brighten(mp_gabor(x, y, z, s.v[0], s.v[1], s.v[2], (int)s.v[3]), s.v[4], s.v[5]);
    else if (pattern == MP_POISSON) t = mp_brighten(mp_poisson(type, x, y, z, s), s.v[8], s.v[9]);
    else if (pattern >= 100 && pattern < 130) t = mp_enhance_noise(pattern, type, x, y, z, s);
    else if (pattern >= 130 && pattern < 160) t = mp_enhance_organic(pattern, type, x, y, z, s);
    else if (pattern >= 160 && pattern < 200) t = mp_enhance_tiles(pattern, type, x, y, z, s);
    else if (pattern >= 200 && pattern < 220) t = mp_enhance_geometric(pattern, type, x, y, z, s);
    else if (pattern >= 220 && pattern < 240) t = mp_enhance_waveform(pattern, x, s);
    else if (pattern >= 240 && pattern < 260) t = mp_enhance_skins(pattern, type, x, y, z, s);
    else if (pattern >= 260 && pattern < 280) t = mp_enhance_space(pattern, type, x, y, z, s);
    else if (pattern >= 280 && pattern < 290) t = mp_enhance_panels(pattern, type, x, y, z, s);
    else if (pattern >= 290 && pattern < 300) t = mp_enhance_water(pattern, type, x, y, z, s);
    else if (pattern >= 300 && pattern < 310) t = mp_enhance_display(pattern, x, y, s);
    return mp_gain(mp_bias(t, s.v[22]), s.v[23]);
}

#endif
