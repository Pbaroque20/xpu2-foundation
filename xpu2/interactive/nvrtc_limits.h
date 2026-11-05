// <cuda/std/limits> for NanoVDB on the GPU (volume_grid.h): XPU2 compiles with NVRTC and has no libcu++; NanoVDB asks only
// numeric_limits<T>::max (math::Maximum). Handed to NVRTC under that name by gpu_build.compile_ptx.
#pragma once
namespace cuda { namespace std {
template <class T> struct numeric_limits;
template <> struct numeric_limits<float> { __host__ __device__ static constexpr float max() { return 3.402823466e+38f; } };
template <> struct numeric_limits<double> { __host__ __device__ static constexpr double max() { return 1.7976931348623157e+308; } };
template <> struct numeric_limits<int> { __host__ __device__ static constexpr int max() { return 2147483647; } };
template <> struct numeric_limits<unsigned int> { __host__ __device__ static constexpr unsigned int max() { return 4294967295u; } };
template <> struct numeric_limits<long long> { __host__ __device__ static constexpr long long max() { return 9223372036854775807ll; } };
template <> struct numeric_limits<unsigned long long> { __host__ __device__ static constexpr unsigned long long max() { return 18446744073709551615ull; } };
}}
