#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <cstdio>

#include "domain.hpp"
#include "cudaHelpers.cuh"

static constexpr double sThreshold = 10.;
static constexpr size_t sMaxIterations = 100;

// maps value from one range to another
template <typename A, typename B, typename C>
__host__ __device__ C map(A x, B x1, B x2, C y1, C y2) {
    return (x - x1) * (y2 - y1) / (x2 - x1) + y1;
}

// performs mandelbrot iterations on complex coord
template <typename float_T>
__host__ __device__ float_T PerformMandelbrotIterations(float_T x, float_T y) {
    const auto cRe = x;
    const auto cIm = y;

    float_T zRe = 0.;
    float_T zIm = 0.;

    static constexpr float_T threshold = static_cast<float_T>(sThreshold);
    static constexpr float_T maxIterations = static_cast<size_t>(sMaxIterations);

    for (size_t i = 0; i < maxIterations; ++i) {
        float_T zReTemp = zRe * zRe - zIm * zIm + cRe;
        zIm = 2 * zRe * zIm + cIm;
        zRe = zReTemp;
        if (zRe * zRe + zIm * zIm > threshold) {
            return static_cast<float_T>(i) / static_cast<float_T>(maxIterations);
        }
    }
    return static_cast<float_T>(1.);
}

// performs mandelbrot iterations on every point in domain
// expects pixelsX == blockDim.x, pixelsY == gridDim.x
template<typename float_T1, typename float_T2>
__global__ void RunMandelbrot(Domain<float_T1> domain, float_T2* out) {
    const auto pixelsX = blockDim.x;
    const auto pixelsY = gridDim.x;
    const auto col = threadIdx.x;
    const auto row = blockIdx.x;

    const auto x = map(col, 0u, pixelsX, domain.MinX(), domain.MaxX());
    const auto y = map(row, 0u, pixelsY, domain.MinY(), domain.MaxY());
    
    const auto idx = IndexRowMaj(row, col, pixelsX);
    const auto val = PerformMandelbrotIterations(x, y);

    out[idx] = val;
}

// performs mandelbrot iterations on every point in domain
// expects:
//      blockDim.x == blockDim.y == 8
//      pixelsX == gridDim.x * 8
//      pixelsY == gridDim.y * 8
// NOTE: Does not check for out of bounds when indexing out,
// out MUST have dimensions at least as big as each axis of
// your input pixels rounded up to the nearest multiple of 8
template<typename float_T1, typename float_T2>
__global__ void RunMandelbrot8By8(Domain<float_T1> domain, float_T2* out) {
    const auto pixelsX = gridDim.x * blockDim.x;
    const auto pixelsY = gridDim.y * blockDim.y;

    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("%u | ", row);
    // printf("%u | ", col);
    // printf("%u %u\n", blockIdx.x, blockIdx.y);
    // printf("%2u %2u\n", row, col);
    const auto x = map(col, 0u, pixelsX, domain.MinX(), domain.MaxX());
    const auto y = map(row, 0u, pixelsY, domain.MinY(), domain.MaxY());
    
    const auto idx = IndexRowMaj(row, col, pixelsX);
    const auto val = PerformMandelbrotIterations(x, y);
    // printf("%s\n", __PRETTY_FUNCTION__);
    out[idx] = val;
}
