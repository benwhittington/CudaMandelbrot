#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "colour.hpp"

Colour ultraFractal[] = 
{
    {  0,   7, 100},
    { 32, 107, 203},
    {237, 255, 255},
    {255, 170,   0},
    {  0,   2,   0}
};

template<typename float_T>
__global__ void MapValueToColour8By8(const float_T* values, Colour* data) {
    const auto pixelsX = gridDim.x * blockDim.x;
    // const auto pixelsY = gridDim.y * blockDim.y;

    const auto col = blockIdx.x * blockDim.x + threadIdx.x;
    const auto row = blockIdx.y * blockDim.y + threadIdx.y;

    const auto idx = IndexRowMaj(row, col, pixelsX);

    data[idx] = Interp(values[idx], ultraFractal);
}

template <size_t size, typename float_T>
__device__ Colour Interp(float_T x, const Colour (&data)[size]) {
    // const float_T xScaled  = x * (size - 1);
    // const float_T midIdx  = std::floor(xScaled);
    // const float_T t  = xScaled - midIdx;
    // Colour c0 = data[static_cast<size_t>(xScaled)];
    // Colour c1 = data[static_cast<size_t>(std::ceil(xScaled))];
    // constexpr float_T one = static_cast<float_T>(1.0);

    // return (one - t) * c0 + t * c1;
    return Colour();
}
