#pragma once 

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "screen.hpp"
#include "cudaHelpers.cuh"
#include "domain.hpp"

template<typename T>
__host__ __device__ char MapValueToChar(T val) {
    if (val < 0.1) {
        return ' ';
    }
    else if (val < 0.3) {
        return '.';
    }
    else if (val < 0.6) {
        return 'o';
    }
    else if (val < 0.8) {
        return '*';
    }
    else {
        return '#';
    }
}

// maps values to ascii characters
// expects pixelsY == gridDim.x, blockDim.x == pixelsX
template<typename int_T>
__global__ void MapValuesToChars(const int_T* arr, char* charArr) {
    const auto pixelsX = blockDim.x;
    const auto pixelsY = gridDim.x;

    const auto col = threadIdx.x;
    const auto row = blockIdx.x;
    char out;

    if (row == 0 || row == pixelsY - 1) {
        out = '+';
    }
    else if (col == 0 || col == pixelsX - 1) {
        out = '+';
    }
    else {
        const int_T val = arr[indexRowMaj(row, col, pixelsX)];
        out = MapValueToChar(val);
    }

    charArr[indexRowMaj(row, col, pixelsX)] = out;
}

// prints characters in charsOut in grid based on screen dims
void PrintChars(const Screen& screen, const char* charsOut) {
    for (size_t row = 0; row < screen.PixelsY(); ++row) {
        for (size_t col = 0; col < screen.PixelsX(); ++col) {
            const char val = charsOut[indexRowMaj(row, col, screen.PixelsX())];
            std::cout << val;
        }
        std::cout << std::setfill(' ') << std::setw(5);
        std::cout << row <<  std::endl;
    }
}

// allocates device memory once when instantiated, writes rendered ascii symbols to out
// template parm specifies type to use on device
template<typename float_T>
class Mb1ByCols {
private:
    float_T* m_pDevFloatsOut;
    char* m_pDevCharsOut;
    Screen const* m_pScreen;
    dim3 m_blockDim;
    dim3 m_gridDim;

    void PopulateValues(const Domain<float_T>& domain) {
        RunMandelbrot<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(domain, m_pDevFloatsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void PopulateCharacters() {
        MapValuesToChars<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(m_pDevFloatsOut, m_pDevCharsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

public:
    Mb1ByCols(Screen const* screen) : m_pScreen(screen) {
        cuda_malloc(reinterpret_cast<void **>(&m_pDevFloatsOut), sizeof(float_T) * m_pScreen->NumPixels());
        // todo: fix potentially ununused memory
        cuda_malloc(reinterpret_cast<void**>(&m_pDevCharsOut), sizeof(char) * m_pScreen->NumPixels());
    }

    ~Mb1ByCols() {
        cuda_free(m_pDevFloatsOut);
        cuda_free(m_pDevCharsOut);
    }

    void operator()(const Domain<float_T>& domain, char* out) {
        PopulateValues(domain);
        PopulateCharacters();
        cuda_mem_cpy(out, m_pDevCharsOut, sizeof(char) * m_pScreen->NumPixels(), cudaMemcpyDeviceToHost);
    }

    void operator()(const Domain<float_T>& domain, float_T* out) {
        PopulateValues(domain);
        cuda_mem_cpy(out, m_pDevFloatsOut, sizeof(float_T) * m_pScreen->NumPixels(), cudaMemcpyDeviceToHost);
    }
};

template<typename float_T>
class Mb8By8 {
private:
    float_T* m_pDevFloatsOut;
    char* m_pDevCharsOut;
    Screen const* m_pScreen;
    dim3 m_blockDim;
    dim3 m_gridDim;
    size_t m_arraySize;

    void PopulateValues(const Domain<float_T>& domain) {
        RunMandelbrot8By8<<<m_gridDim, m_blockDim>>>(domain, m_pDevFloatsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void PopulateCharacters() {
        MapValuesToChars<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(m_pDevFloatsOut, m_pDevCharsOut);
        cuda_sync();
        cuda_peek_last_error();
    }    

public:
    Mb8By8(Screen const* pScreen) : m_pScreen(pScreen) {
        m_blockDim = dim3(8, 8, 1);
        const size_t blocksX = static_cast<unsigned int>(1 + (m_pScreen->PixelsX() - 1) / 8);
        const size_t blocksY = static_cast<unsigned int>(1 + (m_pScreen->PixelsY() - 1) / 8);
        m_gridDim = dim3(blocksX, blocksY, 1);

        // over allocate here so there's no branching in the kernel
        m_arraySize = blocksX * 8 * blocksY * 8;
        cuda_malloc(reinterpret_cast<void **>(&m_pDevFloatsOut), sizeof(float_T) * m_arraySize);
        // todo: fix potentially unused memory
        cuda_malloc(reinterpret_cast<void**>(&m_pDevCharsOut), sizeof(char) * m_arraySize);
    }

    ~Mb8By8() {
        cuda_free(m_pDevFloatsOut);
        cuda_free(m_pDevCharsOut);
    }

    size_t ArraySize() {
        return m_arraySize;
    }

    void operator()(const Domain<float_T>& domain, char* out) {
        PopulateValues(domain);
        PopulateCharacters();
        cuda_mem_cpy(out, m_pDevCharsOut, sizeof(char) * m_arraySize, cudaMemcpyDeviceToHost);
    }

    void operator()(const Domain<float_T>& domain, float_T* out) {
        PopulateValues(domain);
        cuda_mem_cpy(out, m_pDevFloatsOut, sizeof(float_T) * m_arraySize, cudaMemcpyDeviceToHost);
    }
};
