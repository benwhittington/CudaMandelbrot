#pragma once 

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <iomanip>
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

// maps values and prints grid
template<typename int_T>
void RenderMandelbrot(const Screen& screen, const std::vector<int_T>& arr) {
    for (size_t row = 0; row < screen.m_pixelsY; ++row) {
        for (size_t col = 0; col < screen.m_pixelsX; ++col) {
            const int_T val = arr[indexRowMaj(row, col, screen.m_pixelsX)];
            auto out = MapValueToChar(val);

            if (row == 0 || row == screen.m_pixelsY - 1) {
                out = '+';
            } 
            else if (col == 0 || col == screen.m_pixelsX - 1) {
                out = '+';
            }

            std::cout << out;
        }
        std::cout << std::setfill(' ') << std::setw(5);
        std::cout << row <<  std::endl;
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
    for (size_t row = 0; row < screen.m_pixelsY; ++row) {
        for (size_t col = 0; col < screen.m_pixelsX; ++col) {
            const char val = charsOut[indexRowMaj(row, col, screen.m_pixelsX)];
            std::cout << val;
        }
        std::cout << std::setfill(' ') << std::setw(5);
        std::cout << row <<  std::endl;
    }
}

// writes raw values to out
template <typename float_T1, typename float_T2>
void RunMandelbrotDevice(const Domain<float_T1>& domain, const Screen& screen, float_T2* out) {
    float_T2* devOut;
    cuda_malloc(reinterpret_cast<void**>(&devOut), sizeof(float_T2) * screen.m_numPixels);

    RunMandelbrot<<<screen.m_pixelsY, screen.m_pixelsX>>>(domain, devOut);

    cuda_sync();
    cuda_peek_last_error();

    cuda_mem_cpy(out, devOut, sizeof(float_T2) * screen.m_numPixels, cudaMemcpyDeviceToHost);

    cuda_free(devOut);
}

// allocates device memory once when instantiated, writes rendered ascii symbols to out
// template parm specifies type to use on device
template<typename float_T1>
class RunAndRenderMandelbrotDeviceRaii {
private:
    float_T1* m_devOut;
    char* m_devCharsOut;

public:
    RunAndRenderMandelbrotDeviceRaii(size_t numPixels) {
        cuda_malloc(reinterpret_cast<void **>(&m_devOut), sizeof(float_T1) * numPixels);
        cuda_malloc(reinterpret_cast<void**>(&m_devCharsOut), sizeof(char) * numPixels);
    }

    ~RunAndRenderMandelbrotDeviceRaii() {
        cuda_free(m_devOut);
        cuda_free(m_devCharsOut);
    }

    template<typename float_T2>
    void operator()(const Domain<float_T2>& domain, const Screen& screen, char* out) {
        RunMandelbrot<<<screen.m_pixelsY, screen.m_pixelsX>>>(domain, m_devOut);

        cuda_sync();
        cuda_peek_last_error();

        MapValuesToChars<<<screen.m_pixelsY, screen.m_pixelsX>>>(m_devOut, m_devCharsOut);

        cuda_sync();
        cuda_peek_last_error();

        cuda_mem_cpy(out, m_devCharsOut, sizeof(char) * screen.m_numPixels, cudaMemcpyDeviceToHost);
    }
};

// allocates/frees device memory on every call, writes ascii characters to out
template <typename float_T>
void RunAndRenderMandelbrotDevice(const Domain<float_T>& domain, const Screen& screen, char* out) {
    unsigned int* devOut;
    char* devCharsOut;

    cuda_malloc(reinterpret_cast<void**>(&devOut), sizeof(unsigned int) * screen.m_numPixels);
    cuda_malloc(reinterpret_cast<void**>(&devCharsOut), sizeof(char) * screen.m_numPixels);

    RunMandelbrot<<<screen.m_pixelsY, screen.m_pixelsX>>>(domain, devOut);

    cuda_sync();
    cuda_peek_last_error();

    MapValuesToChars<<<screen.m_pixelsY, screen.m_pixelsX>>>(devOut, devCharsOut);

    cuda_sync();
    cuda_peek_last_error();

    cuda_mem_cpy(out, devCharsOut, sizeof(char) * screen.m_numPixels, cudaMemcpyDeviceToHost);

    cuda_free(devOut);
    cuda_free(devCharsOut);
}