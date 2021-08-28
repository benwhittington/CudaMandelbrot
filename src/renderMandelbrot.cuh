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

// maps values and prints grid
template<typename int_T>
void RenderMandelbrot(const Screen& screen, const std::vector<int_T>& arr) {
    for (size_t row = 0; row < screen.PixelsY(); ++row) {
        for (size_t col = 0; col < screen.PixelsX(); ++col) {
            const int_T val = arr[indexRowMaj(row, col, screen.PixelsX())];
            auto out = MapValueToChar(val);

            if (row == 0 || row == screen.PixelsY() - 1) {
                out = '+';
            } 
            else if (col == 0 || col == screen.PixelsX() - 1) {
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
template<typename float_T1>
class Mb1ByCols {
private:
    float_T1* m_devOut;
    char* m_devCharsOut;
    Screen const* m_screen;
    dim3 m_blockDim;
    dim3 m_gridDim;

    void GetValues(const Domain<float_T1>& domain) {
        RunMandelbrot<<<m_screen->PixelsY(), m_screen->PixelsX()>>>(domain, m_devOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void GetChars() {
        MapValuesToChars<<<m_screen->PixelsY(), m_screen->PixelsX()>>>(m_devOut, m_devCharsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

public:
    Mb1ByCols(Screen const* screen) : m_screen(screen) {
        cuda_malloc(reinterpret_cast<void **>(&m_devOut), sizeof(float_T1) * m_screen->NumPixels());
        // todo: fix potentially unused memory
        cuda_malloc(reinterpret_cast<void**>(&m_devCharsOut), sizeof(char) * m_screen->NumPixels());
    }

    ~Mb1ByCols() {
        cuda_free(m_devOut);
        cuda_free(m_devCharsOut);
    }

    void operator()(const Domain<float_T1>& domain, char* out, float_T1 scaleFactor) {
        GetValues(domain);
        GetChars();
        cuda_mem_cpy(out, m_devCharsOut, sizeof(char) * m_screen->NumPixels(), cudaMemcpyDeviceToHost);
    }

    void operator()(const Domain<float_T1>& domain, float_T1* out, float_T1 scaleFactor) {
        GetValues(domain);
        cuda_mem_cpy(out, m_devOut, sizeof(float_T1) * m_screen->NumPixels(), cudaMemcpyDeviceToHost);
    }
};

template<typename float_T1>
class Mb8By8 {
private:
    float_T1* m_devOut;
    char* m_devCharsOut;
    Screen const* m_screen;
    dim3 m_blockDim;
    dim3 m_gridDim;

    void GetValues(const Domain<float_T1>& domain) {
        RunMandelbrot8By8<<<m_gridDim, m_blockDim>>>(domain, m_devOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void GetChars() {
        MapValuesToChars<<<m_screen->PixelsY(), m_screen->PixelsX()>>>(m_devOut, m_devCharsOut);
        cuda_sync();
        cuda_peek_last_error();
    }    

public:
    Mb8By8(Screen const* screen) : m_screen(screen) {
        m_blockDim = dim3(8, 8, 1);
        const size_t blocksX = static_cast<unsigned int>(1 + (m_screen->PixelsX() - 1) / 8);
        const size_t blocksY = static_cast<unsigned int>(1 + (m_screen->PixelsY() - 1) / 8);
        m_gridDim = dim3(blocksX, blocksY, 1);

        // over allocate here so there's no branching in the kernel
        const size_t arraySize = blocksX * 8 * blocksY * 8;
        cuda_malloc(reinterpret_cast<void **>(&m_devOut), sizeof(float_T1) * arraySize);
        // todo: fix potentially used memory
        cuda_malloc(reinterpret_cast<void**>(&m_devCharsOut), sizeof(char) * arraySize);

        std::cout 
            << "Screen is " << m_screen->PixelsX() << " by " << m_screen->PixelsY() << "\n"
            << "Sucessfully allocated " << blocksX * 8 << " by " << blocksY * 8 << "\n"
            << std::endl;
    }

    ~Mb8By8() {
        cuda_free(m_devOut);
        cuda_free(m_devCharsOut);
    }

    void operator()(const Domain<float_T1>& domain, char* out) {
        GetValues(domain);
        GetChars();

        cuda_mem_cpy(out, m_devCharsOut, sizeof(char) * m_screen->NumPixels(), cudaMemcpyDeviceToHost);
    }

    void operator()(const Domain<float_T1>& domain, float_T1* out) {
        GetValues(domain);

        cuda_mem_cpy(out, m_devOut, sizeof(char) * m_screen->NumPixels(), cudaMemcpyDeviceToHost);
    }
};
