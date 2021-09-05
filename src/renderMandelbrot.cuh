#pragma once 

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "colour.hpp"
#include "colourMapping.cuh"
#include "cudaHelpers.cuh"
#include "domain.hpp"
#include "screen.hpp"

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
template<typename float_T>
__global__ void MapValuesToChars(const float_T* arr, char* charArr, size_t paddingX = 0, size_t paddingY = 0) {
    const size_t pixelsX = blockDim.x;
    const size_t pixelsY = gridDim.x;

    const size_t col = threadIdx.x;
    const size_t row = blockIdx.x;
    char out;

    if (row == 0 || row == pixelsY - paddingY - 1) {
        out = '+';
    }
    else if (col == 0 || col == pixelsX - paddingX - 1) {
        out = '+';
    }
    else {
        const float_T val = arr[IndexRowMaj(row, col, pixelsX, paddingX)];
        out = MapValueToChar(val);
    }

    charArr[IndexRowMaj(row, col, pixelsX)] = out;
}

// prints characters in charsOut in grid based on screen dims
void PrintChars(const Screen& screen, const char* charsOut, size_t paddingX = 0) {
    for (size_t row = 0; row < screen.PixelsY(); ++row) {
        for (size_t col = 0; col < screen.PixelsX(); ++col) {
            const char val = charsOut[IndexRowMaj(row, col, screen.PixelsX(), paddingX)];
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
    std::unique_ptr<Indexer<RowMaj>> m_pIndexer;
    std::vector<float_T> m_valuesOut;
    dim3 m_blockDim;
    dim3 m_gridDim;
    size_t m_arraySize;
    size_t m_paddingX;
    size_t m_paddingY;

    void PopulateValues(const Domain<float_T>& domain) {
        RunMandelbrot<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(domain, m_pDevFloatsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void PopulateCharacters() {
        MapValuesToChars<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(m_pDevFloatsOut, m_pDevCharsOut, m_paddingX, m_paddingY);
        cuda_sync();
        cuda_peek_last_error();
    }

public:
    Mb1ByCols(Screen const* screen) : m_pScreen(screen) {
        m_paddingX = 0;
        m_paddingY = 0;
        m_arraySize = m_pScreen->NumPixels();
        m_valuesOut = std::vector<float_T>(m_arraySize);
        m_pIndexer.reset(new Indexer<RowMaj>(m_pScreen->PixelsX(), m_paddingX));
        cuda_malloc(reinterpret_cast<void **>(&m_pDevFloatsOut), sizeof(float_T) * m_arraySize);
        // todo: fix potentially ununused memory
        cuda_malloc(reinterpret_cast<void**>(&m_pDevCharsOut), sizeof(char) * m_arraySize);
    }

    ~Mb1ByCols() {
        cuda_free(m_pDevFloatsOut);
        cuda_free(m_pDevCharsOut);
    }

    size_t PaddingX() {
        return m_paddingX;
    }

    size_t PaddingY() {
        return m_paddingY;
    }

    size_t ArraySize() {
        return m_arraySize;
    }

    float_T* Data() {
        return m_valuesOut.data();
    }

    float_T GetValue(size_t row, size_t col) {
        return m_valuesOut.data()[(*m_pIndexer)(row, col)];
    }

    void Run(const Domain<float_T>& domain, char* out) {
        PopulateValues(domain);
        PopulateCharacters();
        cuda_mem_cpy(out, m_pDevCharsOut, sizeof(char) * m_pScreen->NumPixels(), cudaMemcpyDeviceToHost);
    }

    void Run(const Domain<float_T>& domain) {
        PopulateValues(domain);
        cuda_mem_cpy(m_valuesOut.data(), m_pDevFloatsOut, sizeof(float_T) * m_pScreen->NumPixels(), cudaMemcpyDeviceToHost);
    }
};

template<typename float_T>
class Mb8By8 {
private:
    float_T* m_pDevFloatsOut;
    char* m_pDevCharsOut;
    Colour* m_pDevColoursOut;
    Screen const* m_pScreen;
    std::unique_ptr<Indexer<RowMaj>> m_pIndexer;
    std::vector<float_T> m_valuesOut;
    std::vector<Colour> m_coloursOut;
    dim3 m_blockDim;
    dim3 m_gridDim;
    size_t m_arraySize;
    size_t m_paddingX;
    size_t m_paddingY;

    void PopulateValues(const Domain<float_T>& domain) {
        RunMandelbrot8By8<<<m_gridDim, m_blockDim>>>(domain, m_pDevFloatsOut);
        cuda_sync();
        cuda_peek_last_error();
    }

    void PopulateColours() {
        MapValueToColour8By8<<<m_gridDim, m_blockDim>>>(m_pDevFloatsOut, m_pDevColoursOut);
    }

    // todo see below
    // void PopulateCharacters() {
    //     MapValuesToChars<<<m_pScreen->PixelsY(), m_pScreen->PixelsX()>>>(m_pDevFloatsOut, m_pDevCharsOut, m_paddingX, m_paddingY);
    //     cuda_sync();
    //     cuda_peek_last_error();
    // }    

public:
    Mb8By8(Screen const* pScreen) : m_pScreen(pScreen) {
        m_blockDim = dim3(8, 8, 1);

        const size_t blocksX = static_cast<size_t>(1 + (m_pScreen->PixelsX() - 1) / 8);
        const size_t blocksY = static_cast<size_t>(1 + (m_pScreen->PixelsY() - 1) / 8);
        m_gridDim = dim3(blocksX, blocksY, 1);
        m_paddingX = blocksX * 8 - m_pScreen->PixelsX();
        m_paddingY = blocksY * 8 - m_pScreen->PixelsY();
        m_pIndexer.reset(new Indexer<RowMaj>(m_pScreen->PixelsX(), m_paddingX));
        // over allocate here so there's no branching in the kernel
        m_arraySize = blocksX * 8 * blocksY * 8;
        m_valuesOut = std::vector<float_T>(m_arraySize);
        m_coloursOut = std::vector<Colour>(m_arraySize);
        cuda_malloc(reinterpret_cast<void **>(&m_pDevFloatsOut), sizeof(float_T) * m_arraySize);
        cuda_malloc(reinterpret_cast<void **>(&m_pDevColoursOut), sizeof(Colour) * m_arraySize);

        // todo: fix potentially unused memory
        // cuda_malloc(reinterpret_cast<void**>(&m_pDevCharsOut), sizeof(char) * m_arraySize);
    }

    ~Mb8By8() {
        cuda_free(m_pDevFloatsOut);
        cuda_free(m_pDevCharsOut);
    }

    size_t PaddingX() {
        return m_paddingX;
    }

    size_t PaddingY() {
        return m_paddingY;
    }

    size_t ArraySize() {
        return m_arraySize;
    }

    float_T GetValue(size_t row, size_t col) {
        return m_valuesOut.data()[(*m_pIndexer)(row, col)];
    }

    Colour GetColour(size_t row, size_t col) {
        return m_coloursOut.data()[(*m_pIndexer)(row, col)];
    }

    // todo indexing doesn't work with funky padded device mem
    // void Run(const Domain<float_T>& domain, char* out) {
    //     PopulateValues(domain);
    //     PopulateCharacters();
    //     cuda_mem_cpy(out, m_pDevCharsOut, sizeof(char) * m_arraySize, cudaMemcpyDeviceToHost);
    // }

    void Run(const Domain<float_T>& domain) {
        PopulateValues(domain);
        cuda_mem_cpy(m_valuesOut.data(), m_pDevFloatsOut, sizeof(float_T) * m_arraySize, cudaMemcpyDeviceToHost);
    }

    void RunColours(const Domain<float_T>& domain) {
        PopulateValues(domain);
        PopulateColours();
        cuda_mem_cpy(m_coloursOut.data(), m_pDevColoursOut, sizeof(Colour) * m_arraySize, cudaMemcpyDeviceToHost);
    }

};
