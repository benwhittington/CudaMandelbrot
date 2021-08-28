#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include <iostream>

#define CUDA_REQUIRE_SUCCESS(expr) { cuda_check((expr),__FILE__,__PRETTY_FUNCTION__,#expr,__LINE__); }
#define cuda_peek_last_error() { CUDA_REQUIRE_SUCCESS(cudaPeekAtLastError()); }
#define cuda_sync(){ CUDA_REQUIRE_SUCCESS(cudaDeviceSynchronize()); }
#define cuda_malloc(devPtr, size) { CUDA_REQUIRE_SUCCESS(cudaMalloc(devPtr, size)); }
#define cuda_mem_cpy(dst, src, count, kind) { CUDA_REQUIRE_SUCCESS(cudaMemcpy(dst, src, count, kind)); }
#define cuda_free(ptr) { CUDA_REQUIRE_SUCCESS(cudaFree(ptr)); }

// shamelessly stolen from David (github.com/DCGroothuizenDijkema/), thank you David.
inline void cuda_check(const cudaError_t code, const char * const file, const char * const func, const char * const call, const int line) {
    if (code != cudaSuccess) {
        std::cerr
        << file << ":" << line << ": CUDA ERROR (" << code << "): " << cudaGetErrorName(code) << ": " << cudaGetErrorString(code) << '\n'
        << "  " << func << "() {\n"
        << "    " << call << '\n'
        << "  }" << std::endl;
        exit(code);
    }
}

template<typename T>
__host__ __device__ T IndexRowMaj(T row, T col, T numCols, T padding = 0) {
    return row * (numCols + padding) + col;
}

struct RowMaj {};
struct ColMaj {};

template<typename Order>
class Indexer;

template<>
class Indexer<RowMaj> {
    size_t m_numCols;
    size_t m_padding;
public:
    Indexer(size_t numCols, size_t padding) : m_numCols(numCols),
                                              m_padding(padding)
    {}

    Indexer(size_t numCols) : m_numCols(numCols),
                              m_padding(0)
    {}

    size_t operator()(size_t row, size_t col) {
        return row * (m_numCols + m_padding) + col;
    }
};

template<>
class Indexer<ColMaj> {
    size_t m_numRows;
    size_t m_padding;
public:
    Indexer(size_t numRows, size_t padding) : m_numRows(numRows),
                                              m_padding(padding)
    {}

    Indexer(size_t numRows) : m_numRows(numRows),
                                              m_padding(0)
    {}

    size_t operator()(size_t row, size_t col) {
        return col * (m_numRows + m_padding) + row;
    }
};
