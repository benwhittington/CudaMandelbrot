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
__host__ __device__ T indexRowMaj(T row, T col, T numCols) {
    return row * numCols + col;
}
