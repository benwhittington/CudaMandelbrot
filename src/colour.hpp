#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

// inspiration from tiny colour map
struct Colour {
private:
    unsigned char m_r;
    unsigned char m_g;
    unsigned char m_b;
public:
    __host__ __device__ Colour(unsigned char r, unsigned char g, unsigned char b) : m_r(r),
                                         m_g(g),
                                         m_b(b)
    {}

    __host__ __device__ Colour() {
        m_r = 0;
        m_g = 0;
        m_b = 0;
    }

    template<typename T>
    __host__ __device__ Colour operator*(T s) {
        return { s * R(), s * G(), s * B() };
    }

    __host__ __device__ unsigned char& R() {
        return m_r;
    }

    __host__ __device__ unsigned char& G() {
        return m_g;
    }

    __host__ __device__ unsigned char& B() {
        return m_b;
    }

    __host__ __device__ unsigned char R() const {
        return m_r;
    }

    __host__ __device__ unsigned char G() const {
        return m_g;
    }

    __host__ __device__ unsigned char B() const {
        return m_b;
    }
};

// __host__ __device__ struct Colour;