#pragma once

#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

template<typename T>
struct Domain {
    Domain(T minX, T maxX, T minY, T maxY) : m_minX(minX),
                                             m_maxX(maxX),
                                             m_minY(minY),
                                             m_maxY(maxY),
                                             m_initMinX(minX),
                                             m_initMaxX(maxX),
                                             m_initMinY(minY),
                                             m_initMaxY(maxY)
    {}

    __host__ __device__ T MinX() const {
        return m_minX;
    }

    __host__ __device__ T MaxX() const {
        return m_maxX;
    }

    __host__ __device__ T MinY() const {
        return m_minY;
    }

    __host__ __device__ T MaxY() const {
        return m_maxY;
    }


    void ZoomIn() {
        zoomIn(m_minX, m_maxX);
        zoomIn(m_minY, m_maxY);
    }

    void ZoomOut() {
        zoomOut(m_minX, m_maxX);
        zoomOut(m_minY, m_maxY);
    }

    void Up() { // shift the content not the view
        neg(m_minY, m_maxY);
    }

    void Down() { // shift the content not the view
        pos(m_minY, m_maxY);
    }

    void Left() {
        neg(m_minX, m_maxX);
    }

    void Right() {
        pos(m_minX, m_maxX);
    }

    void Reset() {
        m_minX = m_initMinX;
        m_maxX = m_initMaxX;
        m_minY = m_initMinY;
        m_maxY = m_initMaxY;  
    }

private:
    T m_minX;
    T m_maxX;
    T m_minY;
    T m_maxY;

    T m_initMinX;
    T m_initMaxX;
    T m_initMinY;
    T m_initMaxY;

    static constexpr T m_sFactor = 1.1;
    static constexpr T m_sShift = 0.1;

    static void pos(T& x0, T& x1){
        const auto absShift = m_sShift * (x1 - x0);
        x0 += absShift;
        x1 += absShift;
    }

    static void neg(T& x0, T& x1){
        const auto absShift = m_sShift * (x1 - x0);
        x0 -= absShift;
        x1 -= absShift;
    }

    static void zoomOut(T& x0, T& x1) {
        const T middle = x0 + (x1 - x0) / 2;
        const T halfWidth = m_sFactor * (x1 - x0) / 2;
        
        x0 = middle - halfWidth; 
        x1 = middle + halfWidth;
    }

    static void zoomIn(T& x0, T& x1) {
        const T middle = x0 + (x1 - x0) / 2;
        const T halfWidth = (x1 - x0) / (m_sFactor * 2);
        
        x0 = middle - halfWidth; 
        x1 = middle + halfWidth;
    }
};
