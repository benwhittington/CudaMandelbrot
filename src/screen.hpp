#pragma once

#include <stddef.h>

#include "domain.hpp"

struct Screen {
    template<typename T>
    Screen(Domain<T> domain, size_t pixelsPerUnit) {
        m_pixelsPerX = m_pixelsPerY = pixelsPerUnit;
        m_pixelsX = static_cast<size_t>(static_cast<T>(m_pixelsPerX) * (domain.m_maxX - domain.m_minX));
        m_pixelsY = static_cast<size_t>(static_cast<T>(m_pixelsPerY) * (domain.m_maxY - domain.m_minY));
        m_numPixels = m_pixelsX * m_pixelsY;
    }

    template<typename T>
    Screen(Domain<T> domain, size_t pixelsPerX, size_t pixelsPerY) : m_pixelsPerX(pixelsPerX), 
                                                                     m_pixelsPerY(pixelsPerY) {
        m_pixelsX = static_cast<size_t>(static_cast<T>(m_pixelsPerX) * (domain.m_maxX - domain.m_minX));
        m_pixelsY = static_cast<size_t>(static_cast<T>(m_pixelsPerY) * (domain.m_maxY - domain.m_minY));
        m_numPixels = m_pixelsX * m_pixelsY;
    }

    size_t m_pixelsPerX;
    size_t m_pixelsPerY;
    size_t m_pixelsX;
    size_t m_pixelsY;
    size_t m_numPixels;
};