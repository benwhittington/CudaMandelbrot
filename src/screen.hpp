#pragma once

#include <stddef.h>

#include "domain.hpp"

struct Screen {
    template<typename T>
    Screen(Domain<T> domain, size_t pixelsPerUnit) {
        m_pixelsPerX = m_pixelsPerY = pixelsPerUnit;
        m_pixelsX = static_cast<size_t>(static_cast<T>(m_pixelsPerX) * (domain.MaxX() - domain.MinX()));
        m_pixelsY = static_cast<size_t>(static_cast<T>(m_pixelsPerY) * (domain.MaxY() - domain.MinY()));
        m_numPixels = m_pixelsX * m_pixelsY;
    }

    template<typename T>
    Screen(Domain<T> domain, size_t pixelsX, size_t pixelsY) : m_pixelsX(pixelsX), 
                                                               m_pixelsY(pixelsY) {
        m_pixelsPerX = static_cast<size_t>(m_pixelsX / (domain.MaxX() - domain.MinX()));
        m_pixelsPerY = static_cast<size_t>(m_pixelsY / (domain.MaxY() - domain.MinY()));
        m_numPixels = m_pixelsX * m_pixelsY;
    }

    size_t PixelsPerX() const {
        return m_pixelsPerX;
    }

    size_t PixelsPerY() const {
        return m_pixelsPerY;
    }

    size_t PixelsX() const {
        return m_pixelsX;
    }

    size_t PixelsY() const {
        return m_pixelsY;
    }

    size_t NumPixels() const {
        return m_numPixels;
    }

private:
    size_t m_pixelsPerX;
    size_t m_pixelsPerY;
    size_t m_pixelsX;
    size_t m_pixelsY;
    size_t m_numPixels;
};