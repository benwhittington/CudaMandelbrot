#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "cudaHelpers.cuh"
#include "domain.hpp"
#include "mandelbrot.cuh"
#include "renderMandelbrot.cuh"
#include "screen.hpp"

#define FAIL "[ FAIL ]"
#define PASS "[ PASS ]"
#define ASSERT(expr)                                                                                       \
    {                                                                                                      \
        (expr) ? (std::cout << PASS << std::endl)                                                          \
               : (std::cout << FAIL << " " << __FILE__ << "(" << __LINE__ << "): " << #expr << std::endl); \
    }

void mb() {
    constexpr double minX = -2.;
    constexpr double maxX = 1.;
    constexpr double minY = -1;
    constexpr double maxY = 1;

    // constexpr size_t density = 5;
    // constexpr size_t pixelsPerX = 7 * density;
    // constexpr size_t pixelsPerY = 3 * density;

    auto domain = Domain<double>(minX, maxX, minY, maxY);
    const auto screen = Screen(domain, 100, 50);

    auto runner = RunAndRenderMandelbrotDeviceRaii<double>(screen.NumPixels());

    std::vector<char> out(screen.NumPixels());
    // std::vector<double> charsOut(screen.NumPixels());

    while (true) {
        // std::cout << "\033c" << std::endl;

        // RunMandelbrotDevice(domain, screen, out.data());
        // RenderMandelbrot(screen, out);

        // RunAndRenderMandelbrotDevice(domain, screen, out.data());
        // PrintChars(screen, charsOut);

        runner(domain, screen, out.data());
        PrintChars(screen, out.data());

        // runner.RunMb2(domain, screen, charsOut.data());
        // exit(EXIT_SUCCESS);

        auto val = std::cin.get();
        switch (val) {
        case 61: // =
            domain.zoomIn();
            break;
        case 45: // -
            domain.zoomOut();
            break;
        case 119: // w
            domain.up();
            break;
        case 115: // s
            domain.down();
            break;
        case 100: // d
            domain.right();
            break;
        case 97: // a
            domain.left();
            break;
        case 104: // h
            domain.reset();
        }
    }
}

void tests() {
    ASSERT(map(0, 0, 1, -1, 1) == -1);
    ASSERT(map(0.5, 0., 1., -1, 1) == 0.0);
    ASSERT(map(0.5, 0., 1., -1, 1) == 0.0);
    ASSERT(map(0, 0, 4, -1, 1) == -1);

    ASSERT(indexRowMaj(0, 0, 10) == 0);
    ASSERT(indexRowMaj(1, 0, 10) == 10);
    ASSERT(indexRowMaj(1, 6, 10) == 16);
    ASSERT(indexRowMaj(2, 9, 10) == 29);
}

int main() {
    tests();
    mb();
    return 0;
}