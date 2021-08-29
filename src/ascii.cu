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

void mb(size_t screenWidth, size_t screenHeight) {
    constexpr double minX = -2.;
    constexpr double maxX = 1.;
    constexpr double minY = -1;
    constexpr double maxY = 1;

    auto domain = Domain<double>(minX, maxX, minY, maxY);
    const auto screen = Screen(screenWidth, screenHeight);

    auto runner = std::unique_ptr<Mb1ByCols<double>>(new Mb1ByCols<double>(&screen));
    auto paddingX = runner->PaddingX();
    auto paddingY = runner->PaddingY();

    std::vector<char> out(runner->ArraySize());

    while (true) {
        // std::cout << "\033c" << std::endl;

        runner->Run(domain, out.data());
        PrintChars(screen, out.data(), paddingX);

        auto val = std::cin.get();
        switch (val) {
            case 61: // =
                domain.ZoomIn();
                break;
            case 45: // -
                domain.ZoomOut();
                break;
            case 119: // w
                domain.Up();
                break;
            case 115: // s
                domain.Down();
                break;
            case 100: // d
                domain.Right();
                break;
            case 97: // a
                domain.Left();
                break;
            case 104: // h
                domain.Reset();
        }
    }
}

void tests() {
    ASSERT(map(0, 0, 1, -1, 1) == -1);
    ASSERT(map(0.5, 0., 1., -1, 1) == 0.0);
    ASSERT(map(0.5, 0., 1., -1, 1) == 0.0);
    ASSERT(map(0, 0, 4, -1, 1) == -1);

    ASSERT(IndexRowMaj(0, 0, 10) == 0);
    ASSERT(IndexRowMaj(1, 0, 10) == 10);
    ASSERT(IndexRowMaj(1, 6, 10) == 16);
    ASSERT(IndexRowMaj(2, 9, 10) == 29);
}

int main(int argc, char *argv[]) {
	size_t screenWidth = 1024;
	size_t screenHeight = screenWidth / 1.5;

	switch (argc) {
		case 1:
			break;
		case 2:
			screenWidth = std::atoi(argv[1]);
            screenHeight =  screenWidth / 1.5;
			break;
		case 3:
			screenWidth = std::atoi(argv[1]);
			screenHeight = std::atoi(argv[2]);
			break;
		default:
			std::cout << "Incorrect number of command line arguments" << std::endl;
			exit(EXIT_FAILURE);
	}

    mb(screenWidth, screenHeight);

	return 0;
}