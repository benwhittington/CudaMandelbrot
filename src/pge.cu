#define OLC_PGE_APPLICATION

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <olcPixelGameEngine.h>

#include "cudaHelpers.cuh"
#include "domain.hpp"
#include "mandelbrot.cuh"
#include "renderMandelbrot.cuh"
#include "screen.hpp"

class Mandelbrot : public olc::PixelGameEngine {
private:
	std::unique_ptr<Domain<double>> m_domain;
	std::unique_ptr<Screen> m_screen;
	std::unique_ptr<Mb1ByCols<double>> m_runner;
	std::vector<double> m_out;
	bool m_updateRequired;

public:
	Mandelbrot() {
		sAppName = "Mandelbrot";
		m_updateRequired = true;
	}

	bool OnUserCreate() override {
		constexpr double minX = -2.;
		constexpr double maxX = 1.;
		constexpr double minY = -1;
		constexpr double maxY = 1;

		m_domain.reset(new Domain<double>(minX, maxX, minY, maxY));
		m_screen.reset(new Screen(*m_domain, ScreenWidth(), ScreenHeight()));
		m_runner.reset(new Mb1ByCols<double>(m_screen->NumPixels()));
		// m_runner.reset(new Mb1ByCols<double>(m_screen->NumPixels()));
		m_out = std::vector<double>((m_screen->NumPixels()));

		return true;
	}

	bool OnUserUpdate(float) override {
		using K = olc::Key;
        if (GetKey(K::W).bPressed) {
            m_domain->up();
			m_updateRequired = true;
        }
		else if (GetKey(K::S).bPressed) {
            m_domain->down();
			m_updateRequired = true;

        }
		else if (GetKey(K::D).bPressed) {
            m_domain->right();
			m_updateRequired = true;

        }
		else if (GetKey(K::A).bPressed) {
            m_domain->left();
			m_updateRequired = true;

        }
		else if (GetKey(K::R).bPressed) {
            m_domain->reset();
			m_updateRequired = true;

        }
		else if (GetKey(K::EQUALS).bHeld) {
            m_domain->zoomIn();
			m_updateRequired = true;

        }
		else if (GetKey(K::MINUS).bHeld) {
            m_domain->zoomOut();
			m_updateRequired = true;
        }

		if (!m_updateRequired) {
			return false;
		}

		(*m_runner)(*m_domain, *m_screen, m_out.data());

		for (size_t col = 0; col < ScreenWidth(); col++) {
			for (size_t row = 0; row < ScreenHeight(); row++) {
				const int32_t pixelValue = m_out.data()[indexRowMaj(row, col, m_screen->PixelsX())] * 256;
				Draw(col, row, olc::Pixel(pixelValue, pixelValue, pixelValue));
			}
		}

		return true;
	}
};

int main(int argc, char *argv[]) {
	int32_t screenWidth = 1024;
	int32_t screenHeight = screenWidth / 1.5;

	switch (argc) {
		case 1:
			break;
		case 2:
			screenWidth = screenHeight = std::atoi(argv[1]);
			break;
		case 3:
			screenWidth = std::atoi(argv[1]);
			screenHeight = std::atoi(argv[2]);
			break;
		default:
			std::cout << "Incorrect number of command line arguments" << std::endl;
			exit(EXIT_FAILURE);
	}

	std::cout << screenWidth << ", " << screenHeight << std::endl;

	Mandelbrot demo;
	if (demo.Construct(screenWidth, screenHeight, 1, 1, false))
		demo.Start();

	return 0;
}
