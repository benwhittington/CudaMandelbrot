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
	std::unique_ptr<RunAndRenderMandelbrotDeviceRaii<double>> m_runner;
	std::vector<double> m_charsOut;

    int m_transStart[2] = {0, 0};
    int m_trans[2] = {0, 0};

public:
	Mandelbrot() {
		sAppName = "Example";
	}

public:
	bool OnUserCreate() override {
		constexpr double minX = -2.;
		constexpr double maxX = 1.;
		constexpr double minY = -1;
		constexpr double maxY = 1;

		m_domain.reset(new Domain<double>(minX, maxX, minY, maxY));
		m_screen.reset(new Screen(*m_domain, ScreenWidth(), ScreenHeight()));
		m_runner.reset(new RunAndRenderMandelbrotDeviceRaii<double>(m_screen->NumPixels()));
		m_charsOut = std::vector<double>((m_screen->NumPixels()));

		return true;
	}

	bool OnUserUpdate(float) override {
        if (GetKey(olc::Key::W).bPressed) {
            m_domain->up();
        }
		else if (GetKey(olc::Key::S).bPressed) {
            m_domain->down();
        }
		else if (GetKey(olc::Key::D).bPressed) {
            m_domain->right();
        }
		else if (GetKey(olc::Key::A).bPressed) {
            m_domain->left();
        }
		else if (GetKey(olc::Key::R).bPressed) {
            m_domain->reset();
        }
		else if (GetKey(olc::Key::EQUALS).bPressed) {
            m_domain->zoomIn();
        }
		else if (GetKey(olc::Key::MINUS).bPressed) {
            m_domain->zoomOut();
        }

		(*m_runner)(*m_domain, *m_screen, m_charsOut.data());
		// m_runner->RunMb2(*m_domain, *m_screen, m_charsOut.data());

		for (size_t col = 0; col < ScreenWidth(); col++) {
			for (size_t row = 0; row < ScreenHeight(); row++) {
				const int32_t pixelValue = m_charsOut.data()[indexRowMaj(row, col, m_screen->PixelsX())] * 256;
				Draw(col, row, olc::Pixel(pixelValue, pixelValue, pixelValue));
			}
		}

		return true;
	}
};


int main() {
	Mandelbrot demo;
	if (demo.Construct(100, 50, 4, 4, false))
		demo.Start();

	return 0;
}
