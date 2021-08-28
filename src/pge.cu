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

template<typename float_T>
class Mandelbrot : public olc::PixelGameEngine {
private:
	std::unique_ptr<Domain<float_T>> m_domain;
	std::unique_ptr<Screen> m_screen;
	std::unique_ptr<Mb8By8<float_T>> m_runner;
	// std::unique_ptr<Mb1ByCols<float_T>> m_runner;
	std::vector<float_T> m_out;
	float_T m_lastMaxValue;
	float_T m_lastMinValue;
	bool m_updateRequired;

public:
	Mandelbrot() {
		sAppName = "Mandelbrot";
	}

	bool OnUserCreate() override {
		m_updateRequired = true;
		m_lastMaxValue = 1;
		m_lastMinValue = 0;
		constexpr float_T minX = -2.;
		constexpr float_T maxX = 1.;
		constexpr float_T minY = -1;
		constexpr float_T maxY = 1;

		m_domain.reset(new Domain<float_T>(minX, maxX, minY, maxY));
		m_screen.reset(new Screen(*m_domain, ScreenWidth(), ScreenHeight()));
		m_runner.reset(new Mb8By8<float_T>(m_screen.get()));
		// m_runner.reset(new Mb1ByCols<float_T>(m_screen.get()));
		m_out = std::vector<float_T>((m_screen->NumPixels()));

		return true;
	}

	void DrawScreen() {
		float_T thisMaxValue = 0;
		float_T thisMinValue = 1;

		for (size_t col = 0; col < ScreenWidth(); col++) {
			for (size_t row = 0; row < ScreenHeight(); row++) {
				const auto rawValue = m_out.data()[indexRowMaj(row, col, m_screen->PixelsX())];
				const auto mappedValue = map(rawValue, m_lastMinValue, m_lastMaxValue, 0., 1.);
				Draw(col, row, olc::Pixel(mappedValue * 200, mappedValue * 100, mappedValue * 100));
				if (rawValue > thisMaxValue) {
					thisMaxValue = rawValue;
				}
				else if (rawValue < thisMinValue) {
					thisMinValue = rawValue;
				}
			}
		}
		std::cout << "------------------------" << std::endl;
		m_lastMaxValue = thisMaxValue;
		m_lastMinValue = thisMinValue;
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

		if (m_updateRequired) {
			(*m_runner)(*m_domain, m_out.data());
			DrawScreen();
			m_updateRequired = false;
		}

		return true;
	}
};

int main(int argc, char *argv[]) {
	int32_t screenWidth = 1024;
	int32_t screenHeight = screenWidth / 1.5;
	int32_t pixelSize = 1;

	switch (argc) {
		case 1:
			break;
		case 2:
			screenWidth = screenHeight = std::atoi(argv[1]);
			break;
		case 4:
			pixelSize = std::atoi(argv[3]);
		case 3:
			screenWidth = std::atoi(argv[1]);
			screenHeight = std::atoi(argv[2]);
			break;
		default:
			std::cout << "Incorrect number of command line arguments" << std::endl;
			exit(EXIT_FAILURE);
	}

	Mandelbrot<float> demo;
	if (demo.Construct(screenWidth, screenHeight, pixelSize, pixelSize, false))
		demo.Start();

	return 0;
}
