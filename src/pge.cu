#define OLC_PGE_APPLICATION

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <olcPixelGameEngine.h>
#include <cstdio>

#include "cudaHelpers.cuh"
#include "domain.hpp"
#include "mandelbrot.cuh"
#include "renderMandelbrot.cuh"
#include "screen.hpp"

template<typename float_T, typename Runner_T>
class Mandelbrot : public olc::PixelGameEngine {
private:
	std::unique_ptr<Domain<float_T>> m_pDomain;
	std::unique_ptr<Screen> m_pScreen;
	std::unique_ptr<Runner_T> m_pRunner;

	float_T m_lastMaxValue;
	float_T m_lastMinValue;

	bool m_updateRequired;

public:
	Mandelbrot() {
		sAppName = "Mandelbrot";
	}

	bool OnUserCreate() override {
		m_lastMaxValue = 1;
		m_lastMinValue = 0;
		m_updateRequired = true;
		
		constexpr float_T minX = -2.;
		constexpr float_T maxX = 1.;
		constexpr float_T minY = -1;
		constexpr float_T maxY = 1;

		m_pDomain.reset(new Domain<float_T>(minX, maxX, minY, maxY));
		m_pScreen.reset(new Screen(ScreenWidth(), ScreenHeight()));
		m_pRunner.reset(new Runner_T(m_pScreen.get()));

		return true;
	}

	void DrawScreen() {
		float_T thisMaxValue = 0;
		float_T thisMinValue = 1;

		for (size_t col = 0; col < m_pScreen->PixelsX(); col++) {
			for (size_t row = 0; row < m_pScreen->PixelsY(); row++) {
				const auto rawValue = m_pRunner->GetValue(row, col);
				const auto mappedValue = map(rawValue, m_lastMinValue, m_lastMaxValue, 0., 1.);
				if (rawValue > thisMaxValue) {
					thisMaxValue = rawValue;
				}
				else if (rawValue < thisMinValue) {
					thisMinValue = rawValue;
				}
				Draw(col, row, olc::Pixel(mappedValue * 200, mappedValue * 100, mappedValue * 100));
			}
		}

		m_lastMaxValue = thisMaxValue;
		m_lastMinValue = thisMinValue;
	}

	void GetUserInput() {
		using K = olc::Key;
		if (GetKey(K::W).bPressed) {
            m_pDomain->Up();
			m_updateRequired = true;
        }
		else if (GetKey(K::S).bPressed) {
            m_pDomain->Down();
			m_updateRequired = true;

        }
		else if (GetKey(K::D).bPressed) {
            m_pDomain->Right();
			m_updateRequired = true;

        }
		else if (GetKey(K::A).bPressed) {
            m_pDomain->Left();
			m_updateRequired = true;

        }
		else if (GetKey(K::R).bPressed) {
            m_pDomain->Reset();
			m_updateRequired = true;

        }
		else if (GetKey(K::EQUALS).bHeld) {
            m_pDomain->ZoomIn();
			m_updateRequired = true;

        }
		else if (GetKey(K::MINUS).bHeld) {
            m_pDomain->ZoomOut();
			m_updateRequired = true;
        }
	}

	bool OnUserUpdate(float) override {
		GetUserInput();

		if (m_updateRequired) {
			m_pRunner->Run(*m_pDomain);
			DrawScreen();
			m_updateRequired = false;
		}

		return true;
	}
};

int main(int argc, char *argv[]) {
	int32_t screenWidth = 1025;
	int32_t screenHeight = 688;
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

	typedef double float_T;

	Mandelbrot<float_T, Mb8By8<float_T>> demo;
	// Mandelbrot<float_T, Mb1ByCols<float_T>> demo;
	if (demo.Construct(screenWidth, screenHeight, pixelSize, pixelSize, true)) {
		demo.Start();
	}

	return 0;
}
