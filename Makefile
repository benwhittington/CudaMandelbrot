CXX = g++
NVXX = nvcc
LD = ld
RM = rm
MKDIR = mkdir

CXXFLAGS = -std=c++2a -Wall -Wextra -Wpedantic -L/usr/local/cuda/lib64 -c
NVXXFLAGS = --compiler-options -Wall --compiler-options -Wextra --compiler-options -std=c++14
# NVXXFLAGS = -c
LDFLAGS = -lcuda -lcudart -lrt -lX11 -lGL -lpthread -lpng -lstdc++fs 

ALLFLAGS = --compiler-options -Wall --compiler-options -Wextra --compiler-options -std=c++14 -lcuda -lcudart -lrt -lX11 -lGL -lpthread -lpng -lstdc++fs

OBJS = obj/main.o
ASCII = ascii
PGE = pge

all: $(ASCII) mkobjdir

mkobjdir:
	$(MKDIR) -p ./obj

rmobjs:
	$(RM) -r ./**/*.o

rmtarget:
	$(RM) $(ASCII)

target: $(ASCII)

# $(ASCII): obj/ascii.o
# 	$(CXX) -o $@ $^ $(LDFLAGS)

# $(PGE): obj/pge.o
# 	$(CXX) -o $@ $^ $(LDFLAGS)

clean: rmobjs rmtarget

$(ASCII): ./src/main.cu ./src/renderMandelbrot.cuh ./src/domain.hpp ./src/screen.hpp ./src/mandelbrot.cuh ./src/cudaHelpers.cuh
	$(NVXX) -o $@ $< $(NVXXFLAGS) $(LDFLAGS)

$(PGE): ./src/pge.cu ./src/renderMandelbrot.cuh ./src/domain.hpp ./src/screen.hpp ./src/mandelbrot.cuh ./src/cudaHelpers.cuh
	$(NVXX) -o $@ $< $(NVXXFLAGS) $(LDFLAGS)


