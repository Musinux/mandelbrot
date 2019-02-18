default:
	nvcc --compiler-options='-O3' -lm -lpthread threads.cu colors.cu sdl.cu times.cu main.cu `sdl2-config --cflags --libs`
