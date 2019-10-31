INC=-I/usr/local/cuda-10.0/include
NVCC=/usr/local/cuda-10.0/bin/nvcc
NVCC_OPT=-std=c++11


gpu_example : main.o CUDAFunc.o CNNFunc.o Config.o Memblock.o utils.o
	#g++ -L/usr/local/cuda-10.0/lib64 -lm -o $@ main.o CUDAFunc.o CNNFunc.o Config.o Memblock.o utils.o
	$(NVCC) -lm -o $@ main.o CUDAFunc.o CNNFunc.o Config.o Memblock.o utils.o
main.o : main.cu Define.h MemBlock.h Config.h utils.h CNNFunc.h CUDAFunc.cuh
	$(NVCC) -c main.cu -o $@

CUDAFunc.o : CUDAFunc.cu CUDAFunc.cuh
	$(NVCC) -c CUDAFunc.cu -o $@ 

CNNFunc.o : CNNFunc.cpp Config.h utils.h
	g++ -lm -o $@ -c CNNFunc.cpp

Config.o : Config.cpp Config.h
	g++ -lm -o $@ -c Config.cpp
	
Memblock.o : MemBlock.cpp MemBlock.h
	g++ -lm -o $@ -c MemBlock.cpp

utils.o : utils.cpp utils.h
	g++ -lm -o $@ -c utils.cpp
clean:
	-rm -f gpu_example *.o
