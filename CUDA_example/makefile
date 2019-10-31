INC=-I/usr/local/cuda-10.0/include
NVCC=/usr/local/cuda-10.0/bin/nvcc
NVCC_OPT=-std=c++11

all:
	$(NVCC) $(NVCC_OPT) example.cu -o gpu-example

clean:
	-rm -f gpu-example
