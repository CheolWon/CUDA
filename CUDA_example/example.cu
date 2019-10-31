#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#define ITER 65536

// CPU version of the vector add function
void vector_add_cpu(int *a, int *b, int *c, int n) {
    int i;

    // Add the vector elements a and b to the vector c
    for (i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
    }
}

// GPU version of the vector add function
__global__ void vector_add_gpu(int *gpu_a, int *gpu_b, int *gpu_c, int n) {
    //int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    // No for loop needed because the CUDA runtime
    // will thread this ITER times
    gpu_c[idx] = gpu_a[idx] * gpu_b[idx];
}

int main() {

    int *a, *b, *c, *gpu_r;
    int *gpu_a, *gpu_b, *gpu_c;

    a = (int *)malloc(ITER * sizeof(int));
    b = (int *)malloc(ITER * sizeof(int));
    c = (int *)malloc(ITER * sizeof(int));
    gpu_r = (int *)malloc(ITER * sizeof(int)); 

    // We need variables accessible to the GPU,
    // so cudaMallocManaged provides these
    cudaMalloc((void**)&gpu_a, ITER * sizeof(int));
    cudaMalloc((void**)&gpu_b, ITER * sizeof(int));
    cudaMalloc((void**)&gpu_c, ITER * sizeof(int));

    for (int i = 0; i < ITER; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = i;
	gpu_r[i] = i;
    }

    // Call the CPU function and time it
    auto cpu_start = Clock::now();
    vector_add_cpu(a, b, c, ITER);
    auto cpu_end = Clock::now();
    std::cout << "vector_add_cpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()
    << " nanoseconds.\n";

    for(int i=0;i<10;i++)
	std::cout << "vector_add_cpu : " << c[i] << " ";
    std::cout<<"\n";
    /*
    for(int i=0;i<10;i++)
	std::cout << "result : " << result[i] << " ";
    std::cout<<"\n";
    */

    // Call the GPU function and time it
    // The triple angle brakets is a CUDA runtime extension that allows
    // parameters of a CUDA kernel call to be passed.
    // In this example, we are passing one thread block with ITER threads.
    //cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_a, a, ITER * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b, ITER * sizeof(int), cudaMemcpyHostToDevice);
    
    auto gpu_start = Clock::now();
    //vector_add_gpu <<<2, ITER/2>>> (gpu_a, gpu_b, gpu_c, ITER);
    vector_add_gpu <<<1, ITER>>> (gpu_a, gpu_b, gpu_c, ITER);
    cudaDeviceSynchronize();
    auto gpu_end = Clock::now();
    std::cout << "vector_add_gpu: "
    << std::chrono::duration_cast<std::chrono::nanoseconds>(gpu_end - gpu_start).count()
    << " nanoseconds.\n";
     
    /*
    for(int i=0;i<10;i++)
	std::cout << "vector_add_gpu : " << gpu_r[i] << " ";
    std::cout<<"\n";
    */

    cudaMemcpy(gpu_r, gpu_c, ITER * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout<<"result of gpu_c"<<std::endl;
    for(int i=0;i<10;i++)
	std::cout << "vector_add_gpu : " << gpu_r[i] << " ";
    std::cout<<"\n";

    //Free the GPU-function based memory allocations
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    // Free the CPU-function based memory allocations
    free(a);
    free(b);
    free(c);
    free(gpu_r);
    /* 
    int InputData[5] = {1, 2, 3, 4, 5};
    int OutputData[5] = {0};
 
    int* GraphicsCard_memory;
 
    //그래픽카드 메모리의 할당
    cudaMalloc((void**)&GraphicsCard_memory, 5*sizeof(int));
 
    //PC에서 그래픽 카드로 데이터 복사
    cudaMemcpy(GraphicsCard_memory, InputData, 5*sizeof(int), cudaMemcpyHostToDevice);
 
    //그래픽 카드에서 PC로 데이터 복사
    cudaMemcpy(OutputData, GraphicsCard_memory, 5*sizeof(int), cudaMemcpyDeviceToHost);
 
    //결과 출력
    for( int i = 0; i < 5; i++)
    {
        printf(" OutputData[%d] : %d\n, i, OutputData[i]);
    }
 
    //그래픽 카드 메모리의 해체
    cudaFree(GraphicsCard_memory);
    */
    return 0;
}
