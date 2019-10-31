
//////////////////////////////////////////////
// CNN CUDA ÇÔ¼ö      ////////////////////////
//////////////////////////////////////////////

#ifndef __CUDAFUNC_H__
#define __CUDAFUNC_H__

#include "cuda_runtime.h"

#include "Config.h"
#include "CNNFunc.h"
#include <stdio.h>

// Conv, Pool layer
extern float *dev_Conv_kernel[nConvLayer], *dev_Conv_grad[nConvLayer], *dev_Conv_m_prev[nConvLayer], *dev_Conv_v_prev[nConvLayer];
extern char *devicePoolMark[nPoolLayer];
// FC Layer
extern float *dev_FC_w[nFCLayer], *dev_FC_grad[nFCLayer], *dev_FC_m_prev[nFCLayer], *dev_FC_v_prev[nFCLayer];
// shared node
extern float *dev_Node[nCnPLayer + nFCLayer + 1], *dev_Node_delta[nCnPLayer + nFCLayer];

#if Dropout
// test dropout node
extern float *dev_drop[nFCLayer - 1];
#endif

__global__ void forward_layer(float *d_weights, int weightOffset, int weightsPerNeuron, float *d_ins, int neuronsPrev, float *d_outs, bool softmax);

__global__ void CUDA_Conv2D(float *I, float* M, float *P, int inmap, int outmap, int width, int height, int kernel_size, int padding);

__global__ void CUDA_MaxPooling(float *I, float *P, char *pool_mark, int inmap, int width, int height);

void CUDA_CnP_MemAlloc(ConvLayer *Conv, PoolLayer *Pool);
void CUDA_FC_MemAlloc(FCLayer *FC);
void CUDA_ioNode_MemAlloc(MemBlock32F *ioNode);

void CUDA_FC_Forward(FCLayer *FC, int &layer_idx);
void CUDA_Conv_Forward(ConvLayer *Conv, int Padding, int &layer_idx, int &kernel_num);
void CUDA_Pool_Forward(PoolLayer *Pool, int &layer_idx, int &poolMark_num);

#endif