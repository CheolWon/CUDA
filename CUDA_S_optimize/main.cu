//cuda 1d
#include <iostream>
#include <time.h>
#include <math.h>
#include <stdio.h>			
#include <stdlib.h>
#include <string.h>

#include "Define.h"
#include "MemBlock.h"
#include "Config.h"
#include "utils.h"
#include "CNNFunc.h"
#include "CUDAFunc.cuh"
#include "cuda_runtime.h"

#define lrn_alpha		0.0001		// 0.0001f
#define lrn_beta		0.5		// 0.75f
#define	lrn_radius		5		// 5
#define lrn_k			2.0		// 1.0f
		
int main(){
	
	srand((unsigned int)time(NULL));
	float		 train_input[Image_W*Image_depth][Image_H], fc_target[NUM_OUTPUTS];
	unsigned int train_label;

	float target[NUM_OUTPUTS][NUM_OUTPUTS] = { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },  // '0','1','2'
											{ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },	  // '3','4','5' 
											{ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },	  // '6','7','8'											
											{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } }; 																	  // '9'

	///////////////////////////////////////////////////////////////////////////////////
	// 네트워크 구조 설정
	int input_size, map[nConvLayer], kernel_size[nConvLayer];
	int Layerinfo[nCnPLayer] = { 1, 0, 1, 0 };		// 1:conv,0:pool
	int ConvPad_info[nConvLayer] = { 0, 0 };		// 1:padding
	//int ConvPad_info[nConvLayer] = { 1, 1 };		// 1:padding
	input_size = Image_H;							// Input data	
	
	map[0] = 8, kernel_size[0] = 5;
	map[1] = 16, kernel_size[1] = 3;  

	//char FilePath[30] = "./mnist_0_3";
	//char FilePath[30] = "./mnist_5_728";
	char FilePath[30] = "./mnist_9_169";
	
	///////////////////////////////////////////////////////////////////////////////////
	ConvLayer	Conv[nConvLayer];
	PoolLayer	Pool[nPoolLayer];
	FCLayer		FC[nFCLayer];
	memset(&Conv, 0, sizeof(ConvLayer[nConvLayer]));
	memset(&Pool, 0, sizeof(PoolLayer[nPoolLayer]));
	memset(&FC, 0, sizeof(FCLayer[nFCLayer]));

	// Layer 간 공유 노드
	MemBlock32F ioNode[nCnPLayer + nFCLayer + 1], CNN_Delta[nCnPLayer + nFCLayer], prevPatch[nCnPLayer + 1];
	// Conv,Pool Layer	
	MemBlock32F kernel[nConvLayer], ConvPad[nConvLayer], deltaPad[nConvLayer - 1], CNN_Moment[nConvLayer],
		CNN_gradSum[nConvLayer], CNN_mean_prev[nConvLayer], CNN_var_prev[nConvLayer];
	MemBlock8C	pool_mark[nPoolLayer];
	// FC Layer
	MemBlock32F FC_Weight[nFCLayer], FC_Moment[nFCLayer], FC_gradSum[nFCLayer], FC_mean_prev[nFCLayer], FC_var_prev[nFCLayer];

	setLayer(input_size, Image_depth, map, kernel_size, ConvPad_info, Layerinfo, prevPatch,
		kernel, ConvPad, pool_mark, deltaPad, CNN_Delta, CNN_Moment, CNN_gradSum, CNN_mean_prev, CNN_var_prev, ioNode, Conv, Pool,
		FC_Weight, FC_Moment, FC_gradSum, FC_mean_prev, FC_var_prev, FC);
	//cout << "ifCUDA" << endl;
#if CUDA
	int kernel_num = 0, poolMark_num = 0, CnP_layer_idx = 0, FC_layer_idx = 0;

	// shared node (CUDA memAlloc)
	CUDA_ioNode_MemAlloc(ioNode);

	// FC Layer (CUDA memAlloc)
	const int node4 = NUM_OUTPUTS;

	CUDA_FC_MemAlloc(FC);

	// Conv/Pool Layer (CUDA memAlloc)
	CUDA_CnP_MemAlloc(Conv, Pool);
#endif
#if Training
	int num_cnt[NUM_OUTPUTS], _num_cnt[NUM_OUTPUTS], randVal, epoch, num_data;
	float total_loss = 0, loss = 0, acc = 0, _LearningRate = learningRate, batch_loss = 0, time = 0;
	if (Test){ // Test 
		loadWeight(Conv, FC);
		for (int i = 0; i < nFCLayer; i++){
			cudaMemcpy(dev_FC_w[i], FC[i].Weight->mem1D, FC[i].Weight->total * sizeof(float), cudaMemcpyHostToDevice);
		}
		for (int i = 0; i < nConvLayer; i++){
			reshape3Dto1D(Conv[i].kernel);
			cudaMemcpy(dev_Conv_kernel[i], Conv[i].kernel->mem1D, Conv[i].kernel->total * sizeof(float), cudaMemcpyHostToDevice);
		}
		epoch = 1;
		num_data = NUM_TEST_DATA;
		for (int i = 0; i < NUM_OUTPUTS; i++)  _num_cnt[i] = _NUM_TEST_DATA;
	}
	else{	  // Train
		epoch = iter;
		num_data = NUM_TRAIN_DATA;
	}
	for (int i = 0; i < 1; ++i) {  // Epoch
		// epoch 당 초기화 변수
		for (int j = 0; j < NUM_OUTPUTS; j++)  num_cnt[j] = _NUM_TRAIN_DATA; // num_cnt[i] = MNIST;
		if(!Test)	 for (int j = 0; j < NUM_OUTPUTS; j++)  _num_cnt[j] = _NUM_VALID_DATA;
		total_loss = acc = 0;

		// learning rate decay 
		if ((i + 1) % 10 == 0) _LearningRate *= 0.9;
		//cout << "validation" << endl;

		for (int q = 0; q < 1; q++) {
			randVal = ((rand() % 100) % NUM_OUTPUTS);
				//readDataset(train_input, _num_cnt, train_label, randVal, path, 0); 
			readDataset(train_input, _num_cnt, train_label, randVal, FilePath, 0);
			copyInputData1D(train_input, &ioNode[0]);
			
			for (int i = 0; i < 28; i++) {
				for (int j = 0; j < 28; j++) {
					printf("%1.0f ", Conv[0].Input_data->mem1D[i * 28 + j]);
				}
				printf("\n");
			}
			
			cudaMemcpy(dev_Node[0], Conv[0].Input_data->mem1D, Image_H * Image_W * Image_depth * sizeof(float), cudaMemcpyHostToDevice);
			///// 특징추출부 ///////////////////////////////////////////////////////////

			CUDA_Conv_Forward(&Conv[0], ConvPad_info[0], CnP_layer_idx, kernel_num);
			
			CUDA_Pool_Forward(&Pool[0], CnP_layer_idx, poolMark_num);
			CUDA_Conv_Forward(&Conv[1], ConvPad_info[1], CnP_layer_idx, kernel_num);

			CUDA_Pool_Forward(&Pool[1], CnP_layer_idx, poolMark_num);


			memcpy(FC[0].Input_data->mem1D, Pool[1].Output_data->mem1D, FC[0].Input_data->total);

			
			CUDA_FC_Forward(&FC[0], FC_layer_idx); // 드롭아웃 테스트


			cudaMemcpy(FC[0].Output_data->mem1D, dev_Node[nCnPLayer + 1], 200 * sizeof(float), cudaMemcpyDeviceToHost);


			CUDA_FC_Forward(&FC[1], FC_layer_idx);


			// output (toHost)
			cudaMemcpy(FC[1].Output_data->mem1D, dev_Node[nCnPLayer + nFCLayer], node4 * sizeof(float), cudaMemcpyDeviceToHost);

			/////////////////////////////////////////////////////////////////////////////
			// Softmax, Output Error, Loss function (cross-entropy)
			int label = train_label;
			memcpy(fc_target, target[label], sizeof(target[label]));

			loss = 0;
			Softmax_CrossEntropy(&acc, &loss, fc_target, &FC[nFCLayer - 1]);
			total_loss += loss;
#endif
		}
		acc /= num_data;
		total_loss /= num_data;
		printf("\n");
	}

	// 메모리 해제 //////////////////////////////////////////////////////////////
	// # 공유노드
	for (int i = 0; i < nCnPLayer + nFCLayer + 1; i++) memRelease32F(&ioNode[i]);
	for (int i = 0; i < nCnPLayer + nFCLayer; i++)  memRelease32F(&CNN_Delta[i]);

	// # 특징추출부
	for (int i = 0; i < nConvLayer; i++){
		memRelease32F(&kernel[i]);
		memRelease32F(&CNN_Moment[i]);
		memRelease32F(&CNN_gradSum[i]);
		if (i < nConvLayer - 1) memRelease32F(&deltaPad[i]);
	}
	for (int i = 0; i < nPoolLayer; i++)		memRelease8C(&pool_mark[i]);

	// # 분류부
	for (int i = 0; i < nFCLayer; i++){
		memRelease32F(&FC_Weight[i]);
		memRelease32F(&FC_Moment[i]);
		memRelease32F(&FC_gradSum[i]);
	}

#if CUDA
	// Shared node
	for (int i = 0; i < nCnPLayer + nFCLayer + 1; i++){
		cudaFree(dev_Node[i]);
		cudaFree(dev_Node_delta[i]);
	}
	// Conv layer
	for (int i = 0; i < nConvLayer; i++){
		cudaFree(dev_Conv_kernel[i]);
		cudaFree(dev_Conv_grad[i]);
		cudaFree(dev_Conv_m_prev[i]);
		cudaFree(dev_Conv_v_prev[i]);
	}
	// Pool layer
	for (int i = 0; i < nPoolLayer; i++){
		cudaFree(devicePoolMark[i]);
	}
	// FC layer
	for (int i = 0; i < nFCLayer; i++){
		cudaFree(dev_FC_w[i]);
		cudaFree(dev_FC_grad[i]);
		cudaFree(dev_FC_m_prev[i]);
		cudaFree(dev_FC_v_prev[i]);
	}
#endif

	return 0;
}

