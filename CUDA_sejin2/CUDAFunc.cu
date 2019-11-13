#include "CUDAFunc.cuh"

// Conv, Pool layer
float *dev_Conv_kernel[nConvLayer], *dev_Conv_grad[nConvLayer], *dev_Conv_m_prev[nConvLayer], *dev_Conv_v_prev[nConvLayer];
char *devicePoolMark[nPoolLayer];
// FC Layer
float *dev_FC_w[nFCLayer], *dev_FC_grad[nFCLayer], *dev_FC_m_prev[nFCLayer], *dev_FC_v_prev[nFCLayer];
// shared node
float *dev_Node[nCnPLayer + nFCLayer + 1], *dev_Node_delta[nCnPLayer + nFCLayer];

// test drop node
float *dev_drop[nFCLayer - 1];

__global__ void forward_layer(float *d_weights, int weightOffset, int weightsPerNeuron, float *d_ins, int neuronsPrev, float *d_outs, bool softmax)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int wid = threadIdx.x * weightsPerNeuron + (blockIdx.x * weightsPerNeuron);
	
	float a = .0f;

	for (int i = 0; i < weightsPerNeuron; ++i){
		a += d_weights[wid + i] * d_ins[i];
				//printf("\n [tid:%d], d_weights[%d](%.1f) * d_ins[%d](%.1f) = %.1f", tid, wid+i, d_weights[wid+i], i, d_ins[i]);
		//printf("\n [tid:%d], d_weights[%d](%f) * d_ins[%d](%f) = %f", tid, wid + i, d_weights[wid + i], i, d_ins[i], a);
		//printf("d_outs[%d] : %f\n", tid, a);
	}


	if (softmax) d_outs[tid] = a;
	else		d_outs[tid] = (a > 0.0f ? a : a*0.01f);
}

__global__ void CUDA_Conv2D(float *I, float* M, float *P, int inmap, int outmap, int width, int height, int kernel_size, int padding)
{
	//CUDA_Conv2D << < Conv->Outmap, Conv->Output_data->y * Conv->Output_data->z >> > (dev_Node[layer_idx], dev_Conv_kernel[kernel_num], dev_Node[layer_idx + 1],
	//Conv->Inmap, Conv->Outmap, Conv->Input_data->y, Conv->Input_data->z, Conv->kernel->y, Padding);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = 0, tidx = 0, mask_idx = 0, oddCnt = 0;

	int x, y, outputSize, outputDim, _padding = 0;
	int mask_dim = kernel_size*kernel_size;
	float val = 0.0;

	if (padding == 0){
		outputSize = (width - (kernel_size - 1)) * (width - (kernel_size - 1));
		outputDim = width - (kernel_size - 1);
	}
	else{
		outputSize = width * height;
		outputDim = width;
		_padding = kernel_size / 2;
	}

	for (int mapCnt = 0; mapCnt < inmap; mapCnt++, tidx += (width * (outputDim))){
		//printf("\n mapCnt : %d", mapCnt);
		idx = tidx + threadIdx.x;

		x = idx / outputDim;
		y = idx % outputDim;

		oddCnt = mapCnt * mask_dim;

		for (int i = 0; i < kernel_size; i++){
			int xx = x + i - _padding;
			for (int j = 0; j < kernel_size; j++){
				int yy = y + j - _padding;

				// 나중에 조건문 제거할 것
				if (padding == 0){
					val += I[xx * width + yy] * M[((tid / (outputSize)) * mask_dim * inmap + oddCnt) + (i * kernel_size + j)];
					//printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f , i : %d , j : %d, tidx : %d, width : %d, outputDim : %d",
						//mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
						//I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val, i, j, tidx,width, outputDim);
					//printf("\n blockIdx : %d, threadIdx : %d, tidx : %d , blockDimx : %d, width : %d, outputDim : %d", tid, idx, blockIdx.x, threadIdx.x, tidx, blockDim.x, width, outputDim);
				}
				else{
					if ((xx >= 0 && yy >= 0) &&
						((xx < (width*(mapCnt + 1))) &&
						(yy < height)) &&
						((xx * width + yy) < (outputSize * (mapCnt + 1))) &&
						((xx * width + yy) >= tidx)){

						val += I[xx * width + yy] * M[((tid / (outputSize)) * mask_dim * inmap + oddCnt) + (i * kernel_size + j)]; 
						/*
						printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f , i : %d , j : %d, tidx : %d, width : %d, outputDim : %d",
						mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
							//I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val, i, j, tidx, width, outputDim);
							*/
						//printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f x : %d, y : %d, outputDim : %d",
							//mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
							//I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val, x, y, outputDim);
						/*
						if (tid == 0){
							printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f x : %d, y : %d",
								mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
								I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val, x, y);
						}
						*/
					}
				}
				/*
				printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f , i : %d , j : %d, tidx : %d, width : %d, outputDim : %d",
					mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
					I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val, i, j, tidx, width, outputDim);
					*/
				if (tid == 0){
					//printf("\n mapCnt : %d, idx : %d, tid : %d, val += I[%d] * M[%d] : %.4f * %.4f = %.3f ",
						//mapCnt, idx, tid, xx * width + yy, ((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j),
						//I[xx * width + yy], M[((tid / (outputSize))*mask_dim*inmap + oddCnt) + (i * kernel_size + j)], val);
				}
			}
		}
		P[tid] = (val > 0 ? val : val * 0.01);	// Act. func.

		//printf("\n P[%d] : %.3f", tid, P[tid]);
		//if (tid == 0)	printf("\n P[%d] : %.3f", tid, P[tid]);
	}
}

__global__ void CUDA_MaxPooling(float *I, float *P, char *pool_mark, int inmap, int width, int height){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = 0, tidx = 0, cnt = 0;
	char pool_idx = 0;

	const int outputDim = width / 2;

	//float _max = 0;

	int init_h = (threadIdx.x / outputDim) * 2;
	int init_w = (threadIdx.x % outputDim) * 2;

	float _max = I[(blockIdx.x * width * width) + (init_h * width + init_w)];

	//printf("\n B : %d, T : %d, tid  :%d, Max : %.4f", blockIdx.x, threadIdx.x, tid, _max);
	//const int outputDim = width / 2;

	for (int h = 0; h < 2; h++){

		int _h = h + (threadIdx.x / outputDim) * 2;

		for (int w = 0; w < 2; w++){

			int _w = w + (threadIdx.x % outputDim) * 2;  // 수정

			if (I[(blockIdx.x * width * width) + (_h * width + _w)] > _max){
				_max = I[(blockIdx.x * width * width) + (_h * width + _w)];
				pool_idx = cnt;
			}
			cnt++;
		}
	}
	P[tid] = _max;
	pool_mark[tid] = pool_idx;

	//printf("\n B:%d,T:%d, p[%d] : %.2f , pool_mark[%d] : %d", 
	//	blockIdx.x, threadIdx.x, tid, P[tid], tid, pool_mark[tid]);
}

////////////////////////////////////////////////////////////

void CUDA_CnP_MemAlloc(ConvLayer *Conv, PoolLayer *Pool){

	for (int i = 0; i < nConvLayer; i++){
		cudaMalloc((void **)&dev_Conv_kernel[i], (Conv[i].Inmap * Conv[i].Outmap) * (Conv[i].kernel->y * Conv[i].kernel->z) * sizeof(float));
		cudaMalloc((void **)&dev_Conv_grad[i], (Conv[i].Inmap * Conv[i].Outmap) * (Conv[i].kernel->y * Conv[i].kernel->z) * sizeof(float));
		cudaMalloc((void **)&dev_Conv_m_prev[i], (Conv[i].Inmap * Conv[i].Outmap) * (Conv[i].kernel->y * Conv[i].kernel->z) * sizeof(float));
		cudaMalloc((void **)&dev_Conv_v_prev[i], (Conv[i].Inmap * Conv[i].Outmap) * (Conv[i].kernel->y * Conv[i].kernel->z) * sizeof(float));

		reshape3Dto1D(Conv[i].kernel);

		cudaMemcpy(dev_Conv_kernel[i], Conv[i].kernel->mem1D,
			(Conv[i].Inmap * Conv[i].Outmap) * (Conv[i].kernel->y * Conv[i].kernel->z) * sizeof(float), cudaMemcpyHostToDevice);
	}
	for (int i = 0; i < nPoolLayer; i++){
		cudaMalloc((void **)&devicePoolMark[i], Pool[i].Outmap * Pool[i].pool_mark->y * Pool[i].pool_mark->z * sizeof(char));
	}

}
void CUDA_FC_MemAlloc(FCLayer *FC){

	for (int i = 0; i < nFCLayer; i++){
		if (i < nFCLayer - 1){
			cudaMalloc((void**)&dev_FC_w[i], FC[i].Input_size * FC[i + 1].Input_size * sizeof(float));
			cudaMalloc((void**)&dev_FC_grad[i], FC[i].Input_size * FC[i + 1].Input_size * sizeof(float));
			cudaMalloc((void**)&dev_FC_m_prev[i], FC[i].Input_size * FC[i + 1].Input_size * sizeof(float));
			cudaMalloc((void**)&dev_FC_v_prev[i], FC[i].Input_size * FC[i + 1].Input_size * sizeof(float));

			cudaMemcpy(dev_FC_w[i], FC[i].Weight->mem1D, FC[i].Input_size * FC[i + 1].Input_size * sizeof(float), cudaMemcpyHostToDevice);
		}
		else{
			cudaMalloc((void**)&dev_FC_w[i], FC[i].Input_size * FC[i].Output_size * sizeof(float));
			cudaMalloc((void**)&dev_FC_grad[i], FC[i].Input_size * FC[i].Output_size* sizeof(float));
			cudaMalloc((void**)&dev_FC_m_prev[i], FC[i].Input_size * FC[i].Output_size * sizeof(float));
			cudaMalloc((void**)&dev_FC_v_prev[i], FC[i].Input_size * FC[i].Output_size * sizeof(float));

			cudaMemcpy(dev_FC_w[i], FC[i].Weight->mem1D, FC[i].Input_size * FC[i].Output_size * sizeof(float), cudaMemcpyHostToDevice);
		}
	}

}
void CUDA_ioNode_MemAlloc(MemBlock32F *ioNode){

	// 입력 노드
	cudaMalloc((void**)&dev_Node[0], ioNode[0].total * sizeof(float));
	for (int i = 1; i < nCnPLayer + nFCLayer + 1; i++){
		cudaMalloc((void**)&dev_Node[i], ioNode[i].total * sizeof(float));
		cudaMalloc((void**)&dev_Node_delta[i], ioNode[i].total * sizeof(float));
	}

#if Dropout
	for (int i = 1; i < nFCLayer; i++)
		cudaMalloc((void**)&dev_drop[i - 1], (ioNode + nCnPLayer + i)->total * sizeof(int));
#endif

}

void CUDA_FC_Forward(FCLayer *FC, int &layer_idx){

	int block = 0, thread = 0;
	int offset = (FC - 1)->Input_size * (FC - 1)->Output_size;
	bool softmax = false;

	if (layer_idx == 0) offset = 0;
	if (layer_idx == nFCLayer - 1) softmax = true;

	if (FC->Output_size % MAX_THREAD == 0){
		block = FC->Output_size / MAX_THREAD;
	}
	else{
		block = (FC->Output_size / MAX_THREAD) + 1;
	}

	if (block > 1) thread = MAX_THREAD;
	else		   thread = FC->Output_size;

	//printf("forward layer\n");
	//printf("offset : %d, FC->Input_size : %d\n", offset, FC->Input_size);
	//printf("block : %d, thread : %d\n", block, thread);
	//		forward_layer << < 1, FC->Output_size >> > (dev_FC_w[layer_idx], offset, FC->Input_size, dev_Node[nCnPLayer + layer_idx], FC->Input_size, dev_Node[nCnPLayer + layer_idx + 1], softmax);
	forward_layer << < block, thread >> > (dev_FC_w[layer_idx], offset, FC->Input_size, dev_Node[nCnPLayer + layer_idx], FC->Input_size, dev_Node[nCnPLayer + layer_idx + 1], softmax);

	layer_idx++;
}
void CUDA_Conv_Forward(ConvLayer *Conv, int Padding, int &layer_idx, int &kernel_num){
	//CUDA_Conv_Forward(&Conv[0], ConvPad_info[0], CnP_layer_idx, kernel_num);

	CUDA_Conv2D << < Conv->Outmap, Conv->Output_data->y * Conv->Output_data->z >> > (dev_Node[layer_idx], dev_Conv_kernel[kernel_num], dev_Node[layer_idx + 1],
		Conv->Inmap, Conv->Outmap, Conv->Input_data->y, Conv->Input_data->z, Conv->kernel->y, Padding);


	// update에 사용.. update CUDA 적용 시 제거
	cudaMemcpy(Conv->Output_data->mem1D, dev_Node[layer_idx], Conv->Output_data->total * sizeof(float), cudaMemcpyDeviceToHost);
	reshape1Dto3D(Conv->Output_data);

	layer_idx++; // 인덱스 증가
	kernel_num++;

}
void CUDA_Pool_Forward(PoolLayer *Pool, int &layer_idx, int &poolMark_num){

	CUDA_MaxPooling << < Pool->Inmap, Pool->Output_data->y * Pool->Output_data->z >> >(dev_Node[layer_idx], dev_Node[layer_idx + 1], devicePoolMark[poolMark_num]
		, Pool->Inmap, Pool->Input_data->y, Pool->Input_data->z);

	cudaMemcpy(Pool->Output_data->mem1D, dev_Node[layer_idx + 1], Pool->Output_data->total * sizeof(float), cudaMemcpyDeviceToHost);
	reshape1Dto3D(Pool->Output_data);	 // update에 사용.. update CUDA 적용 시 제거

	layer_idx++;	// 인덱스 증가
	poolMark_num++;
}
