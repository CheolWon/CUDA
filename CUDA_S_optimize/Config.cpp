#include "Config.h"

//cv::Mat img;

ConvLayer Conv_Info(MemBlock32F *input, MemBlock32F *output, MemBlock32F *kernel, MemBlock32F *input_Delta, MemBlock32F *output_Delta, 
	MemBlock32F *deltaPad, MemBlock32F *Moment, MemBlock32F *gradSum, MemBlock32F *mean_prev, MemBlock32F *var_prev, MemBlock32F *ConvPad){
	ConvLayer layer_info;

	layer_info.Inmap = input->x;						// 입력 특징맵 수
	layer_info.Outmap = output->x;						// 출력 특징맵 수
	layer_info.Input_size = input->y;					// 입력 특징맵 크기
	layer_info.Output_size = input->y - (kernel->y - 1);// 출력 특징맵 크기

	layer_info.Input_data = input;
	layer_info.Output_data = output;
	layer_info.kernel = kernel;
	layer_info.kernel_num = layer_info.Inmap * layer_info.Outmap; // 커널 개수

	layer_info.input_Delta = input_Delta;
	layer_info.output_Delta = output_Delta;

	layer_info.Moment = Moment;
	layer_info.deltaPad = deltaPad;
	layer_info.gradSum = gradSum;
	layer_info.mean_prev = mean_prev;
	layer_info.var_prev = var_prev;

	layer_info.ConvPad = ConvPad;

	int Cnt1D, i;
	 
	// 커널 초기화
	for (Cnt1D = 0, i = 0; i < layer_info.kernel_num; i++){
		for (int h = 0; h < kernel->y; h++){
			for (int w = 0; w < kernel->z; w++){
				//가우시안 분포
				//layer_info.kernel->mem3D[i][h][w] = gaussianRandom(0, 0.01);	// 평균 0, 표준편차 0.01 (0.05x)	
				if (CnP_ReLU){				// 6.0
					float init_range = sqrt(6.0 / ((kernel->x * kernel->y * kernel->z) + 1));
					float Kernel = -init_range + (rand() / (RAND_MAX / (init_range - (-init_range))));

					layer_info.kernel->mem3D[i][h][w] = Kernel;
				}
				else if (CnP_Sigmoid){
					layer_info.kernel->mem3D[i][h][w] = ((float)(rand() % 500 + 1) / 1000.0f) * ((float)(rand() % 2 == 0 ? 1 : -1));
				}
			}
		}
	}

	return layer_info;
}

PoolLayer Pool_Info(MemBlock32F *input, MemBlock32F *output, MemBlock8C *pool_mark, MemBlock32F *input_Delta, MemBlock32F *output_Delta){
	PoolLayer layer_info;

	if (input->y % 2 != 0)
		cout << " # 풀링 입력 사이즈가 홀수 입니다." << endl;
	else{
		layer_info.Inmap = input->x;
		layer_info.Outmap = output->x;
		layer_info.Input_size = input->y;
		layer_info.Output_size = input->y / 2;  // pooling 커널 =  2

		layer_info.Input_data = input;
		layer_info.Output_data = output;
		layer_info.pool_mark = pool_mark;

		layer_info.input_Delta = input_Delta;
		layer_info.output_Delta = output_Delta;
	}
	return layer_info;
}
FCLayer	  FC_Info(MemBlock32F *input, MemBlock32F *output, MemBlock32F *Weight, MemBlock32F *input_delta, MemBlock32F *output_delta, MemBlock32F *Moment, MemBlock32F *gradSum, MemBlock32F *mean_prev, MemBlock32F *var_prev){
	FCLayer layer_info;

	layer_info.Input_data = input;
	layer_info.Output_data = output;
	layer_info.Input_size = input->total;
	layer_info.Output_size = output->total;

	layer_info.Weight = Weight;
	layer_info.input_Delta = input_delta;
	layer_info.output_Delta = output_delta;

	layer_info.Moment = Moment;
	layer_info.gradSum = gradSum;

	layer_info.mean_prev = mean_prev;
	layer_info.var_prev = var_prev;

#if !CUDA
	// 가중치 초기화
	for (int i = 0; i < layer_info.Input_size; i++){
		for (int j = 0; j < layer_info.Output_size; j++){
			if (FC_ReLU){
				// 가중치 초기화 ( = Xavier initialization )   M + (rand() / ( RAND_MAX / (N-M) ) ) // [M,N]
				float init_range = sqrt(6.0 / (Weight->y + Weight->z));
				float Weight = -init_range + (rand() / (RAND_MAX / (init_range - (-init_range))));

				layer_info.Weight->mem2D[i][j] = Weight;
			}
			else if (FC_Sigmoid){
				layer_info.Weight->mem2D[i][j] = ((float)(rand() % 500 + 1) / 1000.0f) * ((float)(rand() % 2 == 0 ? 1 : -1));
			}
			// 가중치 초기화 ( = Gaussian Distribution )			
			//	layer_info.Weight->mem2D[i][j] = gaussianRandom(0, 0.1);	// 평균 0, 표준편차 0.01
		}
	}
#endif
#if CUDA
	for (int i = 0; i < layer_info.Input_size * layer_info.Output_size; i++){
		if (FC_ReLU){		      // 6.0
			float init_range = sqrt(6.0 / (Weight->y + Weight->z));
			float Weight = -init_range + (rand() / (RAND_MAX / (init_range - (-init_range))));
			layer_info.Weight->mem1D[i] = Weight;
		}
		else if (FC_Sigmoid){
			layer_info.Weight->mem1D[i] = ((float)(rand() % 500 + 1) / 1000.0f) * ((float)(rand() % 2 == 0 ? 1 : -1));
		}
	}
#endif

	return layer_info;
}

void setLayer(int input_size, int depth, int *map, int *kernel_size, int *ConvPad_info, int *Layerinfo, MemBlock32F *prevPatch,
	MemBlock32F *kernel, MemBlock32F *ConvPad, MemBlock8C *pool_mark, MemBlock32F *deltaPad, MemBlock32F *CNN_Delta, MemBlock32F *CNN_Moment, MemBlock32F *CNN_gradSum, MemBlock32F *CNN_mean_prev, MemBlock32F *CNN_var_prev, MemBlock32F *ioNode, ConvLayer *Conv, PoolLayer *Pool,
	MemBlock32F *FC_Weight, MemBlock32F *FC_Moment, MemBlock32F *FC_gradSum, MemBlock32F *FC_mean_prev, MemBlock32F *FC_var_prev, FCLayer *FC){

	int map_size[nCnPLayer + 1], num_conv = 0, num_pool = 0;

	// Input
	map_size[0] = input_size;
	ioNode[0] = CreateMemBlock32F(depth, input_size, input_size);
	// 테스트 출력	
	//printf("\n ##########################");
	//printf("\n # Input : %d @ %dx%d", depth, input_size, input_size);

	// Conv,Pool Layer //////////////////////////////////////////////////////////////
	for (int i = 1; i < nCnPLayer + 1; i++){

		if (Layerinfo[i - 1] == 1){
			if (ConvPad_info[num_conv] == 1){
				map_size[i] = map_size[i - 1];
			}
			else{
				map_size[i] = map_size[i - 1] - (kernel_size[num_conv] - 1);
			}

			if (i == 1){
				kernel[num_conv] = CreateMemBlock32F(depth * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				CNN_gradSum[num_conv] = CreateMemBlock32F(depth * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				if (ConvPad_info[num_conv] == 1) ConvPad[num_conv] = CreateMemBlock32F(depth, input_size + (kernel_size[num_conv] - 1), input_size + (kernel_size[num_conv] - 1));
				CNN_Moment[num_conv] = CreateMemBlock32F(depth * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);

				CNN_mean_prev[num_conv] = CreateMemBlock32F(depth * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				CNN_var_prev[num_conv] = CreateMemBlock32F(depth * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
			}

			else{
				kernel[num_conv] = CreateMemBlock32F(map[num_conv - 1] * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				CNN_gradSum[num_conv] = CreateMemBlock32F(map[num_conv - 1] * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				if (ConvPad_info[num_conv] == 1) ConvPad[num_conv] = CreateMemBlock32F(map[num_conv - 1], map_size[i - 1] + (kernel_size[num_conv] - 1), map_size[i - 1] + (kernel_size[num_conv] - 1));
				CNN_Moment[num_conv] = CreateMemBlock32F(map[num_conv - 1] * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);

				CNN_mean_prev[num_conv] = CreateMemBlock32F(map[num_conv - 1] * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);
				CNN_var_prev[num_conv] = CreateMemBlock32F(map[num_conv - 1] * map[num_conv], kernel_size[num_conv], kernel_size[num_conv]);

				deltaPad[num_conv - 1] = CreateMemBlock32F(map[num_conv], map_size[i] + ((kernel_size[num_conv] - 1) * 2), map_size[i] + ((kernel_size[num_conv] - 1) * 2));
			}

			// 테스트 출력
			//if (ConvPad_info[num_conv] == 1)
			//	printf("\n # Conv[%d] : %d @ %dx%d (padding)", num_conv, map[num_conv], map_size[i], map_size[i]);
			//else
			//	printf("\n # Conv[%d] : %d @ %dx%d ", num_conv, map[num_conv], map_size[i], map_size[i]);

			num_conv++;
		}

		else if (Layerinfo[i - 1] == 0){
			map_size[i] = map_size[i - 1] / 2;

			pool_mark[num_pool] = CreateMemBlock8C(map[num_conv - 1], map_size[i], map_size[i]);

			// 테스트 출력
			//printf("\n # Pool[%d] : %d @ %dx%d", num_pool, map[num_conv - 1], map_size[i], map_size[i]);

			num_pool++;
		}
		ioNode[i] = CreateMemBlock32F(map[num_conv - 1], map_size[i], map_size[i]);
		CNN_Delta[i - 1] = CreateMemBlock32F(map[num_conv - 1], map_size[i], map_size[i]);
		////// LRN에 사용 ///////
		prevPatch[i] = CreateMemBlock32F(map[num_conv - 1], map_size[i], map_size[i]);
		////////////////////////
		if (Layerinfo[i - 1] == 1){
			if (num_conv == 1)
				Conv[num_conv - 1] = Conv_Info(&ioNode[i - 1], &ioNode[i], &kernel[num_conv - 1], &CNN_Delta[i - 1], NULL, NULL, &CNN_Moment[num_conv - 1], &CNN_gradSum[num_conv - 1], &CNN_mean_prev[num_conv - 1], &CNN_var_prev[num_conv - 1], &ConvPad[num_conv - 1]);
			else
				Conv[num_conv - 1] = Conv_Info(&ioNode[i - 1], &ioNode[i], &kernel[num_conv - 1], &CNN_Delta[i - 1], &CNN_Delta[i - 2], &deltaPad[num_conv - 2], &CNN_Moment[num_conv - 1], &CNN_gradSum[num_conv - 1], &CNN_mean_prev[num_conv - 1], &CNN_var_prev[num_conv - 1], &ConvPad[num_conv - 1]);
		}
		else{
			Pool[num_pool - 1] = Pool_Info(&ioNode[i - 1], &ioNode[i], &pool_mark[num_pool - 1], &CNN_Delta[i - 1], &CNN_Delta[i - 2]);
		}

	}

	// FC Layer ////////////////////////////////////////////////////////////
	int FC_Node[nFCLayer + 1], fc_cnt = 1;

	FC_Node[0] = map_size[nCnPLayer] * map_size[nCnPLayer] * map[nConvLayer - 1]; // flat node
	FC_Node[nFCLayer] = NUM_OUTPUTS;

	for (int i = 1; i < nFCLayer; i++) FC_Node[i] = FC_Node[i - 1] * 0.5; // / 2;

	for (int i = nCnPLayer + 1; i < nCnPLayer + nFCLayer + 1; i++, fc_cnt++){
		ioNode[i] = CreateMemBlock32F(FC_Node[fc_cnt], 1, 1);
		CNN_Delta[i - 1] = CreateMemBlock32F(FC_Node[fc_cnt], 1, 1);
	}

	for (int i = 0; i < nFCLayer; i++){
		FC_Weight[i] = CreateMemBlock32F(1, FC_Node[i], FC_Node[i + 1]);
		FC_Moment[i] = CreateMemBlock32F(1, FC_Node[i], FC_Node[i + 1]);
		FC_gradSum[i] = CreateMemBlock32F(1, FC_Node[i], FC_Node[i + 1]);
		FC_mean_prev[i] = CreateMemBlock32F(1, FC_Node[i], FC_Node[i + 1]);
		FC_var_prev[i] = CreateMemBlock32F(1, FC_Node[i], FC_Node[i + 1]);
	}
	for (int i = 0; i < nFCLayer; i++){
		// 테스트 출력
		//printf("\n # FC[%d] : %d", i, FC_Node[i]);

		FC[i] = FC_Info(&ioNode[num_conv + num_pool + i], &ioNode[num_conv + num_pool + i + 1],
			&FC_Weight[i], &CNN_Delta[i + nCnPLayer], &CNN_Delta[i + (nCnPLayer - 1)], &FC_Moment[i], &FC_gradSum[i],
			&FC_mean_prev[i], &FC_var_prev[i]);
	}
	// 테스트 출력
	//printf("\n # Output : %d \n", NUM_OUTPUTS);
}
