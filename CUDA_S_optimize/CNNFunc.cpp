#include "CNNFunc.h"

void Softmax_CrossEntropy(float *acc, float *loss, float *fc_target, FCLayer *FC){

	// Output Delta ... [ delta = output*(1-output)*(target-output) → μ*delta*node ] (sigmoid)
	// FC output layer ( Soft-max )
	float probabilities[NUM_OUTPUTS], sum = 0, _out = 0; int o = 0;
	memset(&probabilities, 0, sizeof(probabilities[NUM_OUTPUTS]));

	float max = 0;
	for (int i = 0; i < FC->Output_size; i++){
		_out = FC->Output_data->mem1D[i];
		if (_out > max) max = _out;
	}

	for (sum = 0, o = 0; o < FC->Output_size; o++, _out = 0){
		_out = FC->Output_data->mem1D[o];
		probabilities[o] = exp(_out - max);		//  ###  exp(out)
		sum += probabilities[o];
		//	printf("\n out : %lf, exp_out[%d] : %lf",_out ,o, probabilities[o]);
	}
	//printf("\n");

	for (int o = 0; o < FC->Output_size; o++){
		probabilities[o] /= sum;
		//	printf("\n softmax_prob[%d] : %.5f", o, probabilities[o]);
	}

	// FC output layer ( error, loss function )
	for (int i = 0; i < FC->Output_size; i++){
		//_out = FC->Output_data->mem1D[i];
		// Cross-entropy loss function : -∑[t*log(o) + (1-t)*log(1-o)], < softmax일때, -∑t*log(o) >
		//if (probabilities[i] < 1e-10)		probabilities[i] = 1e-10;
		//else if (probabilities[i] > 1.0)	probabilities[i] = 1.0;
		probabilities[i] += 1e-10;

		*loss -= fc_target[i] * log(probabilities[i]);
		FC->input_Delta->mem1D[i] = (probabilities[i] - fc_target[i]);	// FC Output layer delta (Cross-entropy)
	}

	for (int i = 0; i < FC->Output_size; i++) {
		if (i == 0) printf("\n  [ 0 \t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 1) printf("\n  [ 1\t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 2) printf("\n  [ 2\t: %.2f %% ] ", probabilities[i] * 100);
		else if (i == 3) printf("\n  [ 3\t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 4) printf("\n  [ 4\t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 5) printf("\n  [ 5\t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 6) printf("\n  [ 6\t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 7) printf("\n  [ 7 \t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 8) printf("\n  [ 8 \t: %.2f %% ]", probabilities[i] * 100);
		else if (i == 9) printf("\n  [ 9\t: %.2f %% ]", probabilities[i] * 100);
	}


#if showImg
	if (Test && MNIST){
		for (int i = 0; i < FC->Output_size; i++){
			if (i == 0) printf("\n  [ 0 \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 1) printf("\n  [ 1\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 2) printf("\n  [ 2\t: %.2f %% ] ", probabilities[i] * 100);
			else if (i == 3) printf("\n  [ 3\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 4) printf("\n  [ 4\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 5) printf("\n  [ 5\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 6) printf("\n  [ 6\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 7) printf("\n  [ 7 \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 8) printf("\n  [ 8 \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 9) printf("\n  [ 9\t: %.2f %% ]", probabilities[i] * 100);
		}

		int tmp_acc = 0;
		if (getMax(probabilities, NUM_OUTPUTS) == getMax(fc_target, NUM_OUTPUTS)) tmp_acc = 100;

		if ((int)getMax(probabilities, NUM_OUTPUTS) == 0)		printf("\n\n  # predict : 0");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 1)	printf("\n\n  # predict : 1");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 2)	printf("\n\n  # predict : 2");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 3)	printf("\n\n  # predict : 3");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 4)	printf("\n\n  # predict : 4");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 5)	printf("\n\n  # predict : 5");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 6)	printf("\n\n  # predict : 6");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 7)	printf("\n\n  # predict : 7");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 8)	printf("\n\n  # predict : 8");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 9)	printf("\n\n  # predict : 9");

		if ((int)getMax(fc_target, NUM_OUTPUTS) == 0)		printf(" / target : 0");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 1)	printf(" / target : 1");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 2)	printf(" / target : 2");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 3)	printf(" / target : 3");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 4)	printf(" / target : 4");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 5)	printf(" / target : 5");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 6)	printf(" / target : 6");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 7)	printf(" / target : 7");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 8)	printf(" / target : 8");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 9)	printf(" / target : 9");


		if (getMax(probabilities, NUM_OUTPUTS) == getMax(fc_target, NUM_OUTPUTS)) printf(" (O) \n\n\n");
		else printf(" (X) \n\n\n");
		resize(img, img, Size(150, 150));
		imshow("test", img);
		waitKey(0);
	}

	if (Test && CIFAR10){
		for (int i = 0; i < FC->Output_size; i++){
			if (i == 0) printf("\n  [ frog \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 1) printf("\n  [ truck\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 2) printf("\n  [ deer\t: %.2f %% ] ", probabilities[i] * 100);
			else if (i == 3) printf("\n  [ automobile\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 4) printf("\n  [ bird\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 5) printf("\n  [ horse\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 6) printf("\n  [ ship\t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 7) printf("\n  [ cat \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 8) printf("\n  [ dog \t: %.2f %% ]", probabilities[i] * 100);
			else if (i == 9) printf("\n  [ airplane\t: %.2f %% ]", probabilities[i] * 100);
		}

		int tmp_acc = 0;
		if (getMax(probabilities, NUM_OUTPUTS) == getMax(fc_target, NUM_OUTPUTS)) tmp_acc = 100;

		if ((int)getMax(probabilities, NUM_OUTPUTS) == 0)		printf("\n\n  # predict : frog");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 1)	printf("\n\n  # predict : truck");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 2)	printf("\n\n  # predict : deer");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 3)	printf("\n\n  # predict : automobile");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 4)	printf("\n\n  # predict : bird");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 5)	printf("\n\n  # predict : horse");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 6)	printf("\n\n  # predict : ship");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 7)	printf("\n\n  # predict : cat");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 8)	printf("\n\n  # predict : dog");
		else if ((int)getMax(probabilities, NUM_OUTPUTS) == 9)	printf("\n\n  # predict : airplane");

		if ((int)getMax(fc_target, NUM_OUTPUTS) == 0)		printf(" / target : frog");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 1)	printf(" / target : truck");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 2)	printf(" / target : deer");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 3)	printf(" / target : automobile");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 4)	printf(" / target : bird");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 5)	printf(" / target : horse");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 6)	printf(" / target : ship");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 7)	printf(" / target : cat");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 8)	printf(" / target : dog");
		else if ((int)getMax(fc_target, NUM_OUTPUTS) == 9)	printf(" / target : airplane");

		if (getMax(probabilities, NUM_OUTPUTS) == getMax(fc_target, NUM_OUTPUTS)) printf(" (O) \n\n\n");
		else printf(" (X) \n\n\n");
		resize(img, img, Size(150, 150));
		imshow("test", img);
		waitKey(0);
	}
#endif

	// 반복 횟수별로 total loss 구하기 위해 누적합, 정확도 계산
	if (getMax(probabilities, NUM_OUTPUTS) == getMax(fc_target, NUM_OUTPUTS))	*acc += 100;
}

void reshape3Dto1D(MemBlock32F *ioNode){

	int mem1d_cnt = 0; float cpy = 0;
	for (int m = 0; m < ioNode->x; ++m){						// CNN Layer의 마지막 특징 맵 수
		for (int y = 0; y < ioNode->y; ++y){				// 특징 맵 크기
			for (int x = 0; x < ioNode->z; ++x){
				cpy = ioNode->mem3D[m][y][x];				// ioNode[4]의 3차원배열을
				//FC->Input_data->mem1D[mem1d_cnt] = cpy;	// FC[0].input_data의 1차원배열로 복사 
				ioNode->mem1D[mem1d_cnt] = cpy;				// ( 어차피 같은 곳을 가리키고 있기 때문에
				mem1d_cnt++;								// ioNode[ioNode_cnt].mem1D[mem1dcnt]=cpy 도 가능 )
			}
		}
	}

}
void reshape1Dto3D(MemBlock32F *Delta){

	int mem1d_cnt = 0;
	for (int m = 0; m < Delta->x; ++m){			// CNN Layer의 마지막 특징 맵 수
		for (int y = 0; y < Delta->y; ++y){		// 특징 맵 크기
			for (int x = 0; x < Delta->z; ++x){
				Delta->mem3D[m][y][x] = Delta->mem1D[mem1d_cnt]; // Delta->mem3D[m][y][x] = FC->output_Delta->mem1D[mem1d_cnt];
				mem1d_cnt++;
			}
		}
	}

}
void reshape1Dto3D_8C(MemBlock8C *Node){

	int mem1d_cnt = 0;
	for (int m = 0; m < Node->x; ++m){			// CNN Layer의 마지막 특징 맵 수
		for (int y = 0; y < Node->y; ++y){		// 특징 맵 크기
			for (int x = 0; x < Node->z; ++x){
				Node->mem3D[m][y][x] = Node->mem1D[mem1d_cnt]; // Delta->mem3D[m][y][x] = FC->output_Delta->mem1D[mem1d_cnt];
				mem1d_cnt++;
			}
		}
	}
}

void copyInputData1D(float input[Image_W*Image_depth][Image_H], MemBlock32F *ioNode){

	int cnt, y, idx;

	if (Image_depth == 1){
		for (cnt = 0, y = 0; y < Image_H; ++y){
			for (int x = 0; x < Image_W; ++x){
				ioNode[0].mem1D[cnt] = input[y][x];
				cnt++;
			}
		}
	}
	else if (Image_depth == 3){
		for (cnt = 0, idx = 0, y = 0; y < Image_H; ++y, idx = 0){
			for (int x = 0; x < Image_W; x++, idx += 3){
				ioNode[0].mem1D[cnt] = input[idx][y];
				ioNode[0].mem1D[cnt + (Image_H * Image_W)] = input[idx + 1][y];
				ioNode[0].mem1D[cnt + (Image_H * Image_W * 2)] = input[idx + 2][y];
				cnt++;
			}
		}
	}

}
void copyInputData3D(float input[Image_W*Image_depth][Image_H], MemBlock32F *ioNode){

	if (Image_depth == 1){
		for (int y = 0; y < Image_H; ++y){
			for (int x = 0; x < Image_W; ++x){
				ioNode[0].mem3D[0][y][x] = input[y][x];
			}
		}
	}
	else if (Image_depth == 3){
		for (int idx = 0, y = 0; y < Image_H; ++y, idx = 0){
			for (int x = 0; x < Image_W; x++, idx += 3){
				ioNode[0].mem3D[0][x][y] = input[idx][y];
				ioNode[0].mem3D[1][x][y] = input[idx + 1][y];
				ioNode[0].mem3D[2][x][y] = input[idx + 2][y];
			}
		}
	}

}
