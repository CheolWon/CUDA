
//////////////////////////////////////////////
// CNN 구조 설정 함수 ////////////////////////
//////////////////////////////////////////////

#ifndef __CONFIG_H__
#define __CONFIG_H__

//#include <opencv2/opencv.hpp>

#include "MemBlock.h"
#include "Define.h"
#include <math.h>

//extern cv::Mat img;

struct ConvLayer{
	MemBlock32F *Input_data;
	MemBlock32F *Output_data;
	MemBlock32F *kernel;

	MemBlock32F *ConvPad;

	MemBlock32F *input_Delta;
	MemBlock32F *output_Delta;
	MemBlock32F *deltaPad;
	MemBlock32F *Moment;
	MemBlock32F *gradSum;
	MemBlock32F *mean_prev;
	MemBlock32F *var_prev;

	int kernel_num;
	int Input_size;
	int Output_size;
	int Inmap;
	int Outmap;
};
struct PoolLayer{
	MemBlock32F *Input_data;
	MemBlock32F *Output_data;
	MemBlock8C	*pool_mark;		// pooling layer의 인덱스 저장

	MemBlock32F *input_Delta;
	MemBlock32F *output_Delta;

	int Input_size;
	int Output_size;
	int Inmap;
	int Outmap;
};
struct FCLayer{
	MemBlock32F *Input_data;
	MemBlock32F *Output_data;
	MemBlock32F	*Weight;

	MemBlock32F *input_Delta;
	MemBlock32F *output_Delta;
	MemBlock32F *Moment;
	MemBlock32F *gradSum;
	MemBlock32F *mean_prev;
	MemBlock32F *var_prev;

	int Input_size;
	int Output_size;
};

ConvLayer Conv_Info(MemBlock32F *input, MemBlock32F *output, MemBlock32F *kernel, MemBlock32F *input_Delta, MemBlock32F *output_Delta, MemBlock32F *deltaPad, MemBlock32F *Moment, MemBlock32F *gradSum, MemBlock32F *mean_prev, MemBlock32F *var_prev, MemBlock32F *ConvPad);
PoolLayer Pool_Info(MemBlock32F *input, MemBlock32F *output, MemBlock8C *pool_mark, MemBlock32F *input_Delta, MemBlock32F *output_Delta);
FCLayer	  FC_Info(MemBlock32F *input, MemBlock32F *output, MemBlock32F *Weight, MemBlock32F *input_delta, MemBlock32F *output_delta, MemBlock32F *Moment, MemBlock32F *gradSum, MemBlock32F *mean_prev, MemBlock32F *var_prev);

void setLayer(int input_size, int depth, int *map, int *kernel_size, int *ConvPad_info, int *Layerinfo, MemBlock32F *prevPatch,
	MemBlock32F *kernel, MemBlock32F *ConvPad, MemBlock8C *pool_mark, MemBlock32F *deltaPad, MemBlock32F *CNN_Delta, MemBlock32F *CNN_Moment, MemBlock32F *CNN_gradSum, MemBlock32F *CNN_mean_prev, MemBlock32F *CNN_var_prev, MemBlock32F *ioNode, ConvLayer *Conv, PoolLayer *Pool,
	MemBlock32F *FC_Weight, MemBlock32F *FC_Moment, MemBlock32F *FC_gradSum, MemBlock32F *FC_mean_prev, MemBlock32F *FC_var_prev, FCLayer *FC);


#endif

