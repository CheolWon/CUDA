
//////////////////////////////////////////////
// Define 값 설정     ////////////////////////
//////////////////////////////////////////////

#ifndef __DEFINE_H__
#define __DEFINE_H__

//#define		LINE_BUFFER_SIZE	100		//Line buffer size for read write 

//#define		Batch			128		// 64
#define		Batch			1		// 64
#define		iter			500		// 10
#define		learningRate	0.01f	// ReLU 0.001 , Sigmoid 0.1
#define		WeightDecay		0.0005	// 0.0005	
#define		NUM_OUTPUTS		10		// 최종 출력 뉴런 수 
#define		NUM_TRAIN_DATA	55000	// Cifar10;50000 . MNIST;55000
#define		NUM_TEST_DATA	10000   // 10000	
#define		NUM_VALID_DATA	5000

#define		_NUM_TRAIN_DATA	NUM_TRAIN_DATA/10		// 레이블당 학습 데이터
#define		_NUM_TEST_DATA	NUM_TEST_DATA/10		
#define		_NUM_VALID_DATA	NUM_VALID_DATA/10		

#define		CUDA			1 
#define		MAX_THREAD		1024
 
// 테스트용..
#define		ConvTest		0
#define		PoolTest		0

// padding
#define		ConvPadding		0

// Training, Validation, Test
#define		Training		1
#define		Validation		0
#define		Test			1
#define		showImg			0
#define		ValidEpoch		2		// 검증 주기

// Loss function
#define		MSE				0
#define		CrossEntropy	1	

// Dropout
#define		Dropout			0
#define		drp_rate		0.5

// Activation function
#define		CnP_Sigmoid		0
#define		CnP_ReLU		1
#define		FC_Sigmoid		0
#define		FC_ReLU			1

// CNN Layer 
#define		nCnPLayer		4	// = Conv layer + Pool layer
#define		nConvLayer		2	// Conv layer 
#define		nPoolLayer		2	// Pool layer
#define		nFCLayer		2	// Fully connected layer

// Dataset
#define		MNIST			1
#define		CIFAR10			0

// Input data
#define		Image_H			28
#define		Image_W			28
#define		Image_depth		1

// Optimizer
#define		MomentumOpt		0
#define		Momentum		0.9f	// 0.9f

#define		AdamOpt			1
#define		beta1			0.9f
#define		beta2			0.999f
#define		epsilon			1e-8		// 1e-8
//#define		gamma			1 - 1e-8	// 1 - 1e-8

#endif