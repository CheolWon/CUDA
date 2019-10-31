
/////////////////////////////////////////////
// CNN 연산 함수      ///////////////////////
/////////////////////////////////////////////

#ifndef __CNNFUNC_H__
#define __CNNFUNC_H__

#include "Config.h"
#include "utils.h"
#include <math.h>
#include <string.h>

void Softmax_CrossEntropy(float *acc, float *loss, float *fc_target, FCLayer *FC);

void reshape3Dto1D(MemBlock32F *ioNode);
void reshape1Dto3D(MemBlock32F *Delta);
void reshape1Dto3D_8C(MemBlock8C *Node);

void copyInputData1D(float input[Image_W*Image_depth][Image_H], MemBlock32F *ioNode);
void copyInputData3D(float input[Image_W*Image_depth][Image_H], MemBlock32F *ioNode);

#endif

