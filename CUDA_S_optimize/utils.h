
//////////////////////////////////////////////
// CNN 기타 함수   ///////////////////////////
//////////////////////////////////////////////

#ifndef __UTILS_H__
#define __UTILS_H__

//#include <opencv2/opencv.hpp>

//using namespace cv;

#include "Config.h"
#include "MemBlock.h"
#include <string.h>

float gaussianRandom(float average, float stdev);
float getMax(float *x, int num);
void predict(ConvLayer *Conv, PoolLayer *Pool, FCLayer *FC);
unsigned char * read_image_pgm(char filename[], int &depth, int &imageHeight, int &imageWidth);
//void read_image_pgm(unsigned char image[], char filename[], int imageWidth, int imageHeight);
void read_image_pgm2(unsigned char image[], char filename[], int imageWidth, int imageHeight);
void read_image_ppm(unsigned char image[], char filename[], int imageWidth, int imageHeight, int depth);
void write_image_pgm(unsigned char image[], const char filename[], int imageWidth, int imageHeight);
void write_image_ppm(unsigned char image[], char filename[], int imageWidth, int imageHeight, int depth);
void readDataset(float data[Image_W*Image_depth][Image_H], int *num_cnt, unsigned int &label, int randVal, char *path, bool flag);
void printConfig();
void printNet(ConvLayer *Conv, PoolLayer *Pool, FCLayer *FC, int *Layerinfo);
void printTrainBar(int num_data, float epoch);
void initMem32F(MemBlock32F *ioNode);
void initMem8C(MemBlock8C *ioNode);
void saveWeight(ConvLayer *Conv, FCLayer *FC);
void loadWeight(ConvLayer *Conv, FCLayer *FC);

#endif
