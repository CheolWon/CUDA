
//////////////////////////////////////////////
// 메모리 할당 / 해제 함수   /////////////////
//////////////////////////////////////////////

#ifndef __MEMBLOCK_H__
#define __MEMBLOCK_H__

#include <iostream>

using namespace std;

struct MemBlock32F{
	float *mem1D;
	float **mem2D;
	float ***mem3D;
	int x, y, z;
	int total;
};
struct MemBlock8C{
	char *mem1D;
	//char **mem2D;
	char ***mem3D;
	int x, y, z;
	int total;
};

MemBlock32F CreateMemBlock32F(int x, int y, int z);
MemBlock8C CreateMemBlock8C(int x, int y, int z);

void memRelease32F(MemBlock32F *data);
void memRelease8C(MemBlock8C *data);

void initMem32F(MemBlock32F *ioNode);
void initMem8C(MemBlock8C *ioNode);

#endif