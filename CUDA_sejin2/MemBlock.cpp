#include "MemBlock.h"


MemBlock32F CreateMemBlock32F(int x, int y, int z){
	MemBlock32F data;

	data.x = x;
	data.y = y;
	data.z = z;
	int total = data.total = x * y * z;

	// 1D Memory
	data.mem1D = new float[total];

	for (int t = 0; t < total; t++){
		data.mem1D[t] = 0;
	}
	
	// 2D Memory ( temp )

	data.mem2D = new float*[y];
	for (int i = 0; i < y; ++i){
		data.mem2D[i] = new float[z];
	}

	for (int t = 0; t < y; t++){
		for (int r = 0; r < z; r++){
			data.mem2D[t][r] = 0;
		}
	}

	// 3D Memory
	if ((data.mem3D = new float**[x]) == NULL)
		cout << "메모리 할당 실패" << endl;
	for (int i = 0; i < x; ++i){
		if ((data.mem3D[i] = new float*[y]) == NULL)
			cout << "메모리 할당 실패" << endl;
		for (int j = 0; j < y; ++j){
			if ((data.mem3D[i][j] = new float[z]) == NULL)
				cout << "메모리 할당 실패" << endl;
		}
	}

	for (int t = 0; t < x; t++){
		for (int r = 0; r < y; r++){
			for (int e = 0; e < z; e++){
				data.mem3D[t][r][e] = 0;
			}
		}
	}

	return data;
}
MemBlock8C CreateMemBlock8C(int x, int y, int z){
	MemBlock8C data;

	data.x = x;
	data.y = y;
	data.z = z;
	int total = data.total = x * y * z;


	// 1D Memory
	data.mem1D = new char[total];

	for (int t = 0; t < total; t++){
		data.mem1D[t] = 0;
	}

	// 3D Memory
	if ((data.mem3D = new char**[x]) == NULL)
		cout << "메모리 할당 실패" << endl;
	for (int i = 0; i < x; ++i){
		if ((data.mem3D[i] = new char*[y]) == NULL)
			cout << "메모리 할당 실패" << endl;
		for (int j = 0; j < y; ++j){
			if ((data.mem3D[i][j] = new char[z]) == NULL)
				cout << "메모리 할당 실패" << endl;
		}
	}

	for (int t = 0; t < x; t++){
		for (int r = 0; r < y; r++){
			for (int e = 0; e < z; e++){
				data.mem3D[t][r][e] = 0;
			}
		}
	}

	return data;
}

void memRelease32F(MemBlock32F *data){
	// 1d release
	delete[]data->mem1D;

	// 2d release
	for (int i = 0; i < data->y; i++){
		delete[]data->mem2D[i];
	}
	delete[]data->mem2D;

	// 3d release
	for (int i = 0; i < data->x; ++i){
		for (int j = 0; j < data->y; ++j){
			delete[]data->mem3D[i][j];
		}
		delete[]data->mem3D[i];
	}
	delete[]data->mem3D;
}
void memRelease8C(MemBlock8C *data){
	// 1d release
	delete[]data->mem1D;

	// 3d release
	for (int i = 0; i < data->x; ++i){
		for (int j = 0; j < data->y; ++j){
			delete[]data->mem3D[i][j];
		}
		delete[]data->mem3D[i];
	}
	delete[]data->mem3D;

	data->mem3D = NULL;
	data->total = data->x = data->y = data->z = 0;
}

void initMem32F(MemBlock32F *ioNode){

	for (int t = 0; t < ioNode->total; t++){
		ioNode->mem1D[t] = 0;
	}
	for (int t = 0; t < ioNode->y; t++){
		for (int r = 0; r < ioNode->z; r++){
			ioNode->mem2D[t][r] = 0;
		}
	}
	for (int t = 0; t < ioNode->x; t++){
		for (int r = 0; r < ioNode->y; r++){
			for (int e = 0; e < ioNode->z; e++){
				ioNode->mem3D[t][r][e] = 0;
			}
		}
	}

}
void initMem8C(MemBlock8C *ioNode){
	for (int t = 0; t < ioNode->x; t++){
		for (int r = 0; r < ioNode->y; r++){
			for (int e = 0; e < ioNode->z; e++){
				ioNode->mem3D[t][r][e] = 0;
			}
		}
	}
}