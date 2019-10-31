#include "utils.h"
#include "Define.h"
#include <stdio.h>

void readDataset(float data[Image_W*Image_depth][Image_H], int *num_cnt, unsigned int &label, int randVal, char *path, bool flag){
	unsigned char train_input[Image_W*Image_depth][Image_H];
	unsigned char test_input[Image_W*Image_depth][Image_H];
	
#if MNIST
	strcat(path, ".pgm");
#endif
#if CIFAR10
	strcat(path, ".ppm");
#endif
	label = randVal;

#if MNIST
	//read_image_pgm(tmp_input, path, Image_H, Image_W);		// 28x28x1
	unsigned char* inImage;
	int depth = 0;
	int imageHeight = 0;
	int imageWidth = 0;

	inImage = read_image_pgm(path, depth, imageHeight, imageWidth);

	for (int x = 0; x < Image_H; x++){
		for (int y = 0; y < Image_W; y++){
			if (flag){
				train_input[x][y] = inImage[(x*Image_H) + y];
				data[x][y] = ((float)train_input[x][y] / 255); // /255
			}
			else{
				test_input[x][y] = inImage[(x*Image_H) + y];
				data[x][y] = ((float)test_input[x][y] / 255);
			}
		}
	}
#endif

}
unsigned char * read_image_pgm(char filename[], int &depth, int &imageHeight, int &imageWidth)
{   /************************************************************************************
	* Function: void read_image_pgm(unsigned char image[], char filename[], int imageWidth, int imageHeight)
	* Input   : uchar array pointer for output result, char array with filename, int with with, int with height
	* Output  : uchar image array
	* Procedure: if image dimensions and layout pgm correct image is read from file to image array
	************************************************************************************/
	int PGM_HEADER_LINES = 3;
	FILE* input;
	const int LINE_BUFFER_SIZE = 300;

	int headerLines = 1;
	int scannedLines = 0;
	long int counter = 0;

	//read header strings
	char *lineBuffer = (char *)malloc(LINE_BUFFER_SIZE + 1);
	char *split;
	char *format = (char *)malloc(LINE_BUFFER_SIZE + 1);
	char P5[] = "P5";
	char comments[LINE_BUFFER_SIZE + 1];
	char * context = NULL;
	//open the input PGM file
	input = fopen(filename, "rb");
	if (!input) {
		printf("fopen fail\n");
	}

	//read the input PGM file header 
	while (scannedLines < headerLines) {
		fgets(lineBuffer, LINE_BUFFER_SIZE, input);
		//if not comments
		if (lineBuffer[0] != '#') {
			scannedLines += 1;
			//read the format
			if (scannedLines == 1) {
				split = strtok_r(lineBuffer, " \n", &context);
				strcpy(format, split);
				if (strcmp(format, P5) == 0) {
					//printf("FORMAT: %s\n",format);
					headerLines = PGM_HEADER_LINES;

					depth = 1;
				}
				else
				{
					depth = 3;
				}
			}
			//read width and height
			if (scannedLines == 2)
			{
				split = strtok_r(lineBuffer, " \n", &context);
				imageWidth = atoi(split); //check if width matches description

				split = strtok_r(NULL, " \n", &context);
				imageHeight = atoi(split); //check if heigth matches description
			}
			// read maximum gray value
			if (scannedLines == 3)
			{
				split = strtok_r(lineBuffer, " \n", &context);
				//printf("GRAYMAX: %d\n", grayMax);
			}
		}
		else
		{
			strcpy(comments, lineBuffer);
			//printf("comments: %s", comments);
		}
	}



	// 동적 할당 
	static unsigned char *temp = NULL;
	if (temp == NULL)
		temp = (unsigned char *)malloc(imageWidth * imageHeight * depth);

	// 0으로 초기화
	memset(temp, 0, sizeof(imageWidth * imageHeight * depth));


	// temp로 파일 읽어오기
	counter = fread(temp, sizeof(unsigned char), imageWidth * imageHeight * depth, input);


	//close the input pgm file and free line buffer
	fclose(input);
	free(lineBuffer);
	free(format);

	return temp;
}

float getMax(float *x, int num){
	int i;
	float index;
	float max;

	max = *x;
	index = 0;

	for (i = 1; i < num; i++){
		if (max < *(x + i)){
			max = *(x + i);
			index = i;
		}
	}
	return index;
}

void loadWeight(ConvLayer *Conv, FCLayer *FC){
	FILE *fp1 = NULL, *fp2 = NULL;

	fp1 = fopen("./saveKernel.txt", "r");
	fp2 = fopen("./saveWeight.txt", "r");

	if (fp1 == NULL || fp2 == NULL)
		printf(" 파일을 읽지 못했습니다. \n");
	else
		printf("\n # loading weights.. ");

	//////////////////////////////////////////////////////////
	// Conv Layer 커널 읽기
	printf("load kernel\n");
	for (int i = 0; i < nConvLayer; i++){
		for (int m = 0; m < Conv[i].kernel->x; m++){
			for (int h = 0; h < Conv[i].kernel->y; h++){
				for (int w = 0; w < Conv[i].kernel->z; w++){
					fscanf(fp1, "%f", &Conv[i].kernel->mem3D[m][h][w]);
					//printf("%f ", &Conv[i].kernel->mem3D[m][h][w]);
				}
				//printf("\n");
			}
			//printf("\n");
		}
	}

	//////////////////////////////////////////////////////////
	// FC Layer 가중치 읽기
#if !CUDA
	// 2D array
	for (int i = 0; i < nFCLayer; i++){
		for (int j = 0; j < FC[i].Weight->y; j++){
			for (int k = 0; k < FC[i].Weight->z; k++){
				fscanf(fp2, "%f", &FC[i].Weight->mem2D[j][k]);
				//	printf("%f \n", FC[i].Weight->mem2D[j][k]);
			}
		}
	}
#endif
#if CUDA
	// 1D array
	//printf("load weight\n");
	for (int n = 0; n < nFCLayer; n++){
		for (int j = 0; j < FC[n].Weight->total; j++){
			fscanf(fp2, "%f", &FC[n].Weight->mem1D[j]);
			//printf("%f ", &FC[n].Weight->mem1D[j]);
		}
		//printf("\n");
	}
#endif
	fclose(fp1);
	fclose(fp2);
}
