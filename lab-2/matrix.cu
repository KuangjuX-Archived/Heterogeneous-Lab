#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// using namespace std;

#define M 64
#define N 2048

__constant__ const int d_M = 32;
__constant__ const int d_N = 32;
__constant__ float* d_matrix;
__device__ __managed__ float* result;

void CPUMatrixPowerBrute() {
	clock_t start = clock();
	float h_matrix[M][M];
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			h_matrix[i][j] = (float)rand() / RAND_MAX / (float)M;
		}
	}
	float h_matrix_result[M][M];
	float h_matrix_temp[M][M];
	memcpy(h_matrix_temp, h_matrix, M * M * sizeof(float));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M; k++) {
				float value = 0.0;
				for (int l = 0; l < M; l++) {
					value += (h_matrix_temp[j][l] * h_matrix[l][k]);
				}
				h_matrix_result[j][k] = value;
			}
		}
		memcpy(h_matrix_temp, h_matrix_result, M * M * sizeof(float));
	}
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("[串行暴力算法] M: %d, N: %d, time: %.10lfs\n", M, N, time);
}

void CPUMatrixPowerEfficient() {
	clock_t start = clock();
	float h_matrix[M][M];
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			h_matrix[i][j] = (float)rand() / RAND_MAX / (float)M;
		}
	}
	float h_matrix_result[M][M];
	float h_matrix_temp[M][M];
	memcpy(h_matrix_temp, h_matrix, M * M * sizeof(float));
	for (int i = 1; i < N; i *= 2) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < M; k++) {
				float value = 0.0;
				for (int l = 0; l < M; l++) {
					value += (h_matrix_temp[j][l] * h_matrix_temp[l][k]);
				}
				h_matrix_result[j][k] = value;
			}
		}
		memcpy(h_matrix_temp, h_matrix_result, M * M * sizeof(float));
	}
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("[串行高效算法] M: %d, N: %d, time: %.10lfs\n", M, N, time);
}

__global__ void GPUMatrixPowerBruteKernel(float* d_result) {
	__shared__ float matrixInit[d_M][d_M];
	__shared__ float matrixTemp[d_M][d_M];
	__shared__ float matrixResult[d_M][d_M];
	int index_x = threadIdx.x;
	int index_y = threadIdx.y;
	matrixInit[index_x][index_y] = *(d_matrix + index_x * blockDim.y + index_y);
	matrixTemp[index_x][index_y] = matrixInit[index_x][index_y];
	__syncthreads();
	for (int i = 1; i < d_N; i++)
	{
		float tempResult = 0.0;
		for(int j = 0; j < d_M; j++)
		{
			tempResult += matrixTemp[index_x][j] * matrixTemp[j][index_y];
		}
		matrixResult[index_x][index_y] = tempResult;
		__syncthreads();
		matrixTemp[index_x][index_y] = matrixResult[index_x][index_y];
		__syncthreads();
	}
	*(d_result + index_x * blockDim.y + index_y) = matrixResult[index_x][index_y];
}

__global__ void GPUMatrixPowerEfficientKernel(float* d_result) {
	__shared__ float matrixTemp[d_M][d_M];
	__shared__ float matrixResult[d_M][d_M];
	int index_x = threadIdx.x;
	int index_y = threadIdx.y;
	matrixTemp[index_x][index_y] = *(d_matrix + index_x * d_M + index_y);
	__syncthreads();
	for(int c = 1; c < d_N; c *= 2)
	{
		float result = 0.0;
		for (int i = 0; i < d_M; i++)
		{
			result += matrixTemp[index_x][i] * matrixTemp[i][index_y];
		}
		matrixResult[index_x][index_y] = result;
		__syncthreads();
		matrixTemp[index_x][index_y] = matrixResult[index_x][index_y];		
		__syncthreads();
	}
	*(d_result + index_x * blockDim.y + index_y) = matrixResult[index_x][index_y];
}

void GPUMatrixPowerBrute()
{
	clock_t start = clock();
	int h_M = M;
	int h_N = N;
	float* h_matrix;
	cudaMallocHost(&h_matrix, M * M * sizeof(float));
	cudaMallocManaged(&result, M * M * sizeof(float));

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			*(h_matrix + (M * i + j)) = rand() / (float)M;
		}
	}
	cudaMemcpyToSymbol(d_matrix, h_matrix, M * M * sizeof(float));
	cudaMemcpyToSymbol(&d_M, &h_M, sizeof(int));
	cudaMemcpyToSymbol(&d_N, &h_N, sizeof(int));

	dim3 threads(M, M);
	GPUMatrixPowerBruteKernel << <1, threads >> > (result);
	cudaDeviceSynchronize();
	// 输出结果
	cudaFreeHost(h_matrix);
	cudaFree(result);
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("[并行暴力算法] M: %d, N: %d, time: %.10lfs\n", M, N, time);
}

void GPUMatrixPowerEfficient() {
	clock_t start = clock();	
	int h_M = M;
	int h_N = N;
	float* h_result;
	float* d_result;
	float* h_matrix;
	size_t pitch;
	cudaMallocHost(&h_result, M * M * sizeof(float));
	cudaMallocPitch(&d_result, &pitch, M * sizeof(float), M);
	cudaMallocHost(&h_matrix, M * M * sizeof(float));
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			*(h_matrix + i * M + j) = rand() / (float)M;  
		}
	}
	cudaMemcpyToSymbol(&d_M, &h_M, sizeof(float));
	cudaMemcpyToSymbol(&d_N, &h_N, sizeof(float));
	cudaMemcpyToSymbol(d_matrix, h_matrix, M * M * sizeof(float));
	cudaMemcpy2D(d_result, pitch, h_result, pitch, M * sizeof(float), M, cudaMemcpyHostToDevice);
	dim3 threads(M, M);
	GPUMatrixPowerEfficientKernel<<<1, threads>>>(d_result);
	cudaDeviceSynchronize();
	cudaMemcpy2D(h_result, pitch, d_result, pitch, M * sizeof(float), M, cudaMemcpyDeviceToHost);
	cudaFreeHost(h_matrix);
	cudaFreeHost(h_result);
	cudaFree(d_result);
	clock_t end = clock();
	double time = (double)(end - start) / CLOCKS_PER_SEC;
	printf("[并行高效算法] M: %d, N: %d, time: %.10lfs\n", M, N, time);
}

int main() {
	srand(time(NULL));
	for(int i = 0; i < 5; i++) {
		CPUMatrixPowerBrute();
		CPUMatrixPowerEfficient();
		GPUMatrixPowerBrute();
		GPUMatrixPowerEfficient();
		printf("----------------------\n");
	}
	return 0;
}