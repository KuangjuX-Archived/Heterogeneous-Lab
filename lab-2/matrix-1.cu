#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_NUM 32
#define THREAD_NUM 256
#define R_SIZE (BLOCK_NUM * THREAD_NUM)
#define M_SIZE (R_SIZE * R_SIZE)

const long double TimeConvert = 1e6;

__global__ void matrix_mul(int *mat1, int *mat2, int *res) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    printf("bid: %d, tid: %d\n", bid, tid);

    // 每个线程计算一行
    const int raw = bid * THREAD_NUM + tid;
    for(int i = 0; i < R_SIZE; i++) {
        for(int j = 0; j < R_SIZE; j++) {
            res[raw * R_SIZE + i] += mat1[raw * R_SIZE + j] * mat2[raw * R_SIZE + i];
            printf("sum += %d + %d\n", mat1[raw * R_SIZE + j], mat2[raw * R_SIZE + i]);
        }
    }
}

int main() {
    int *mat1, *mat2, *res;
    int *g_mat1, *g_mat2, *g_res;

    mat1 = (int*)malloc(M_SIZE * sizeof(int));
    mat2 = (int*)malloc(M_SIZE * sizeof(int));
    res = (int*)malloc(M_SIZE * sizeof(int));

    // 初始化矩阵
    for(int i = 0; i < M_SIZE; i++) {
        mat1[i] = i;
        mat2[i] = i;
        res[i] = 0;
    }

    // 在 GPU 为矩阵分配内存
    cudaMalloc((void **)&g_mat1, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_mat2, sizeof(int) * M_SIZE);
    cudaMalloc((void **)&g_res, sizeof(int) * M_SIZE);

    // 获取 GPU 执行前的时间
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // 将 CPU 内存拷贝到 GPU 中
    cudaMemcpy(g_mat1, mat1, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(g_mat2, mat2, sizeof(int) * M_SIZE, cudaMemcpyHostToDevice);
    // 执行 GPU 计算函数
    matrix_mul<<<BLOCK_NUM,THREAD_NUM>>>(g_mat1, g_mat2, g_res);

    // 将结果从 GPU 拷贝回 CPU
    cudaMemcpy(res, g_res, sizeof(int) * M_SIZE, cudaMemcpyDeviceToHost);

    // 执行后获取程序执行时间
    gettimeofday(&end_time, NULL);
    long long int start, end;
    start = start_time.tv_sec * TimeConvert + start_time.tv_usec;
    end = end_time.tv_sec * TimeConvert + end_time.tv_usec;
    long double span_time;
    span_time = (end - start) / TimeConvert;
    printf("花费时间: %.6LFs.\n", span_time);

    // 释放 GPU 内存
    cudaFree(g_res);
    cudaFree(g_mat1);
    cudaFree(g_mat2);
    // 释放 CPU 内存
    free(res);
    free(mat1);
    free(mat2);

    printf("退出.\n");

    return 0;
}