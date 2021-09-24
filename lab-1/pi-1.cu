#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>

const long double TimeConvert = 1e6;
#define BLOCK_NUM 32   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define LOOP_N ((unsigned long long)(BLOCK_NUM * THREAD_NUM * 100000))

__global__ void integral_pi(double *g_sum) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    double tmp = 0;

    int idx = bid * THREAD_NUM + tid;
    int start = idx * 100000 + 1;
    int end = start + 100000;
    for(int i = start; i < end; i++) {
        tmp += 4. / (1 + ((i + 0.5) / LOOP_N) * ((i + 0.5) / LOOP_N));
    }
    g_sum[bid * THREAD_NUM + tid] = tmp;
}   

int main() {
    double *h_sum, *g_sum;
    double pi_v = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    // allocate host memory
    h_sum = (double*) malloc(sizeof(double) * BLOCK_NUM * THREAD_NUM);

    // Allocate device memory
    cudaMalloc((void **)&g_sum, sizeof(double) * BLOCK_NUM * THREAD_NUM);

    // Execute kernels
    integral_pi<<<BLOCK_NUM,THREAD_NUM>>>(g_sum);

    // Transfer output from device memory to host
    cudaMemcpy(h_sum, g_sum, sizeof(double)*BLOCK_NUM*THREAD_NUM, cudaMemcpyDeviceToHost);

    for (int i = 0; i < BLOCK_NUM * THREAD_NUM; i++) {
        pi_v += h_sum[i];
    }

    gettimeofday(&end_time, NULL);
    long long int start, end;
    start = start_time.tv_sec * TimeConvert + start_time.tv_usec;
    end = end_time.tv_sec * TimeConvert + end_time.tv_usec;
    long double span_time;
    span_time = (end - start) / TimeConvert;
    printf("花费时间: %.6LFs.\n", span_time);

    printf("计算的 pi 值为： %.10f\n", (pi_v/LOOP_N));

    cudaFree(g_sum);
    free(h_sum);

}