#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <sys/time.h>

// long long N;
const long double TimeConvert = 1e6;
long double pi;

#define BLOCK_NUM 32   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define N ((long long)(BLOCK_NUM * THREAD_NUM * 100000))


void pi_compute(){
   for (int i = 1; i <= N; i++){
        int bit = i%2?1:-1;
        pi += bit*(1.0/(2*i-1));
    }
}

int main(){
    pi = 0;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    pi_compute();
    pi = pi*4;

    gettimeofday(&end_time, NULL);
    long long int start, end;
    start = start_time.tv_sec * TimeConvert + start_time.tv_usec;
    end = end_time.tv_sec * TimeConvert + end_time.tv_usec;
    long double span_time;
    span_time = (end - start) / TimeConvert;
    printf("花费时间: %.6LFs.\n", span_time);

    printf("计算的 pi 值为: %.10LF\n",pi);
}