#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// long long int n;
const long double TimeConvert = 1e6;
long double pi;

#define BLOCK_NUM 32   //块数量
#define THREAD_NUM 256 // 每个块中的线程数
#define N ((long long)(BLOCK_NUM * THREAD_NUM * 100000))


void pi_compute(){
    int start,end;
    start = 0;
    end = N;
    
    for(int i = start+1; i <= end; i++){
        pi += (4/((1+((i+0.5)/N)*((i+0.5)/N))))*(1.0/N);
    }
}

int main(int args, char* argc[]){
    pi=0;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    pi_compute();

    gettimeofday(&end_time, NULL);
    long long int start, end;
    start = start_time.tv_sec * TimeConvert + start_time.tv_usec;
    end = end_time.tv_sec * TimeConvert + end_time.tv_usec;
    long double span_time;
    span_time = (end - start) / TimeConvert;
    printf("花费时间: %.6LFs.\n", span_time);

    printf("计算的 pi 值为: %.10LF\n",pi);
    
}