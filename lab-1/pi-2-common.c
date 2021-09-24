#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>

// long long N;
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
    // if (args < 1){
    //     printf("Expected Arguments");
    //     return -1;
    // }

    // N = atoi(argc[1]);

    // Calculate pi
    pi_compute();
    pi = pi*4;
    printf("pi: %.10LF\n",pi);
}