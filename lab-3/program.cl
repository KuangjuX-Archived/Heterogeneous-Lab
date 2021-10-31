__kernel void KERNEL_CPUS_BRUTE(
    __global const float* init,
    __global float* temp,
    __global float* result
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    for(int i = 1; i < 2048; i++) {
        float r = 0;
        for(int j = 0; j < 64; j++) {
            r += temp[x * 64 + j] * init[y + j * 64];
        }
        result[x * 64 + y] = r;
        barrier(CLK_GLOBAL_MEM_FENCE);
        temp[x * 64 + y] = r;
        barrier(CLK_GLOBAL_MEM_FENCE);
    } 
}


__kernel void KERNEL_GPU_BRUTE(
    __global const float* init,
    __global float* temp,
    __global float* result
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    for(int i = 1; i < 2048; i++) {
        float r = 0;
        for(int j = 0; j < 64; j++) {
            r += temp[x * 64 + j] * init[y + j * 64];
        }
        result[x * 64 + y] = r;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        temp[x * 64 + y] = r;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}



__kernel void KERNEL_CPUS_EFFICIENT(
    __global const float* init,
    __global float* temp,
    __global float* result
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    for(int i = 1; i < 2048; i *= 2) {
        float r = 0;
        for(int j = 0; j < 64; j++) {
            r += temp[x * 64 + j] * temp[y + j * 64];
        }
        result[x * 64 + y] = r;
        barrier(CLK_GLOBAL_MEM_FENCE);
        temp[x * 64 + y] = r;
        barrier(CLK_GLOBAL_MEM_FENCE);
    } 
}



__kernel void KERNEL_GPU_EFFICIENT(
    __global const float* init,
    __global float* temp,
    __global float* result
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    for(int i = 1; i < 2048; i *= 2) {
        float r = 0;
        for(int j = 0; j < 64; j++) {
            r += temp[x * 64 + j] * temp[y + j * 64];
        }
        result[x * 64 + y] = r;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        temp[x * 64 + y] = r;
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
