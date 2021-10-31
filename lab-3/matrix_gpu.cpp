#include<iostream>
#include<CL/cl.h>
#include<ctime>
#include<cstdio>
#include<cstdlib>
#include<fstream>
#include<cstring>
#include<sstream>

using namespace std;

const int M = 32;
const int N = 1024;

#define GPU_BRUTE 0;
#define GPU_EFFICIENT 1;

string kernels[2] = { "KERNEL_GPU_BRUTE", "KERNEL_GPU_EFFICIENT" };
void OCLMatrixPower(FILE*, int);

int main(int argc, char** argv){
    srand(time(NULL));
    FILE* fp;
	fp = fopen("./data", "a+");
    for(int i = 0; i < 5; i++){
        OCLMatrixPower(fp, 0);
        OCLMatrixPower(fp, 1);
    }    
	fprintf(fp, "\n");
	fclose(fp);
	return 0;
}

/**
 * @brief Create a Command Queue
 * 
 * @param type 计算类型
 * @param device 设备列表
 * @param numDevices 设备数量
 * @param context 上下文
 * @return cl_command_queue 创建的命令队列
 */
cl_command_queue CreateCommandQueue(int type, cl_device_id *device, cl_uint *numDevices, cl_context *context){
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id platformId;
    cl_command_queue commandQueue;

    errNum = clGetPlatformIDs(1, &platformId, &numPlatforms);
    if(errNum != CL_SUCCESS || numPlatforms <= 0){
        cerr << "Error getting platform IDs.\n";
        return NULL;
    }

    clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, device, NULL);

    *context = clCreateContext(NULL, 1, device, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS){
        cerr << "Error creating context.\n";
        return NULL;
    }

    commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    if(commandQueue == NULL){
        cerr << "Error creating command queue.\n";
        return NULL;
    }

    return commandQueue;
}

/**
 * @brief Create a Program
 * 
 * @param context 上下文
 * @param deviceId 设备列表
 * @param fileName 文件名
 * @return cl_program 创建的程序
 */
cl_program CreateProgram(cl_context *context, cl_device_id *device, const char *fileName){
    cl_int errNum = CL_SUCCESS;
    cl_program program;

    ifstream kernelFile(fileName, ios::in);
    ostringstream oss;
    oss << kernelFile.rdbuf();

    string srcStdStr = oss.str();
    const char * srcStr = srcStdStr.c_str();

    program = clCreateProgramWithSource(*context, 1, (const char **)&srcStr, NULL, NULL);
    if(program == NULL){
        cerr << "Error creating program.\n";
        return NULL;
    }

    errNum = clBuildProgram(program, 1, device, NULL, NULL, NULL);
    if(errNum != CL_SUCCESS){
        cerr << "Error building program.\n";
        return NULL;
    }

    return program;
}

bool CreateMemObjects(cl_context context, cl_mem memObjects[3], void* h_matrix){
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * M, h_matrix, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * M, h_matrix, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * M * M, h_matrix, NULL);

    if(memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL){
        cerr << "Error creating buffer.\n";
        return false;
    }

    return true;
}

/**
 * @brief Release some resources
 * 
 * @param context 上下文
 * @param commandQueue 命令队列
 * @param program 程序
 * @param kernel 内核
 * @param memObjects 内存对象
 */
void Cleanup(cl_context context, cl_command_queue commandQueue,
            cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}


void OCLMatrixPower(FILE* fp, int type){
    clock_t start = clock();
    cl_device_id device = 0;
    cl_uint numDevices = 0;
    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_kernel kernel = 0;
    cl_mem memObjects[3] = { 0, 0, 0 }; // init, temp, result
    cl_int errNum;

    float h_matrix[M][M];
    float result[M][M];
    for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			h_matrix[i][j] = rand() / (float)M;

    //TODO:DEVICE修改
    commandQueue = CreateCommandQueue(type, &device, &numDevices, &context);
    program = CreateProgram(&context, &device, "program.cl");
    kernel = clCreateKernel(program, kernels[type].c_str(), NULL);
    if(!CreateMemObjects(context, memObjects, h_matrix)){
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return;
    }
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS){
        cerr << "Error setting kernel arguments.\n";
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return;
    }

    size_t globalSize[2] = {(size_t)M, (size_t)M};
    size_t localSize[2] = {(size_t)M, (size_t)M};

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS){
        cerr << "Error enqueuing kernel.\n";
        return;
    }

    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0, sizeof(float) * M * M, result, 0, NULL, NULL);
    if(errNum != CL_SUCCESS){
        cerr << "Error reading buffer.\n";
        return;
    }

    Cleanup(context, commandQueue, program, kernel, memObjects);

    clock_t end = clock();
    double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
    fprintf(fp, "%s: M: %d, N: %d, NUMDEVICES: %d, time: %.10lf\n", kernels[type].c_str(), M, N, 1, elapsed);
    return;
}
