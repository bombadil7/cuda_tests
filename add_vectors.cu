#include <time.h>
#include <cuda.h>
#include "book.h"

void init_vector(int* v, int size)  {
    for (int i = 0; i < size; ++i)
        v[i] = i;
}

__global__ void add(int *a, int *b, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < size) {
        b[tid] += a[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main(void){
    struct timespec old_time, new_time;
    unsigned long int oldNs, newNs; 
    
    const int len =   310000000;
    //const int len = 500000000;

    int* a = (int*) malloc(len * sizeof(int));
    int* b = (int*) malloc(len * sizeof(int));
    int* c = (int*) malloc(len * sizeof(int));

    int *d_a, *d_b;
    HANDLE_ERROR( cudaMalloc((void**) &d_a, len * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**) &d_b, len * sizeof(int)) );
 //   HANDLE_ERROR( cudaMalloc((void**) &d_c, len * sizeof(int)) );


    //init_vector(a, len);
    //init_vector(b, len);
    for (int i = 0; i < len; ++i){
        a[i] = -i;
        b[i] = i * i;
    }

    HANDLE_ERROR(
        cudaMemcpy(d_a, a, len * sizeof(int), 
            cudaMemcpyHostToDevice));
    HANDLE_ERROR(
        cudaMemcpy(d_b, b, len * sizeof(int),
            cudaMemcpyHostToDevice));

    free(a);
    free(b);

    clock_gettime(CLOCK_MONOTONIC, &old_time);
    //add<<<65000, 1000>>>(d_a, d_b, len);
    add<<<65031, 1024>>>(d_a, d_b, len);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &new_time);

    HANDLE_ERROR(
        cudaMemcpy(c, d_b, len * sizeof(int),
            cudaMemcpyDeviceToHost));

    printf("Allocated %ld MB of GPU memory", 2 * sizeof(int) * len /1024/1024);
    oldNs = old_time.tv_sec * 1000000000ull + old_time.tv_nsec;
    newNs = new_time.tv_sec * 1000000000ull + new_time.tv_nsec;
    float dt = (newNs - oldNs) * 0.000000001f;
    printf("Resulting array size is %d, addition took %0.4f seconds \n", len, dt);

    int num_errors = 0;
    for (int i = 0; i < len; ++i){
        if (c[i] != -i + (i*i))
            num_errors += 1;
    }
    printf("Detected %d errors\n", num_errors);

    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
  //  cudaFree(d_c);
    return 0;
}