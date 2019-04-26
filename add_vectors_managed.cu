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
    
    const int len =   50000000;
    //const int len = 500000000;

    int *d_a, *d_b;
    HANDLE_ERROR( cudaMallocManaged((void**) &d_a, len * sizeof(int)) );
    HANDLE_ERROR( cudaMallocManaged((void**) &d_b, len * sizeof(int)) );

    for (int i = 0; i < len; ++i){
        d_a[i] = -i;
        d_b[i] = i * i;
    }

    clock_gettime(CLOCK_MONOTONIC, &old_time);
    add<<<65000, 1024>>>(d_a, d_b, len);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &new_time);

    printf("Allocated %ld MB of GPU memory", 2 * sizeof(int) * len /1024/1024);
    oldNs = old_time.tv_sec * 1000000000ull + old_time.tv_nsec;
    newNs = new_time.tv_sec * 1000000000ull + new_time.tv_nsec;
    float dt = (newNs - oldNs) * 0.000000001f;
    printf("Resulting array size is %d, addition took %0.4f seconds \n", len, dt);

    int num_errors = 0;
    for (int i = 0; i < len; ++i){
        if (d_b[i] != -i + (i*i))
            num_errors += 1;
    }
    printf("Detected %d errors\n", num_errors);

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}