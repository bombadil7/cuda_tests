#include "book.h"
#include <time.h>

#define imin(a, b) (a < b ? a : b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N){
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0){
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}


int main(void) {
    struct timespec old_time, new_time;
    unsigned long int oldNs, newNs; 

    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = (float*) malloc(N * sizeof(float));
    b = (float*) malloc(N * sizeof(float));
    partial_c = (float*) malloc(blocksPerGrid * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**) &dev_a, N * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**) &dev_b, N * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**) &dev_partial_c, blocksPerGrid * sizeof(float)) );

    for(int i = 0; i < N; ++i){
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice) );

    clock_gettime(CLOCK_MONOTONIC, &old_time);
    dot <<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_partial_c);
    cudaDeviceSynchronize();
    HANDLE_ERROR( cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost) );

    clock_gettime(CLOCK_MONOTONIC, &new_time);
    oldNs = old_time.tv_sec * 1000000000ull + old_time.tv_nsec;
    newNs = new_time.tv_sec * 1000000000ull + new_time.tv_nsec;
    float dt = (newNs - oldNs) * 0.000000001f;
    printf("Original vector sizes were %d, dot product took %0.6f seconds \n", N, dt);

    //HANDLE_ERROR( cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost) );

    c = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        c += partial_c[i];
    }

    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float) (N - 1) ));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    free(a);
    free(b);
    free(partial_c);

    return 0;
}