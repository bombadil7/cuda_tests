#include "book.h"

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
    float *dev_a, *dev_b, *dev_partial_c;
    float c;

    HANDLE_ERROR( cudaMallocManaged((void**) &dev_a, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged((void**) &dev_b, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged((void**) &dev_partial_c, blocksPerGrid * sizeof(float)) );

    fprintf(stderr, "Allocated device memory\n");

    for(int i = 0; i < N; ++i){
        dev_a[i] = i;
        dev_b[i] = i * 2;
    }

    fprintf(stderr, "Initialized arrays\n");

    dot <<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_partial_c);
    cudaDeviceSynchronize();

    fprintf(stderr, "Calculated partial array\n");

    c = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        c += dev_partial_c[i];
    }
    fprintf(stderr, "Accumulated final number %g\n", c);

    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float) (N - 1) ));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    return 0;
}