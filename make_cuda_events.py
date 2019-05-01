import subprocess
import sys

def gen_file(f_name='test.c', arsize=3300, nthrd=256):
    prog = r"""
#include "book.h"

#define imin(a, b) (a < b ? a : b)

const int N = %d * 1024;
const int threadsPerBlock = %d;
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
    cudaEvent_t start, stop;
    float *dev_a, *dev_b, *dev_partial_c;
    float c;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );

    HANDLE_ERROR( cudaMallocManaged((void**) &dev_a, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged((void**) &dev_b, N * sizeof(float)) );
    HANDLE_ERROR( cudaMallocManaged((void**) &dev_partial_c, blocksPerGrid * sizeof(float)) );

    for(int i = 0; i < N; ++i){
        dev_a[i] = i;
        dev_b[i] = i * 2;
    }

    dot <<<blocksPerGrid, threadsPerBlock>>> (dev_a, dev_b, dev_partial_c);
    HANDLE_ERROR( cudaEventRecord(stop, 0));
    cudaDeviceSynchronize();

    c = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        c += dev_partial_c[i];
    }

    cudaEventSynchronize(stop);
    float dt;
    HANDLE_ERROR( cudaEventElapsedTime(&dt, start, stop));
    printf(%s, dt);

    #define sum_squares(x) (x * (x + 1) * (2 * x + 1) / 6)
     
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    return c == 2 * sum_squares( (float) (N - 1) );
}
    """ % (arsize, nthrd, '"%0.6f"')
    with open(f_name, 'w') as f:
        f.write(prog)


if __name__=="__main__":
    f_name = 'test.cu'
    vectorSizeK = 3300 

    for power in range(3, 11):
        blockSize = 2 ** power 
        times = []
        gen_file(f_name, vectorSizeK, blockSize)
        subprocess.run(['/usr/local/cuda/bin/nvcc', 
                            '-o', 
                            'out', 
                            f_name])
        for _ in range(100):
            result = subprocess.run(['./out'], stdout=subprocess.PIPE)
            times.append(float(result.stdout))
            print(float(result.stdout), end=' ')
            sys.stdout.flush()

        print(f"\nAverage run time for block size of {blockSize} threads is {sum(times)/len(times):.6f} ms")
                        