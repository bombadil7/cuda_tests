#include <time.h>
#include <cuda.h>
#include "book.h"

void init_vector(int* v, int size)  {
    for (int i = 0; i < size; ++i)
        v[i] = i;
}

__global__ void add(int *a, int *b, int size) {
    int tid = blockIdx.x;
    if (tid < size)
        b[tid] += a[tid];
}

int main(void){
    time_t old_time, new_time;
    
    const int len =   15000000;
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

    time(&old_time);
    add<<<65000, 1>>>(d_a, d_b, len);
    time(&new_time);

    HANDLE_ERROR(
        cudaMemcpy(c, d_b, len * sizeof(int),
            cudaMemcpyDeviceToHost));

    printf("Resulting array size is %d, addition took %ld seconds \n", len, new_time - old_time);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
  //  cudaFree(d_c);
    return 0;
}