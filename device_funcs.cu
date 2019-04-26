#include <stdio.h>

/*
 * Initialize array values on the host.
 */

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * Check all elements have been doubled on the host.
 */

 bool checkElementsAreDoubled(int *a, int N)
 {
   int i;
   for (i = 0; i < N; ++i)
   {
     if (a[i] != i*2) return false;
   }
   return true;
 }
 
 /*
  * Check array initialized correctly.
  */
 
  bool checkArrayInit(int *a, int N)
  {
    int i;
    for (i = 0; i < N; ++i)
    {
      if (a[i] != i) return false;
    }
    return true;
  }
 
__device__
int doub(int num) {
    return 2*num;
}

/*
 * Double elements in parallel on the GPU.
 */

__global__
void doubleElements(int *a, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
//    a[i] *= 2;
    a[i] = doub(a[i]);
  }
}

int main()
{
    int N = 100;
    int *a;

    size_t size = N * sizeof(int);

    /*
    * Refactor this memory allocation to provide a pointer
    * `a` that can be used on both the host and the device.
    */

    a = (int *)malloc(size);

    init(a, N);

    bool isInit = checkArrayInit(a , N);
    printf("Is array initialized correctly? %s\n", isInit ? "TRUE" : "FALSE");

    int *d_a;
    cudaMallocManaged( &d_a, size);
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    size_t threads_per_block = 10;
    size_t number_of_blocks = 10;

    /*
    * This launch will not work until the pointer `a` is also
    * available to the device.
    */

    doubleElements<<<number_of_blocks, threads_per_block>>>(d_a, N);
    cudaDeviceSynchronize();

    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    bool areDoubled = checkElementsAreDoubled(a, N);
    printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

    /*
    * Refactor to free memory that has been allocated to be
    * accessed by both the host and the device.
    */

    free(a);
}