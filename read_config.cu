#include <iostream>
#include "book.h"

using namespace std;

__global__ void add(int a, int b, int *c){
    *c = a + b;
}

int main(void) {
    cudaDeviceProp prop;

    int count;
    HANDLE_ERROR( cudaGetDeviceCount(&count) );

    for (int i = 0; i < count; ++i) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i) );

        cout << "\n   --- General Information for device " << i << " ---\n";
        cout << "Name:  " << prop.name << endl;
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);

        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("Kernel execution timeout: ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");

        printf("Can map host memory: ");
        if (prop.canMapHostMemory)
            printf("Yes\n");
        else
            printf("No\n");

        printf("\n   --- Memory Information for device %d ---\n",  i);
        printf("Total global mem:  %ld\n", prop.totalGlobalMem);
        printf("Shared mem per block:  %ld\n", prop.sharedMemPerBlock);
        printf("Total constant mem:  %ld\n", prop.totalConstMem);
        printf("Total mem pitch:  %ld\n", prop.memPitch);
        printf("Texture Alignment:  %ld\n", prop.textureAlignment);

        printf("\n   --- MP Information for device %d ---\n",  i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", 
                prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", 
                prop.maxGridSize[0], prop.maxGridSize[1],
                prop.maxGridSize[2]);

        printf("\n");


    }

    return 0;
}

