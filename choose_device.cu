#include <iostream>
#include "book.h"

using namespace std;


int main(void) {
    cudaDeviceProp prop;
    int dev;

    HANDLE_ERROR( cudaGetDevice(&dev) );

    printf("ID of current CUDA device: %d\n", dev);

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 3;
    prop.minor = 0;

    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );

    printf("ID of CUDA device closest to revision 3.0: %d\n", dev);

    HANDLE_ERROR( cudaSetDevice( dev ) );


    return 0;
}

