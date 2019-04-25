#include <iostream>
#include <ctime>

using namespace std;

void init_vector(int v[], int size) {
    for (auto i = 0; i < size; ++i)
        v[i] = i;
}

void add(const int a[], const int b[], int c[], const int size) {
    for (auto i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    struct timespec old_time, new_time;
    unsigned long int oldNs, newNs; 

    //const int len = 500000000;
    const int len = 310000000;

    int *a = new int [len];
    int *b = new int [len];
    int *c = new int[len];

    init_vector(a, len);
    init_vector(b, len);


    clock_gettime(CLOCK_MONOTONIC, &old_time);
    add(a, b, c, len);
    clock_gettime(CLOCK_MONOTONIC, &new_time);


    oldNs = old_time.tv_sec * 1000000000ull + old_time.tv_nsec;
    newNs = new_time.tv_sec * 1000000000ull + new_time.tv_nsec;
    float dt = (newNs - oldNs) * 0.000000001f;
    //printf("Resulting array size is %d, addition took %0.4f seconds \n", len, dt);
    cout << "Resulting array size is " << len 
        << " addition took " << dt << " seconds \n";

    free(a);
    free(b);
    free(c);

    return 0;
}
