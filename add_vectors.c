#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void init_vector(int* v, int size)  {
    for (int i = 0; i < size; ++i)
        v[i] = i;
}

void add(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}


int main(int argc, char *argv[]) {
    time_t old_time, new_time;

//    const int len = 500000000;
//    const int len =   45000000;
    const int len =   15000000;

    int* a = (int*) malloc(len * sizeof(int));
    int* b = (int*) malloc(len * sizeof(int));
    int* c = (int*) malloc(len * sizeof(int));

    init_vector(a, len);
    init_vector(b, len);

    time(&old_time);
    add(a, b, c, len);
    time(&new_time);

    printf("Resulting array size is %d, addition took %ld seconds \n", len, new_time - old_time);

    free(a);
    free(b);
    free(c);

    return 0;
}