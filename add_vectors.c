#include <stdio.h>
#include <time.h>

void init_vector(int* v, int size)  {
    for (int i = 0; i < size; ++i)
        v[i] = i;
}

void add(const int a[], const int b[], int c[], int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char *argv[]) {
    int len = 500000000;
    time_t old_time, new_time;

    int a[len];
    int b[len];
    int c[len];
    init_vector(&a, len);
    init_vector(b, len);

    time(&old_time);
    add(a, b, c, len);
    time(&new_time);

    printf("Resulting array size is %d, addition took %ld seconds \n", len, new_time - old_time);

    return 0;
}