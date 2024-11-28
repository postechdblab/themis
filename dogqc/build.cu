#include <cub/cub.cuh>

int main() {

    int* keys;
    int* vals;
    cudaMalloc((void**)&keys, sizeof(int) * 1600);
    cudaMalloc((void**)&vals, sizeof(int) * 1600);

}