#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void print_value(float *v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Correct print in device kernel.
        printf("v[%d] = %f\n", i, v[i]);
        // Incorrect print in device kernel because `<<` is undefined in device.
        // cout << "v[" << i << "] = " v[i] << endl;
    }
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(float);
    const int DIM_BLOCK = 8;
    int dim_grid = (N + DIM_BLOCK - 1) / DIM_BLOCK;

    float *d_v;
    cudaMalloc(&d_v, size);
    cout << "d_v = " << d_v << endl;  // Address stored in device pointer is visible to host.
    cudaMemset(d_v, 0, size);

    print_value<<<dim_grid, DIM_BLOCK>>>(d_v, N);  // Address stored in device pointer is inaccessible to host.

    cudaFree(d_v);

    return 0;
}
