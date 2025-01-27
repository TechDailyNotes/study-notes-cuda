# `cuda_vector_addition_1` Template

```cpp
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

__global__ void add_vector(float *a, float *b, float *c, int n) {
    // Step 0: Compute thread id.
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Bound check.
    if (id < n) {
        // Step 2: Execute logic.
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Step 0: Compute memory and thread size.
    const int N = 10'000'000;
    const int DIM_BLOCK = 256;
    size_t size = N * sizeof(float);
    int DIM_GRID = (N + DIM_BLOCK - 1) / DIM_BLOCK;

    // Step 1: Allocate memory.
    float *h_a = (float*) malloc(size);
    float *h_b = (float*) malloc(size);
    float *h_c = (float*) malloc(size);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    srand(time(NULL));
    init_vector(d_a, N);
    init_vector(d_b, N);

    // Step 2: Move data from host to device.
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Step 3: Call device kernel.
    add_vector<<<DIM_GRID, DIM_BLOCK>>>(d_a, d_b, d_c, N);
    // Necessary between two kernel calls because kernels are not auto synced.
    // Unnecessary between a kernel call and a DMA function because DMA forces devices to sync.
    cudaDeviceSynchronize();

    // Step 4: Move data from device to host.
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Step 5: Free memory.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```
