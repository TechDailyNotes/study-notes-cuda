# Indexing Template

```cpp
__global__ void kernel(void) {}

int main(int argc, char **argv) {
    // Step 1: Define dimensions of block and grid.
    dim3 dimBlock(numThreadX, numThreadY, numThreadZ);
    dim3 dimGrid(numBlockX, numBlockY, numBlockZ);

    // Step 2: Call device kernel.
    kernel<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}
```
