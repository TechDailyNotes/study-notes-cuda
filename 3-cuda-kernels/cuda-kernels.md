# Cuda Kernels

## Kernel Launch Params

```cpp
kernel<<<Dg, Db, Ns, S>>>();
// Dg (dim3) means the dimension of the grid.
// Db (dim3) means the dimension of the block.
// Ns (size_t) means the byte size of the shared memory per block.
// S (cudaStream_t) means the associated cuda stream.
```

## Thread Synchronization

```cpp
cudaDeviceSynchronize();
// Called from host functions.
// Synchronize all the devices.

__syncthreads();
// Called from `__global__` or `__device__` functions.
// Synchronize all the threads within a grid.

__syncwarps();
// Called from `__global__` or `__device__` functions.
// Synchronize all the threads within a warp.
```

### Example: Bit Shift

```cpp
__global__ void shift_bit(int *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int tmp = a[i+1];
        __syncthreads();
        a[i] = tmp;
        __syncthreads();
    }
}
```

## SIMD/SIMT/SPMD

- SIMD (Single Instruction Multiple Data): CPU
- SIMT (Single Instruction Multiple Thread): GPU
- SPMD (Single Program Multiple Device): PyTorch

## Theoretical limits of threads

- Max number of threads in a block is 1024.
- Max number of threads in a warp is 32.
- Max number of warps in a block is 32.

## Math Intrinsics

Device-only hardware instructions for fundamental math operations.
