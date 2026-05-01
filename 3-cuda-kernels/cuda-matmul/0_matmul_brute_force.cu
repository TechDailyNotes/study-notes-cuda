// Latency: 948.95 ms

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void matrix_multiplication_kernel(
    const float* A,  // M * N
    const float* B,  // N * M
    float* C,  // M * K
    int M,
    int N,
    int K
) {
    int ri = blockIdx.y * blockDim.y + threadIdx.y;
    int ci = blockIdx.x * blockDim.x + threadIdx.x;

    // Thread is out of bound.
    if (ri >= M || ci >= K) return;

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        int aIdx = ri * N + i;
        int bIdx = i * K + ci;
        sum += A[aIdx] * B[bIdx];
    }
    int cIdx = ri * K + ci;
    C[cIdx] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (K + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
