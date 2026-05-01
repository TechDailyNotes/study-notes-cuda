// Latency: 854.63 ms

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;
constexpr int BLOCK_SIZE_M = 16;
constexpr int BLOCK_SIZE_N = 16;
constexpr int BLOCK_SIZE_K = 16;

constexpr int THREAD_TILE_SIZE_M = 2;
constexpr int THREAD_TILE_SIZE_N = 2;
constexpr int THREAD_TILE_SIZE_K = 2;

constexpr int THREADS_X = BLOCK_SIZE_K / THREAD_TILE_SIZE_K;
constexpr int THREADS_Y = BLOCK_SIZE_M / THREAD_TILE_SIZE_M;

__global__ void matrix_multiplication_kernel(
    const float* A,  // M * N
    const float* B,  // N * K
    float* C,  // M * K
    int M,
    int N,
    int K
) {
    int C_ri = blockIdx.y * blockDim.y + threadIdx.y;
    int C_ci = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_s[BLOCK_SIZE][BLOCK_SIZE];

    float acc = 0.0f;
    for (int ti = 0; ti < N; ti += BLOCK_SIZE) {
        // Move data from global to shared memory.
        int A_ci = ti + threadIdx.x;
        int A_idx = C_ri * N + A_ci;

        if (C_ri < M && A_ci < N) {
            A_s[threadIdx.y][threadIdx.x] = A[A_idx];
        } else {
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int B_ri = ti + threadIdx.y;
        int B_idx = B_ri * K + C_ci;

        if (B_ri < N && C_ci < K) {
            B_s[threadIdx.y][threadIdx.x] = B[B_idx];
        } else {
            B_s[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute matmul in tiles.
        #pragma unroll
        for (int tj = 0; tj < BLOCK_SIZE; tj++) {
            acc += A_s[threadIdx.y][tj] * B_s[tj][threadIdx.x];
        }

        __syncthreads();
    }

    if (C_ri < M && C_ci < K) {
        int C_idx = C_ri * K + C_ci;
        C[C_idx] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
