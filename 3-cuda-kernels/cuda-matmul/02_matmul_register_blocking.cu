// T4 Latency: 299.64 ms

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 64;

constexpr int THREAD_TILE_SIZE_M = 4;
constexpr int THREAD_TILE_SIZE_N = 4;
constexpr int THREAD_TILE_SIZE_K = 4;

constexpr int THREADS_X = BLOCK_SIZE_K / THREAD_TILE_SIZE_K;
constexpr int THREADS_Y = BLOCK_SIZE_M / THREAD_TILE_SIZE_M;

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int C_r0 = blockIdx.y * BLOCK_SIZE_M + threadIdx.y * THREAD_TILE_SIZE_M;
    int C_c0 = blockIdx.x * BLOCK_SIZE_K + threadIdx.x * THREAD_TILE_SIZE_K;

    // Init the per-block shared memory, to hold the tiled matrix values.
    __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_N];
    __shared__ float B_s[BLOCK_SIZE_N][BLOCK_SIZE_K];

    // Init the per-thread register file, to hold intermediate matmul values.
    float acc[THREAD_TILE_SIZE_M][THREAD_TILE_SIZE_K] = {0.0f};

    for (int ni = 0; ni < N; ni += BLOCK_SIZE_N) {
        // Move A data from the global to shared memory.
        int A_r0 = C_r0;
        int A_c0 = ni + threadIdx.x * THREAD_TILE_SIZE_N;

        #pragma unroll 4
        for (int ti = 0; ti < THREAD_TILE_SIZE_M; ti++) {
            int A_ri = A_r0 + ti;
            int A_s_ri = threadIdx.y * THREAD_TILE_SIZE_M + ti;

            #pragma unroll 4
            for (int tj = 0; tj < THREAD_TILE_SIZE_N; tj++) {
                int A_ci = A_c0 + tj;
                int A_id = A_ri * N + A_ci;
                int A_s_ci = threadIdx.x * THREAD_TILE_SIZE_N + tj;

                if (A_ri < M && A_ci < N) A_s[A_s_ri][A_s_ci] = A[A_id];
                else A_s[A_s_ri][A_s_ci] = 0.0f;
            }
        }

        // Move B data from the global to shared memory.
        int B_r0 = ni + threadIdx.y * THREAD_TILE_SIZE_N;
        int B_c0 = C_c0;

        #pragma unroll 4
        for (int ti = 0; ti < THREAD_TILE_SIZE_N; ti++) {
            int B_ri = B_r0 + ti;
            int B_s_ri = threadIdx.y * THREAD_TILE_SIZE_N + ti;

            #pragma unroll 4
            for (int tj = 0; tj < THREAD_TILE_SIZE_K; tj++) {
                int B_ci = B_c0 + tj;
                int B_id = B_ri * K + B_ci;
                int B_s_ci = threadIdx.x * THREAD_TILE_SIZE_K + tj;

                if (B_ri < N && B_ci < K) B_s[B_s_ri][B_s_ci] = B[B_id];
                else B_s[B_s_ri][B_s_ci] = 0.0f;
            }
        }

        __syncthreads();

        // Calculate the partial matmul in the current blocks.
        #pragma unroll 4
        for (int ni = 0; ni < BLOCK_SIZE_N; ni++) {
            float A_r[THREAD_TILE_SIZE_M] = {0.0f};
            float B_r[THREAD_TILE_SIZE_K] = {0.0f};

            // Move A data from shared memory to register file.
            #pragma unroll 4
            for (int ti = 0; ti < THREAD_TILE_SIZE_M; ti++) {
                A_r[ti] = A_s[threadIdx.y * THREAD_TILE_SIZE_M + ti][ni];
            }

            // Move B data from shared memory to register file.
            #pragma unroll 4
            for (int tj = 0; tj < THREAD_TILE_SIZE_K; tj++) {
                B_r[tj] = B_s[ni][threadIdx.x * THREAD_TILE_SIZE_K + tj];
            }

            // Calculate the partial matmul in thread register.
            #pragma unroll 4
            for (int ti = 0; ti < THREAD_TILE_SIZE_M; ti++) {
                #pragma unroll
                for (int tj = 0; tj < THREAD_TILE_SIZE_K; tj++) {
                    acc[ti][tj] += A_r[ti] * B_r[tj];
                }
            }
        }

        __syncthreads();
    }

    // Move the matmul results back to the global memory.
    #pragma unroll 4
    for (int ti = 0; ti < THREAD_TILE_SIZE_M; ti++) {
        int C_ri = C_r0 + ti;
        if (C_ri >= M) continue;

        #pragma unroll 4
        for (int tj = 0; tj < THREAD_TILE_SIZE_K; tj++) {
            int C_ci = C_c0 + tj;
            if (C_ci >= K) continue;

            int C_id = C_ri * K + C_ci;
            C[C_id] = acc[ti][tj];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(THREADS_X, THREADS_Y);
    dim3 blocksPerGrid(
        (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K,
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
