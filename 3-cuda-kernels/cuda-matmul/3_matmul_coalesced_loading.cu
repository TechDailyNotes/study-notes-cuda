// Latency: 246.27 ms

#include <cuda_runtime.h>

// Logical tile
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 64;

constexpr int BLOCK_SIZE_A = BLOCK_SIZE_M * BLOCK_SIZE_N;
constexpr int BLOCK_SIZE_B = BLOCK_SIZE_N * BLOCK_SIZE_K;
constexpr int BLOCK_SIZE_C = BLOCK_SIZE_M * BLOCK_SIZE_K;

// Physical threads
constexpr int NUM_THREADS_X = 32;  // Ensure thread-warp coalesced memory access.
constexpr int NUM_THREADS_Y = 8;
constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;  // 256

constexpr int THREAD_TILE_SIZE_X = BLOCK_SIZE_K / NUM_THREADS_X;  // 2
constexpr int THREAD_TILE_SIZE_Y = BLOCK_SIZE_M / NUM_THREADS_Y;  // 8

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Get thread base idx in shared memory.
    int tid = threadIdx.x;
    int tx = tid % NUM_THREADS_X;
    int ty = tid / NUM_THREADS_X;

    // Get thread base idx in C.
    int C_r0 = blockIdx.y * BLOCK_SIZE_M + ty * THREAD_TILE_SIZE_Y;
    int C_c0 = blockIdx.x * BLOCK_SIZE_K + tx * THREAD_TILE_SIZE_X;

    // Accumulate matmul with sliding block tiles.
    __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_N];
    __shared__ float B_s[BLOCK_SIZE_N][BLOCK_SIZE_K];

    float acc[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0f};

    for (int n0 = 0; n0 < N; n0 += BLOCK_SIZE_N) {
        // Coalesce data load from A to A_s.
        for (int linear = tid; linear < BLOCK_SIZE_A; linear += NUM_THREADS) {
            int A_s_ri = linear / BLOCK_SIZE_N;
            int A_s_ci = linear % BLOCK_SIZE_N;

            int A_ri = blockIdx.y * BLOCK_SIZE_M + A_s_ri;
            int A_ci = n0 + A_s_ci;
            int A_id = A_ri * N + A_ci;

            if (A_ri < M && A_ci < N) A_s[A_s_ri][A_s_ci] = A[A_id];
            else A_s[A_s_ri][A_s_ci] = 0.0f;
        }

        // Coalesce data load from B to B_s.
        for (int linear = tid; linear < BLOCK_SIZE_B; linear += NUM_THREADS) {
            int B_s_ri = linear / BLOCK_SIZE_K;
            int B_s_ci = linear % BLOCK_SIZE_K;

            int B_ri = n0 + B_s_ri;
            int B_ci = blockIdx.x * BLOCK_SIZE_K + B_s_ci;
            int B_id = B_ri * K + B_ci;

            if (B_ri < N && B_ci < K) B_s[B_s_ri][B_s_ci] = B[B_id];
            else B_s[B_s_ri][B_s_ci] = 0.0f;
        }

        __syncthreads();

        // Move shared memory data to per-thread register blocks to accumulate matmul results.
        for (int ni = 0; ni < BLOCK_SIZE_N; ni++) {
            float A_r[THREAD_TILE_SIZE_Y];
            float B_r[THREAD_TILE_SIZE_X];

            // Move data from smem to register A.
            for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
                int A_r_id = ti;

                int A_s_ri = ty * THREAD_TILE_SIZE_Y + ti;
                int A_s_ci = ni;

                A_r[A_r_id] = A_s[A_s_ri][A_s_ci];
            }

            // Move data from smem to register B.
            for (int ti = 0; ti < THREAD_TILE_SIZE_X; ti++) {
                int B_r_id = ti;

                int B_s_ri = ni;
                int B_s_ci = tx * THREAD_TILE_SIZE_X + ti;

                B_r[B_r_id] = B_s[B_s_ri][B_s_ci];
            }

            // Accumulate partial matmul results.
            for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
                for (int tj = 0; tj < THREAD_TILE_SIZE_X; tj++) {
                    acc[ti][tj] += A_r[ti] * B_r[tj];
                }
            }
        }

        __syncthreads();
    }

    // Save matmul results to C.
    for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
        int C_ri = C_r0 + ti;
        if (C_ri >= M) continue;

        for (int tj = 0; tj < THREAD_TILE_SIZE_X; tj++) {
            int C_ci = C_c0 + tj;
            if (C_ci >= K) continue;

            int C_id = C_ri * K + C_ci;
            C[C_id] = acc[ti][tj];
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(NUM_THREADS);
    dim3 blocksPerGrid(
        (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K,
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
