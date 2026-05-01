// T4 Latency: 172.65 ms

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Logical tiles.
constexpr int BLOCK_SIZE_M = 128;
constexpr int BLOCK_SIZE_N = 8;
constexpr int BLOCK_SIZE_K = 128;

constexpr int BLOCK_SIZE_A = BLOCK_SIZE_M * BLOCK_SIZE_N;
constexpr int BLOCK_SIZE_B = BLOCK_SIZE_N * BLOCK_SIZE_K;
constexpr int BLOCK_SIZE_C = BLOCK_SIZE_M * BLOCK_SIZE_K;

// Physical threads.
constexpr int NUM_THREADS_X = 32;  // Ensure coalesced memory access.
constexpr int NUM_THREADS_Y = 8;
constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;

constexpr int THREAD_TILE_SIZE_X = BLOCK_SIZE_K / NUM_THREADS_X;  // 4
constexpr int THREAD_TILE_SIZE_Y = BLOCK_SIZE_M / NUM_THREADS_Y;  // 4

__global__ void matmul_kernel(
    const float* A,  // M * N
    const float* B,  // N * K
    float* C,  // M * K
    int M, int N, int K
) {
    // Get the thread tile base index in the shared memory.
    int tid = threadIdx.x;
    int tx0 = tid % NUM_THREADS_X;
    int ty0 = tid / NUM_THREADS_X;

    // Get the thread tile base index in C.
    int C_r0 = blockIdx.y * BLOCK_SIZE_M + ty0 * THREAD_TILE_SIZE_Y;
    int C_c0 = blockIdx.x * BLOCK_SIZE_K + tx0 * THREAD_TILE_SIZE_X;

    // Accumulate matmul results in block tiles.
    float acc[THREAD_TILE_SIZE_Y][THREAD_TILE_SIZE_X] = {0.0f};

    // Run tiled kernel with sliding window, in shared memory.
    __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_N];
    __shared__ float B_s[BLOCK_SIZE_N][BLOCK_SIZE_K];

    for (int n0 = 0; n0 < N; n0 += BLOCK_SIZE_N) {
        // Coalesced load A tile to the shared memory.
        for (int linear = tid; linear < BLOCK_SIZE_A; linear += NUM_THREADS) {
            // Get the index in the shared memory.
            int A_s_ri = linear / BLOCK_SIZE_N;
            int A_s_ci = linear % BLOCK_SIZE_N;

            // Get the index in the global memory.
            int A_ri = blockIdx.y * BLOCK_SIZE_M + A_s_ri;
            int A_ci = n0 + A_s_ci;
            int A_id = A_ri * N + A_ci;

            if (A_ri < M & A_ci < N) A_s[A_s_ri][A_s_ci] = A[A_id];
            else A_s[A_s_ri][A_s_ci] = 0.0f;
        }

        // Coalesced load B tile to the shared memory.
        for (int linear = tid; linear < BLOCK_SIZE_B; linear += NUM_THREADS) {
            // Get the index in the shared memory.
            int B_s_ri = linear / BLOCK_SIZE_K;
            int B_s_ci = linear % BLOCK_SIZE_K;

            // Get the index in the global memory.
            int B_ri = n0 + B_s_ri;
            int B_ci = blockIdx.x * BLOCK_SIZE_K + B_s_ci;
            int B_id = B_ri * K + B_ci;

            if (B_ri < N && B_ci < K) B_s[B_s_ri][B_s_ci] = B[B_id];
            else B_s[B_s_ri][B_s_ci] = 0.0f;
        }

        __syncthreads();

        // Accumulate matmul results in per-thread register blocks.
        for (int ni = 0; ni < BLOCK_SIZE_N; ni++) {
            float A_r[THREAD_TILE_SIZE_Y];
            float B_r[THREAD_TILE_SIZE_X];

            // Load shared memory to per-thread register.
            #pragma unroll
            for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
                // Load A.
                int A_r_id = ti;
                int A_s_ri = ty0 * THREAD_TILE_SIZE_Y + ti;
                int A_s_ci = ni;

                A_r[A_r_id] = A_s[A_s_ri][A_s_ci];
            }

            #pragma unroll
            for (int tj = 0; tj < THREAD_TILE_SIZE_X; tj++) {
                // Load B.
                int B_r_id = tj;
                int B_s_ri = ni;
                int B_s_ci = tx0 * THREAD_TILE_SIZE_X + tj;

                B_r[B_r_id] = B_s[B_s_ri][B_s_ci];
            }

            // Compute multiplication results in the thread register.
            #pragma unroll
            for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
                #pragma unroll
                for (int tj = 0; tj < THREAD_TILE_SIZE_X; tj++) {
                    acc[ti][tj] += A_r[ti] * B_r[tj];
                }
            }
        }

        __syncthreads();
    }

    // Store matmul results in C.
    // TODO Change the vector size when `THREAD_TILE_SIZE_X` changes.
    #pragma unroll
    for (int ti = 0; ti < THREAD_TILE_SIZE_Y; ti++) {
        int C_ri = C_r0 + ti;
        if (C_ri >= M) return;

        int C_id = C_ri * K + C_c0;

        if (C_c0 + 3 < K && reinterpret_cast<uintptr_t>(&C[C_id]) % alignof(float4) == 0) {
            // Vectorized store.
            float4 out;
            out.x = acc[ti][0];
            out.y = acc[ti][1];
            out.z = acc[ti][2];
            out.w = acc[ti][3];
            *reinterpret_cast<float4*>(&C[C_id]) = out;
        } else {
            // Scalar store.
            if (C_c0 < K) C[C_id] = acc[ti][0];
            if (C_c0 + 1 < K) C[C_id + 1] = acc[ti][1];
            if (C_c0 + 2 < K) C[C_id + 2] = acc[ti][2];
            if (C_c0 + 3 < K) C[C_id + 3] = acc[ti][3];
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel runtime error: %s\n", cudaGetErrorString(err));
    }
}
