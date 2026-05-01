// T4 Latency: 157.40 ms

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Logical tiles.
constexpr int BLOCK_SIZE_M = 128;
constexpr int BLOCK_SIZE_N = 16;
constexpr int BLOCK_SIZE_K = 128;

constexpr int BLOCK_SIZE_A = BLOCK_SIZE_M * BLOCK_SIZE_N;
constexpr int BLOCK_SIZE_B = BLOCK_SIZE_N * BLOCK_SIZE_K;
constexpr int BLOCK_SIZE_C = BLOCK_SIZE_M * BLOCK_SIZE_K;

// Physical threads.
constexpr int NUM_THREADS_X = 32;
constexpr int NUM_THREADS_Y = 8;
constexpr int NUM_THREADS = NUM_THREADS_X * NUM_THREADS_Y;

constexpr int THREAD_TILE_SIZE_X = BLOCK_SIZE_K / NUM_THREADS_X;  // 4
constexpr int THREAD_TILE_SIZE_Y = BLOCK_SIZE_M / NUM_THREADS_Y;  // 16

// Vectorized blocks.
constexpr int VEC_SIZE = 4;
constexpr int VEC_BLOCK_SIZE_N = BLOCK_SIZE_N / VEC_SIZE;
constexpr int VEC_BLOCK_SIZE_K = BLOCK_SIZE_K / VEC_SIZE;

constexpr int VEC_BLOCK_SIZE_A = BLOCK_SIZE_A / VEC_SIZE;
constexpr int VEC_BLOCK_SIZE_B = BLOCK_SIZE_B / VEC_SIZE;

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
        // Coalesced load A tile in vectors to the shared memory.
        for (int linear_vec = tid; linear_vec < VEC_BLOCK_SIZE_A; linear_vec += NUM_THREADS) {
            int linear_vec_id0 = linear_vec * VEC_SIZE;

            // Get the index in the shared memory.
            int A_s_r0 = linear_vec_id0 / BLOCK_SIZE_N;
            int A_s_c0 = linear_vec_id0 % BLOCK_SIZE_N;

            // Get the index in the global memory.
            int A_r0 = blockIdx.y * BLOCK_SIZE_M + A_s_r0;
            int A_c0 = n0 + A_s_c0;
            int A_id0 = A_r0 * N + A_c0;

            // Check conditions to use vectorized load.
            bool isInBound = A_r0 < M && (A_c0 + VEC_SIZE - 1) < N;
            bool isAligned = (
                reinterpret_cast<uintptr_t>(&A[A_id0]) % alignof(float4) == 0
                && reinterpret_cast<uintptr_t>(&A_s[A_s_r0][A_s_c0]) % alignof(float4) == 0
            );

            // Load in vectors.
            if (isInBound && isAligned) {
                float4 v = *reinterpret_cast<const float4*>(&A[A_id0]);
                *reinterpret_cast<float4*>(&A_s[A_s_r0][A_s_c0]) = v;
            }

            // Fallback: load in scalars.
            else {
                #pragma unroll
                for (int si = 0; si < VEC_SIZE; si++) {
                    if (A_r0 < M && A_c0 + si < N) A_s[A_s_r0][A_s_c0 + si] = A[A_id0 + si];
                    else A_s[A_s_r0][A_s_c0 + si] = 0.0f;
                }
            }
        }

        // Coalesced load B tile in vectors to the shared memory.
        for (int linear_vec = tid; linear_vec < VEC_BLOCK_SIZE_B; linear_vec += NUM_THREADS) {
            int linear_vec_id0 = linear_vec * VEC_SIZE;

            // Get the index in the shared memory.
            int B_s_r0 = linear_vec_id0 / BLOCK_SIZE_K;
            int B_s_c0 = linear_vec_id0 % BLOCK_SIZE_K;

            // Get the index in the global memory.
            int B_r0 = n0 + B_s_r0;
            int B_c0 = blockIdx.x * BLOCK_SIZE_K + B_s_c0;
            int B_id0 = B_r0 * K + B_c0;

            // Check conditions to use vectorized load.
            int isInBound = B_r0 < N && B_c0 + VEC_SIZE - 1 < K;
            int isAligned = (
                reinterpret_cast<uintptr_t>(&B[B_id0]) % alignof(float4) == 0
                && reinterpret_cast<uintptr_t>(&B_s[B_s_r0][B_s_c0]) % alignof(float4) == 0
            );

            // Load in vectors.
            if (isInBound && isAligned) {
                float4 v = *reinterpret_cast<const float4*>(&B[B_id0]);
                *reinterpret_cast<float4*>(&B_s[B_s_r0][B_s_c0]) = v;
            }

            // Fallback: load in scalars.
            else {
                for (int si = 0; si < VEC_SIZE; si++) {
                    if (B_r0 < N && B_c0 + si < K) B_s[B_s_r0][B_s_c0 + si] = B[B_id0 + si];
                    else B_s[B_s_r0][B_s_c0 + si] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Accumulate matmul results in per-thread register blocks.
        float A_r[THREAD_TILE_SIZE_Y];
        float B_r[THREAD_TILE_SIZE_X];

        for (int ni = 0; ni < BLOCK_SIZE_N; ni++) {
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
        if (C_ri < M) {
            int C_id0 = C_ri * K + C_c0;

            #pragma unroll
            for (int C_c0_offset = 0; C_c0_offset < THREAD_TILE_SIZE_X; C_c0_offset += VEC_SIZE) {
                int C_ci = C_c0 + C_c0_offset;
                int C_id = C_id0 + C_c0_offset;

                if (C_ci + VEC_SIZE - 1 < K && reinterpret_cast<uintptr_t>(&C[C_id]) % alignof(float4) == 0) {
                    // Vectorized store.
                    float4 out;
                    out.x = acc[ti][C_c0_offset];
                    out.y = acc[ti][C_c0_offset + 1];
                    out.z = acc[ti][C_c0_offset + 2];
                    out.w = acc[ti][C_c0_offset + 3];
                    *reinterpret_cast<float4*>(&C[C_id]) = out;
                } else {
                    // Scalar store.
                    #pragma unroll
                    for (int si = 0; si < VEC_SIZE; si++) {
                        if (C_ci + si < K) C[C_id + si] = acc[ti][C_c0_offset + si];
                    }
                }
            }
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
