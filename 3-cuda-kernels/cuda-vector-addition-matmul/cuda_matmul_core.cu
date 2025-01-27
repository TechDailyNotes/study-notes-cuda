#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_ERROR 1e-4f

void init_matrix(float *A, int num_row, int num_col) {
    for (int rowi = 0; rowi < num_row; rowi++) {
        for (int coli = 0; coli < num_col; coli++) {
            A[rowi * num_col + coli] = (float) rand() / RAND_MAX;
        }
    }
}

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ki++) {
                sum += A[mi * k + ki] * B[ki * n + ni];
            }
            C[mi * n + ni] = sum;
        }
    }
}

__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    // Step 0: Compute thread id.
    int mi = blockIdx.y * blockDim.y + threadIdx.y;
    int ni = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Bound check.
    if (mi < m && ni < n) {
        // Step 2: Execute logic.
        float sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            sum += A[mi * k + ki] * B[ki * n + ni];
        }
        C[mi * n + ni] = sum;
    }
}

bool validate_accuracy(float *C_cpu, float *C_gpu, int m, int n) {
    for (int mi = 0; mi < m; mi++) {
        for (int ni = 0; ni < n; ni++) {
            if (fabs(C_cpu[mi * n + ni] - C_gpu[mi * n + ni]) > MAX_ERROR) {
                // printf(
                //     "Accuracy error is larger than %.4f:\nmi = %d, ni = %d | CPU result is %f, GPU result is %f.\n",
                //     MAX_ERROR,
                //     mi,
                //     ni,
                //     C_cpu[mi * n + ni],
                //     C_gpu[mi * n + ni]
                // );
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Step 0: Compute memory/thread size.
    const int M = 256;
    const int K = 512;
    const int N = 256;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    const int DIM_BLOCK_X = 32;
    const int DIM_BLOCK_Y = 32;
    dim3 dim_block(DIM_BLOCK_X, DIM_BLOCK_Y);
    dim3 dim_grid((N + DIM_BLOCK_X - 1) / DIM_BLOCK_X, (M + DIM_BLOCK_Y - 1) / DIM_BLOCK_Y);

    // Step 1: Init memory.
    float *h_A = (float*) malloc(size_A);
    float *h_B = (float*) malloc(size_B);
    float *h_C_cpu = (float*) malloc(size_C);
    float *h_C_gpu = (float*) malloc(size_C);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Step 2: Move data from host to device.
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Step 3: Call device kernel.
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    matmul_gpu<<<dim_grid, dim_block>>>(d_A, d_B, d_C, M, N, K);

    // Step 4: Move data from device to host.
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    assert(validate_accuracy(h_C_cpu, h_C_gpu, M, N));
    printf("CPU/GPU results match.\n");

    // Step 5: Free memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
