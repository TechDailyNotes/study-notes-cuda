#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define KERNEL_DIM 7
#define KERNEL_PAD (KERNEL_DIM / 2)

__constant__ int d_kernel[KERNEL_DIM * KERNEL_DIM];

__global__ void convolve_2d(int *d_matrix, int *d_result, int result_dim) {
    extern __shared__ int s_memory[];
    int smem_dim = blockDim.x + 2 * KERNEL_PAD;
    int matrix_dim = result_dim + 2 * KERNEL_PAD;

    int l_tid_c = threadIdx.x;
    int l_tid_r = threadIdx.y;
    int g_tid_c = threadIdx.x + blockIdx.x * blockDim.x;
    int g_tid_r = threadIdx.y + blockIdx.y * blockDim.y;
    if (g_tid_c >= result_dim) return;
    if (g_tid_r >= result_dim) return;

    int offset_r = 0;
    while (offset_r < smem_dim) {
        int offset_c = 0;
        while (offset_c < smem_dim) {
            int smem_r = l_tid_r + offset_r;
            int smem_c = l_tid_c + offset_c;
            if (smem_r < smem_dim && smem_c < smem_dim) {
                s_memory[smem_r * smem_dim + smem_c] = \
                d_matrix[(g_tid_r + offset_r) * matrix_dim + (g_tid_c + offset_c)];
            }

            offset_c += blockDim.x;
        }

        offset_r += blockDim.y;
    }

    __syncthreads();

    int tmp = 0;
    int lo_r = l_tid_r;
    int hi_r = l_tid_r + 2 * KERNEL_PAD;
    int lo_c = l_tid_c;
    int hi_c = l_tid_c + 2 * KERNEL_PAD;

    for (int ri = lo_r; ri <= hi_r; ri++) {
        for (int ci = lo_c; ci <= hi_c; ci++) {
            tmp += (
                s_memory[ri * smem_dim + ci] *
                d_kernel[(ri - lo_r) * KERNEL_DIM + (ci - lo_c)]
            );
        }
    }

    d_result[g_tid_r * result_dim + g_tid_c] = tmp;
}

void verify_result(int *h_matrix, int *h_kernel, int *h_result, int result_dim) {
    for (int rowi = 0; rowi < result_dim; rowi++) {
        for (int coli = 0; coli < result_dim; coli++) {
            int tmp = 0;
            int matrix_dim = result_dim + 2 * KERNEL_PAD;

            for (int ri = rowi; ri <= rowi + 2 * KERNEL_PAD; ri++) {
                for (int ci = coli; ci <= coli + 2 * KERNEL_PAD; ci++) {
                    tmp += (
                        h_matrix[ri * matrix_dim + ci] *
                        h_kernel[(ri - rowi) * KERNEL_DIM + (ci - coli)]
                    );
                }
            }

            // printf(
            //     "tmp = %d, result = %d\n",
            //     tmp, h_result[rowi * result_dim + coli]
            // );
            assert(tmp == h_result[rowi * result_dim + coli]);
        }
    }
}

int main() {
    int result_dim = 1 << 10;
    int matrix_dim = result_dim + KERNEL_PAD * 2;

    int result_bytes = sizeof(int) * result_dim * result_dim;
    int matrix_bytes = sizeof(int) * matrix_dim * matrix_dim;
    int kernel_bytes = sizeof(int) * KERNEL_DIM * KERNEL_DIM;

    int *h_matrix = (int *) malloc(matrix_bytes);
    int *h_kernel = (int *) malloc(kernel_bytes);
    int *h_result = (int *) malloc(result_bytes);

    for (int rowi = 0; rowi < matrix_dim; rowi++) {
        for (int coli = 0; coli < matrix_dim; coli++) {
            if (
                rowi < KERNEL_PAD || rowi >= result_dim + KERNEL_PAD ||
                coli < KERNEL_PAD || coli >= result_dim + KERNEL_PAD
            ) {
                h_matrix[rowi * matrix_dim + coli] = 0;
            } else {
                h_matrix[rowi * matrix_dim + coli] = rand() % 100;
            }
        }
    }
    for (int rowi = 0; rowi < KERNEL_DIM; rowi++) {
        for (int coli = 0; coli < KERNEL_DIM; coli++) {
            h_kernel[rowi * KERNEL_DIM + coli] = rand() % 10;
        }
    }

    int *d_matrix, *d_result;
    cudaMalloc(&d_matrix, matrix_bytes);
    cudaMalloc(&d_result, result_bytes);

    cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_bytes);

    int threads_per_block_dim = 1 << 4;
    int blocks_per_grid_dim = (result_dim + threads_per_block_dim - 1) / threads_per_block_dim;
    dim3 block_size(threads_per_block_dim, threads_per_block_dim);
    dim3 grid_size(blocks_per_grid_dim, blocks_per_grid_dim);

    int smem_per_block_dim = threads_per_block_dim + 2 * KERNEL_PAD;
    size_t smem_size = sizeof(int) * smem_per_block_dim * smem_per_block_dim;

    convolve_2d<<<grid_size, block_size, smem_size>>>(d_matrix, d_result, result_dim);
    cudaMemcpy(h_result, d_result, result_bytes, cudaMemcpyDeviceToHost);
    verify_result(h_matrix, h_kernel, h_result, result_dim);

    cudaFree(d_matrix);
    cudaFree(d_result);

    free(h_matrix);
    free(h_kernel);
    free(h_result);

    printf("Success!\n");
    return 0;
}