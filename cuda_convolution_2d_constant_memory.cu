#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define KERNEL_DIM 7
#define KERNEL_PAD (KERNEL_DIM / 2)

__constant__ int d_kernel[KERNEL_DIM * KERNEL_DIM];

__global__ void convolve_2d(int *d_matrix, int *d_result, int matrix_dim) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (tid_x >= matrix_dim || tid_y >= matrix_dim) return;

    int tmp = 0;
    int lo_col = tid_x - KERNEL_PAD;
    int hi_col = tid_x + KERNEL_PAD;
    int lo_row = tid_y - KERNEL_PAD;
    int hi_row = tid_y + KERNEL_PAD;

    for (int rowi = lo_row; rowi <= hi_row; rowi++) {
        for (int coli = lo_col; coli <= hi_col; coli++) {
            if (rowi < 0 || rowi >= matrix_dim) continue;
            if (coli < 0 || coli >= matrix_dim) continue;
            tmp += (
                d_matrix[rowi * matrix_dim + coli] *
                d_kernel[(rowi - lo_row) * KERNEL_DIM + (coli - lo_col)]
            );
        }
    }

    d_result[tid_y * matrix_dim + tid_x] = tmp;
}

void verify_result(int *h_matrix, int *h_kernel, int *h_result, int matrix_dim) {
    for (int rowi = 0; rowi < matrix_dim; rowi++) {
        for (int coli = 0; coli < matrix_dim; coli++) {
            int tmp = 0;
            int lo_row = rowi - KERNEL_PAD;
            int hi_row = rowi + KERNEL_PAD;
            int lo_col = coli - KERNEL_PAD;
            int hi_col = coli + KERNEL_PAD;

            for (int ri = lo_row; ri <= hi_row; ri++) {
                for (int ci = lo_col; ci <= hi_col; ci++) {
                    if (ri < 0 || ri >= matrix_dim) continue;
                    if (ci < 0 || ci >= matrix_dim) continue;
                    tmp += (
                        h_matrix[ri * matrix_dim + ci] *
                        h_kernel[(ri - lo_row) * KERNEL_DIM + (ci - lo_col)]
                    );
                }
            }

            // printf(
            //     "tmp = %d, result = %d\n",
            //     tmp, h_result[rowi * matrix_dim + coli]
            // );
            assert(tmp == h_result[rowi * matrix_dim + coli]);
        }
    }
}

void init_array(int *arr, int dim) {
    for (int ri = 0; ri < dim; ri++) {
        for (int ci = 0; ci < dim; ci++) {
            arr[ri * dim + ci] = rand() % 100;
        }
    }
}

int main() {
    int matrix_dim = 1 << 10;
    int matrix_bytes = sizeof(int) * matrix_dim * matrix_dim;
    int kernel_bytes = sizeof(int) * KERNEL_DIM * KERNEL_DIM;

    int *h_matrix = (int *) malloc(matrix_bytes);
    int *h_kernel = (int *) malloc(kernel_bytes);
    int *h_result = (int *) malloc(matrix_bytes);

    init_array(h_matrix, matrix_dim);
    init_array(h_kernel, KERNEL_DIM);

    int *d_matrix, *d_result;
    cudaMalloc(&d_matrix, matrix_bytes);
    cudaMalloc(&d_result, matrix_bytes);

    cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_bytes);

    int threads_per_block_dim = 1 << 4;
    int blocks_per_grid_dim = (matrix_dim + threads_per_block_dim - 1) / threads_per_block_dim;
    dim3 block_size(threads_per_block_dim, threads_per_block_dim);
    dim3 grid_size(blocks_per_grid_dim, blocks_per_grid_dim);

    convolve_2d<<<grid_size, block_size>>>(d_matrix, d_result, matrix_dim);
    cudaMemcpy(h_result, d_result, matrix_bytes, cudaMemcpyDeviceToHost);
    verify_result(h_matrix, h_kernel, h_result, matrix_dim);

    cudaFree(d_matrix);
    cudaFree(d_result);

    free(h_matrix);
    free(h_kernel);
    free(h_result);

    printf("Success!\n");
    return 0;
}