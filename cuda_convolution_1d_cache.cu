%%cuda

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 7

__constant__ int d_msk[M];

__global__ void convolve_1d(int *d_vec, int *d_res, int n) {
    extern __shared__ int s_mem[];

    int g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l_tid = threadIdx.x;
    if (g_tid >= n) return;

    s_mem[l_tid] = d_vec[g_tid];
    __syncthreads();

    int tmp = 0;
    int r = M / 2;

    for (int i = l_tid; i <= l_tid + 2 * r; i++) {
        if (i < blockDim.x) tmp += s_mem[i] * d_msk[i - l_tid];
        else tmp += d_vec[g_tid + i - l_tid] * d_msk[i - l_tid];
    }

    d_res[g_tid] = tmp;
}

void verify_result(int *h_vec, int *h_msk, int *h_res, int n) {
    for (int i = 0; i < n; i++) {
        int tmp = 0;
        int r = M / 2;

        for (int j = i; j <= i + 2 * r; j++) {
            tmp += h_msk[j - i] * h_vec[j];
        }

        // printf("tmp = %d, res = %d\n", tmp, h_res[i]);
        assert(tmp == h_res[i]);
    }
}

int main() {
    int n = 1 << 20;
    int r = M / 2;
    int p = n + 2 * r;

    int n_bytes = sizeof(int) * n;
    int m_bytes = sizeof(int) * M;
    int p_bytes = sizeof(int) * p;

    int *h_vec = (int *) malloc(p_bytes);
    int *h_msk = (int *) malloc(m_bytes);
    int *h_res = (int *) malloc(n_bytes);

    for (int i = 0; i < p; i++) {
        if (i < r || i >= n + r) h_vec[i] = 0;
        else h_vec[i] = rand() % 100;
    }
    for (int i = 0; i < M; i++) {
        h_msk[i] = rand() % 10;
    }

    int *d_vec, *d_res;
    cudaMalloc(&d_vec, p_bytes);
    cudaMalloc(&d_res, n_bytes);

    cudaMemcpy(d_vec, h_vec, p_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_msk, h_msk, m_bytes);

    int num_threads = 1 << 8;
    dim3 blck_size(num_threads);
    dim3 grid_size((n + num_threads - 1) / num_threads);
    size_t smem_size = sizeof(int) * num_threads;
    convolve_1d<<<grid_size, blck_size, smem_size>>>(d_vec, d_res, n);

    cudaMemcpy(h_res, d_res, n_bytes, cudaMemcpyDeviceToHost);
    verify_result(h_vec, h_msk, h_res, n);

    free(h_vec);
    free(h_msk);
    free(h_res);

    printf("Succees!");
    return 0;
}