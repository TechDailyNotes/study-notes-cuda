#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define M 7

__constant__ int d_msk[M];

__global__ void convolve_1d(int *d_vec, int *d_res, int n) {
    extern __shared__ int s_vec[];
    int g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int l_tid = threadIdx.x;
    if (g_tid >= n) return;

    int r = M / 2;
    s_vec[l_tid] = d_vec[g_tid - r];
    if (l_tid < 2 * r) {
        s_vec[l_tid + blockDim.x] = d_vec[g_tid - r + blockDim.x];
    }

    __syncthreads();

    int tmp = 0;
    for (int i = g_tid; i <= g_tid + 2 * r; i++) {
        tmp += d_vec[i] * d_msk[i - g_tid];
    }
    d_res[g_tid] = tmp;
}

void verify_result(int *h_vec, int *h_msk, int *h_res, int n) {
    for (int i = 0; i < n; i++) {
        int tmp = 0;
        int r = M / 2;

        for (int j = i; j <= i + 2 * r; j++) {
            tmp += h_vec[j] * h_msk[j - i];
        }

        assert(tmp == h_res[i]);
    }
}

int main() {
    int n = 1 << 20;
    int r = M / 2;
    int p = n + r * 2;

    int n_bytes = sizeof(int) * n;
    int m_bytes = sizeof(int) * M;
    int p_bytes = sizeof(int) * p;

    int *h_vec = (int*) malloc(p_bytes);
    int *h_msk = (int*) malloc(m_bytes);
    int *h_res = (int*) malloc(n_bytes);

    int *d_vec, *d_res;
    cudaMalloc(&d_vec, p_bytes);
    cudaMalloc(&d_res, n_bytes);

    for (int i = 0; i < p; i++) {
        if (i < r || i >= n + r) {
            h_vec[i] = 0;
        } else {
            h_vec[i] = rand() % 100;
        }
    }
    for (int i = 0; i < M; i++) {
        h_msk[i] = rand() % 10;
    }
    for (int i = 0; i < n; i++) {
        h_res[i] = 0;
    }

    cudaMemcpy(d_vec, h_vec, p_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_msk, h_msk, m_bytes);

    int num_threads = 1 << 8;
    dim3 blck_size(num_threads);
    dim3 grid_size((n + num_threads - 1) / num_threads);
    size_t smem_size = (num_threads + r * 2) * sizeof(int);

    convolve_1d<<<grid_size, blck_size, smem_size>>>(d_vec, d_res, n);
    cudaMemcpy(h_res, d_res, n_bytes, cudaMemcpyDeviceToHost);

    verify_result(h_vec, h_msk, h_res, n);

    cudaFree(d_vec);
    cudaFree(d_res);

    free(h_vec);
    free(h_msk);
    free(h_res);

    printf("Succees!");
    return 0;
}
