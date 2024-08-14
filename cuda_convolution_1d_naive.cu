#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolve_1d(int *d_vec, int *d_msk, int *d_res, int n, int m) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int tmp = 0;
    int lo = tid - m / 2;
    int hi = tid + m / 2;

    for (int i = lo; i <= hi; i++) {
        if (i < 0 || i >= n) continue;
        tmp += d_vec[i] * d_msk[i-lo];
    }

    d_res[tid] = tmp;
}

void verify_result(int *h_vec, int *h_msk, int *h_res, int n, int m) {
    for (int i = 0; i < n; i++) {
        int tmp = 0;
        int lo = i - m / 2;
        int hi = i + m / 2;

        for (int j = lo; j <= hi; j++) {
            if (j < 0 || j >= n) continue;
            tmp += h_vec[j] * h_msk[j-lo];
        }

        if (tmp != h_res[i]) {
            printf("Incorrect: (tmp) %d != %d\n", tmp, h_res[i]);
        }
        assert(tmp == h_res[i]);
    }

    printf("All pass!\n");
}

int main() {
    int n = 1 << 20;
    int m = 7;
    int n_bytes = sizeof(int) * n;
    int m_bytes = sizeof(int) * m;

    int *h_vec = (int*) malloc(n_bytes);
    int *h_msk = (int*) malloc(m_bytes);
    int *h_res = (int*) malloc(n_bytes);

    int *d_vec, *d_msk, *d_res;
    cudaMalloc(&d_vec, n_bytes);
    cudaMalloc(&d_msk, m_bytes);
    cudaMalloc(&d_res, n_bytes);

    for (int i = 0; i < n; i++) {
        h_vec[i] = rand() % 100;
        h_res[i] = 0;
    }
    for (int i = 0; i < m; i++) {
        h_msk[i] = rand() % 10;
    }

    cudaMemcpy(d_vec, h_vec, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msk, h_msk, m_bytes, cudaMemcpyHostToDevice);

    int num_threads = 1 << 8;
    dim3 blck_size(num_threads);
    dim3 grid_size((n + num_threads - 1) / num_threads);

    convolve_1d<<<grid_size, blck_size>>>(d_vec, d_msk, d_res, n, m);
    cudaMemcpy(h_res, d_res, n_bytes, cudaMemcpyDeviceToHost);
    verify_result(h_vec, h_msk, h_res, n, m);

    cudaFree(d_vec);
    cudaFree(d_msk);
    cudaFree(d_res);

    free(h_vec);
    free(h_msk);
    free(h_res);

    printf("Success!");
    return 0;
}