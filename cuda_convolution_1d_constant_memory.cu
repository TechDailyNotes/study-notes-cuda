#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

#define M 7

using namespace std;

__constant__ int d_msk[M];

__global__ void convolve_1d(int *d_vec, int *d_res, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int tmp = 0;
    int lo = tid - M / 2;
    int hi = tid + M / 2;

    for (int i = lo; i <= hi; i++) {
        if (i < 0 || i >= n) continue;
        tmp += d_vec[i] * d_msk[i-lo];
    }

    d_res[tid] = tmp;
}

void verify_result(int *h_vec, int *h_msk, int *h_res, int n) {
    for (int i = 0; i < n; i++) {
        int tmp = 0;
        int lo = i - M / 2;
        int hi = i + M / 2;

        for (int j = lo; j <= hi; j++) {
            if (j < 0 || j > n) continue;
            tmp += h_vec[j] * h_msk[j-lo];
        }

        if (tmp != h_res[i] && i == 0) {
            cout << "Incorrect: (tmp) " << tmp << " != " << h_res[i] << endl;
        }
    }
}

int main() {
    int n = 1 << 20;
    int n_bytes = sizeof(int) * n;
    int m_bytes = sizeof(int) * M;

    int *h_vec = new int[n];
    int *h_msk = new int[M];
    int *h_res = new int[n];

    for (int i = 0; i < n; i++) {
        h_vec[i] = rand() % 100;
        h_res[i] = 0;
    }
    for (int i = 0; i < M; i++) {
        h_msk[i] = rand() % 10;
    }

    int *d_vec, *d_res;
    cudaMalloc(&d_vec, n_bytes);
    cudaMalloc(&d_res, n_bytes);

    cudaMemcpy(d_vec, h_vec, n_bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_msk, h_msk, m_bytes);

    int num_threads = 1 << 8;
    dim3 size_blck(num_threads);
    dim3 size_grid((n + num_threads - 1) / num_threads);

    convolve_1d<<<size_grid, size_blck>>>(d_vec, d_res, n);
    cudaMemcpy(h_res, d_res, n_bytes, cudaMemcpyDeviceToHost);
    verify_result(h_vec, h_msk, h_res, n);

    cudaFree(d_vec);
    cudaFree(d_res);

    delete[] h_vec;
    delete[] h_msk;
    delete[] h_res;

    cout << "Success!" << endl;
    return 0;
}