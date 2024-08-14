#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolve(
    int *d_vec, int *d_msk, int *d_res,
    int m_numElementsVec, int m_numElementsMsk
) {
    // Step 0: Get the thread index.
    int g_ti = blockIdx.x * blockDim.x + threadIdx.x;
    // if (g_ti >= m_numElementsVec) return;

    // Step 1: Compute the convolved result of the current grid.
    int sum = 0;
    int radius = m_numElementsMsk / 2;

    for (int mi = 0; mi < m_numElementsMsk; mi++) {
        int vi = g_ti - radius + mi;
        if (vi >= 0 && vi < m_numElementsVec) {
            sum += d_vec[vi] * d_msk[mi];
        }
    }

    // Step 2: Register the result back to the vector.
    d_res[g_ti] = sum;
}

void m_init(int *m_array, int m_size) {
    for (int i = 0; i < m_size; i++) {
        m_array[i] = rand() % 100;
    }
}

void verify(
    int *h_vec, int *h_msk, int *h_res,
    int m_numElementsVec, int m_numElementsMsk
) {
    int radius = m_numElementsMsk / 2;

    for (int ri = 0; ri < m_numElementsVec; ri++) {
        int sum = 0;
        for (int mi = 0; mi < m_numElementsMsk; mi++) {
            int vi = ri - radius + mi;
            if (vi >= 0 && vi < m_numElementsVec) {
                sum += h_vec[vi] * h_msk[mi];
            }
        }
        assert(sum == h_res[ri]);
    }
}

int main() {
    // Step 0: Set up parameters.
    int m_numElementsVec = 1 << 20;
    int m_numElementsMsk = 7;
    size_t m_numBytesVec = sizeof(int) * m_numElementsVec;
    size_t m_numBytesMsk = sizeof(int) * m_numElementsMsk;

    int d_blockDimX = 1 << 8;
    int d_gridDimX = (int) ceil(1.0 * m_numElementsVec / d_blockDimX);

    // Step 1: Init memories on both cpu and gpu.
    int *h_vec = new int[m_numElementsVec];
    int *h_msk = new int[m_numElementsMsk];
    int *h_res = new int[m_numElementsVec];
    m_init(h_vec, m_numElementsVec);
    m_init(h_msk, m_numElementsMsk);

    int *d_vec, *d_msk, *d_res;
    cudaMalloc(&d_vec, m_numBytesVec);
    cudaMalloc(&d_msk, m_numBytesMsk);
    cudaMalloc(&d_res, m_numBytesVec);

    // Step 2: Launch the kernel function to convolve the vector with the mask.
    cudaMemcpy(d_vec, h_vec, m_numElementsVec, cudaMemcpyHostToDevice);
    cudaMemcpy(d_msk, h_msk, m_numElementsMsk, cudaMemcpyHostToDevice);
    convolve<<<d_gridDimX, d_blockDimX>>>(
        d_vec, d_msk, d_res,
        m_numElementsVec, m_numElementsMsk
    );
    cudaMemcpy(h_res, d_res, m_numElementsVec, cudaMemcpyDeviceToHost);

    verify(h_vec, h_msk, h_res, m_numElementsVec, m_numElementsMsk);

    // Step 3: Clear memories.
    delete[] h_vec;
    delete[] h_msk;
    delete[] h_res;
    cudaFree(d_vec);
    cudaFree(d_msk);
    cudaFree(d_res);

    printf("Success!");
    return 0;
}