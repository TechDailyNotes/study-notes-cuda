#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int NUM_CHARS = 26;

ostream& operator<< (ostream& os, vector<char>& vec) {
    for (char c: vec) os << c << endl;
    return os;
}

__global__ void histogram(char *d_elements, int *d_bins, int num_elements, int num_bins) {
    extern __shared__ int s_memory[];

    int l_tid = threadIdx.x;
    int g_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_l_threads = blockDim.x;
    int num_g_threads = gridDim.x * blockDim.x;

    // Step 1: Initialize initial values of all the bins as 0.
    for (int i = l_tid; i < num_bins; i += num_l_threads) s_memory[i] = 0;
    __syncthreads();

    // Step 2: Aggregate elements in the same range to the same bin and
    //         Compute the number of values in all the bins.
    for (int i = g_tid; i < num_elements; i += num_g_threads) {
        int idx_bin = (d_elements[i] - 'a') % num_bins;
        atomicAdd(&s_memory[idx_bin], 1);
    }
    __syncthreads();

    // Step 3: Move the number of values from the shared memory to the global memory.
    for (int i = l_tid; i < num_bins; i += num_l_threads) {
        atomicAdd(&d_bins[i], s_memory[i]);
    }
}

int main() {
    int num_elements = 1 << 20;
    int num_bins = 7;

    size_t byte_elements = sizeof(char) * num_elements;
    size_t byte_bins = sizeof(int) * num_bins;

    vector<char> h_elements(num_elements);
    vector<int> h_bins(num_bins);

    srand(1);
    generate(begin(h_elements), end(h_elements), [](){return 'a' + rand() % NUM_CHARS;});
    // cout << h_elements;

    char *d_elements;
    int *d_bins;
    cudaMalloc(&d_elements, byte_elements);
    cudaMalloc(&d_bins, byte_bins);
    cudaMemcpy(d_elements, h_elements.data(), byte_elements, cudaMemcpyHostToDevice);

    int num_threads = 1 << 8;
    int num_blocks = (num_elements + num_threads - 1) / num_threads;
    int scale_grid = 1 << 5;
    dim3 size_block(num_threads);
    dim3 size_grid(num_blocks / scale_grid);
    size_t size_cache = byte_bins;
    histogram<<<size_grid, size_block, size_cache>>>(d_elements, d_bins, num_elements, num_bins);

    cudaMemcpy(h_bins.data(), d_bins, byte_bins, cudaMemcpyDeviceToHost);
    assert(num_elements == accumulate(begin(h_bins), end(h_bins), 0));

    ofstream f_bins;
    f_bins.open("histogram.dat", ios::out | ios::trunc);
    for (int i: h_bins) f_bins << i << endl;
    f_bins.close();

    cudaFree(d_elements);
    cudaFree(d_bins);

    cout << "[int main()] pass!" << endl;
    return 0;
}