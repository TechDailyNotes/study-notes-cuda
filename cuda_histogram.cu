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

ostream& operator<< (ostream& os, vector& vec) {
    for (char c: vec) os << c << endl;
    return os;
}

__global__ void histogram(char *d_elements, int *d_bins, int num_elements, int num_bins) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < num_elements; i += gridDim.x * blockDim.x) {
        int idx_bin = (d_elements[i] - 'a') % num_bins;
        atomicAdd(&d_bins[idx_bin], 1);
    }
}

int main() {
    int num_elements = 1 << 20;
    int num_bins = 7;

    size_t byte_elements = sizeof(char) * num_elements;
    size_t byte_bins = sizeof(int) * num_bins;

    vector h_elements(num_elements);
    vector h_bins(num_bins);

    srand(1);
    generate(
        begin(h_elements), end(h_elements),
        [](){return 'a' + rand() % NUM_CHARS;}
    );
    // cout << h_elements;

    char *d_elements;
    int *d_bins;

    cudaMalloc(&d_elements, byte_elements);
    cudaMalloc(&d_bins, byte_bins);
    cudaMemcpy(d_elements, h_elements.data(), byte_elements, cudaMemcpyHostToDevice);

    int num_threads = 1 << 8;
    dim3 size_block(num_threads);
    dim3 size_grid((num_elements + num_threads - 1) / num_threads);
    histogram<<>>(d_elements, d_bins, num_elements, num_bins);

    cudaMemcpy(h_bins.data(), d_bins, byte_bins, cudaMemcpyDeviceToHost);
    assert(num_elements == accumulate(begin(h_bins), end(h_bins), 0));

    ofstream f_output;
    f_output.open("histogram.dat", ios::out | ios::trunc);
    for (int i = 0; i < num_bins; i++) {
        f_output << "h_output[" << i << "] = " << h_bins[i] << endl;
    }
    f_output.close();

    cudaFree(d_elements);
    cudaFree(d_bins);

    cout << "Success!" << endl;

    return 0;
}
