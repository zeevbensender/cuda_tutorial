#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void block_reduce(const float *a, float *partial, int n);

float cpu_sum(const float *a, int n);

int main() {
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float); 
    float *h_a = new float[N]; //Local buffer
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;  // simple test data: 1M floats each float is 1.0

    float *d_a, *d_partial; //pointers to GPU buffers
    cudaMalloc(&d_a, size); //allocate memory for the buffer of size 1M floats on GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); // Copy data from the local buffer to the GPU buffer

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2); //The number of thread blocks in the grid.
    cudaMalloc(&d_partial, blocks * sizeof(float)); //Allocate memory on GPU to write calculation results of each block

    // --- GPU timed sum ---
    auto gpu_start = std::chrono::high_resolution_clock::now(); //Start time
    //Run calculations on GPU
    block_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_partial, N);

    float *h_partial = new float[blocks]; //Buffer to copy calculation results from GPU
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_total = 0.0f;
    for (int i = 0; i < blocks; ++i) gpu_total += h_partial[i]; //Summarize the calculations performed on GPU
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count(); //Calculate duration

    // --- CPU timed sum ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_total = cpu_sum(h_a, N); //Perform calcualtions on the host
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count(); //Calculate duration

    std::cout << "CPU sum: " << cpu_total << " (" << cpu_time << " ms)\n";
    std::cout << "GPU sum: " << gpu_total << " (" << gpu_time << " ms)\n";

    cudaFree(d_a);
    cudaFree(d_partial);
    delete[] h_a;
    delete[] h_partial;
    return 0;
}
