#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void block_reduce(const float *a, float *partial, int n);

float cpu_sum(const float *a, int n);

int main() {
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);
    float *h_a = new float[N];
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;  // simple test data

    float *d_a, *d_partial;
    cudaMalloc(&d_a, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_partial, blocks * sizeof(float));

    // --- GPU timed sum ---
    auto gpu_start = std::chrono::high_resolution_clock::now();
    block_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_partial, N);

    float *h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_total = 0.0f;
    for (int i = 0; i < blocks; ++i) gpu_total += h_partial[i];
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    // --- CPU timed sum ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_total = cpu_sum(h_a, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "CPU sum: " << cpu_total << " (" << cpu_time << " ms)\n";
    std::cout << "GPU sum: " << gpu_total << " (" << gpu_time << " ms)\n";

    cudaFree(d_a);
    cudaFree(d_partial);
    delete[] h_a;
    delete[] h_partial;
    return 0;
}
