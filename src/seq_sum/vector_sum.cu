#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void block_reduce(const float *a, float *partial, int n);

float cpu_sum(const float *a, int n);

int main() {
    int N = 1 << 20;  // 1M elements (2^20)
    size_t size = N * sizeof(float); 
    float *h_a = new float[N];  // Host (CPU) buffer
    for (int i = 0; i < N; ++i) h_a[i] = 1.0f;  // Initialize all values to 1.0

    float *d_a, *d_partial;  // Device (GPU) pointers
    cudaMalloc(&d_a, size);  // Allocate memory for input vector on GPU of size 1M floats
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);  // Copy host data to GPU

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);  
    // Number of blocks needed so each thread processes ~2 elements.
    // The "*2" allows each thread to load and sum two elements at once for better efficiency.

    cudaMalloc(&d_partial, blocks * sizeof(float));  
    // Allocate memory on GPU for partial results — one float per block, 
    // because each block reduces its local segment into a single partial sum.

    // --- GPU timed sum ---
    auto gpu_start = std::chrono::high_resolution_clock::now();  // Start timer

    // Launch the kernel on the GPU
    // <<<blocks, threads, threads * sizeof(float)>>>
    // → grid of `blocks` blocks, each with `threads` threads,
    //   and `threads * sizeof(float)` bytes of dynamic shared memory.
    block_reduce<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_partial, N);

    float *h_partial = new float[blocks];  // Host buffer to receive partial results
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);  // Copy GPU partial sums to CPU

    float gpu_total = 0.0f;
    for (int i = 0; i < blocks; ++i)
        gpu_total += h_partial[i];  // Sum up block results on CPU to get total

    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();  // Measure elapsed GPU time (host-side)

    // --- CPU timed sum ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_total = cpu_sum(h_a, N);  // Perform same sum sequentially on CPU
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();  // Measure elapsed CPU time

    std::cout << "CPU sum: " << cpu_total << " (" << cpu_time << " ms)\n";
    std::cout << "GPU sum: " << gpu_total << " (" << gpu_time << " ms)\n";

    cudaFree(d_a);
    cudaFree(d_partial);
    delete[] h_a;
    delete[] h_partial;
    return 0;
}
