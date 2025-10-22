#include <stdio.h>
#define N 10000000

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; b[i] = 2.0f;
    }

    cudaMalloc(&d_a, sizeof(float) * N);
    cudaMalloc(&d_b, sizeof(float) * N);
    cudaMalloc(&d_out, sizeof(float) * N);

    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vector_add<<<numBlocks, blockSize>>>(d_out, d_a, d_b, N);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    float total = 0;
    for (int i = 0; i < N; i++)
        total += out[i];

    printf("Total value is: %f\n", total);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);
}
