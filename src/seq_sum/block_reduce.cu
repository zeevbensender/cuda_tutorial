#include <cstdio>

__global__ void block_reduce(const float *a, float *partial, int n) {
    // Shared memory buffer for this thread block (allocated dynamically at kernel launch)
    extern __shared__ float sdata[];

    // Thread index within the block (0 ... blockDim.x-1)
    unsigned int tid = threadIdx.x;

    // Compute global index 'i' — the position in the full array 'a' this thread starts from.
    // Each block handles a continuous 2*blockDim.x chunk of the array.
    // Each thread loads up to two elements from that chunk.
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Each thread accumulates its local partial sum into 'sum'
    float sum = 0.0f;

    // Load the first element assigned to this thread (if it exists)
    if (i < n)
        sum += a[i];

    // Load the second element, one blockDim.x away (if it exists)
    // → This doubles throughput: each thread reads 2 elements instead of 1.
    if (i + blockDim.x < n)
        sum += a[i + blockDim.x];

    // Store the thread’s partial result into shared memory
    sdata[tid] = sum;

    // Synchronize all threads in the block
    // Ensures all sdata[] values are written before any thread starts reading them
    __syncthreads();

    // Parallel reduction in shared memory:
    // In each iteration, half of the threads become inactive,
    // and the remaining ones add their “partner”’s value to their own slot.
    //
    // Example: with 8 threads
    //  Step 1: thread0+=thread4, thread1+=thread5, ...
    //  Step 2: thread0+=thread2, thread1+=thread3, ...
    //  Step 3: thread0+=thread1  → sdata[0] now holds block sum.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            printf("\n\nThread ID: %u; Block index: %u; Block dimension: %u; \nif (tid < s): if (%u < %u); ==> sdata[tid] += sdata[tid + s] ==> sdata[%u] += sdata[%u + %u] ==> %f += %f\n",
            tid, blockIdx.x, blockDim.x, tid, s, tid, tid, s, sdata[tid], sdata[tid + s]);
            sdata[tid] += sdata[tid + s];  // Each active thread adds its partner’s value
        } else {
            printf("\n\nThread ID: %u; Block index: %u; Block dimension: %u; \nif (tid < s): if (%u < %u); ==> DO NOTHING",
            tid, blockIdx.x, blockDim.x, tid, s);
            sdata[tid] += sdata[tid + s];  // Each active thread adds its partner’s value
        }
        __syncthreads();                   // Wait until all updates are complete
        printf("Threads synchronised. TID: %u\n\n", tid);
    }

    // After the loop, thread 0 holds the total sum for this block
    // Write it to the global array 'partial' — one float per block
    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}
