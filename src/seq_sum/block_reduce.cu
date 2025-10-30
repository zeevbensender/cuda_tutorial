__global__ void block_reduce(const float *a, float *partial, int n) {
    extern __shared__ float sdata[]; //Data shared between the threads of the current block
    unsigned int tid = threadIdx.x; //Current thread index (in this example it's either 0 or 1)
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x; //Calculate the the start index of the block range

    float sum = 0.0f;
    if (i < n) sum += a[i]; // Assign the value of the first member in the block range to sum if i is within the buffer range
    if (i + blockDim.x < n) sum += a[i + blockDim.x]; // Add the value of the last member in the block range to sum
    sdata[tid] = sum; //Assign sum value to the shared data member that belongs to the current thread
    __syncthreads(); //Synchronize threads (Wait until other threads of the block finish???)

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {//??
        if (tid < s)
            sdata[tid] += sdata[tid + s];//??
        __syncthreads(); //Synchronize threads (Wait until other threads of the block finish???)
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}
