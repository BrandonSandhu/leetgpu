#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    // In this kernel CUDA gives me access to threadIdx.x, blockIdx.x, blockDim.x, gridDim.x
    // we can compute our unique global thread id using 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // suppose threadsPerBlock = 256, blocksPerGrid = 4 (total threads for kernel is 1024)
    // then block 0: threadIdx = 5 => i = 5 and block 1: threadIdx = 5 => 256 + 5 = 261
    
    // Each thread runs this kernel function once therefore we are writing the computation for one thread
    if (i < N) { // This ensures we are not causing illegal memory access due to integer divison launching extra threads 
        C[i] = A[i] + B[i];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    // C integer division takes care of ensuring this is an integer
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // This says run the kernel vector_add across blocksPerGrid blocks and each block having threadsPerBlock threads
    // Maximum 1024 threads per block
    // Hierarchy => Grid, blocks, threads
    // Each kernel launch is exactly one grid
    // A, B, C, N are params passed to the GPU
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
