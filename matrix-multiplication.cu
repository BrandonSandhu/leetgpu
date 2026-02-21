#include <cuda_runtime.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N,
                                             int K) {
    // Computing global thread index for resulting matrix entry (i, j) in M x K matrix
    // We think of it as a 2D grid, compute the resulting matrix (i, j) using blocks in X, Y. Then compute global position in C using (i, j) by
    // Matrices are row-major arrays
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    if (row < M & col < K) {
        float total = 0;
        for (int j = 0; j < N; j++) {
            total += A[row * N + j] * B[j*K + col];
        }
        C[row * K + col] = total;
    }
    
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // dim3 is used for 2D or 3D thread blocks
    // Using powers of 2 helps hardware schedule warps effectively
    // Resulting matrix is (M x K) so we make a thread per resulting entry

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
