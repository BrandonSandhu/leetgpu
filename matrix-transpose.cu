#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    // These are (col, row) of the original matrix A
    int currentCol = blockDim.x * blockIdx.x + threadIdx.x;
    int currentRow = blockDim.y * blockIdx.y + threadIdx.y;

    // Using (col, row) of matrix A to determine global thread id of new tranposed matrix
    // (col, row) -> (row, col) -> rows * currentCol + currentRow
    if (currentCol < cols & currentRow < rows) {
        output[rows * currentCol + currentRow] = input[cols * currentRow + currentCol];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
