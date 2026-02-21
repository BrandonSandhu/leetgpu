#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Check we are within allocated memory 4 bytes per pixel
    if (i < width*height*4) {
        // Skip every alpha key
        if (i % 4 != 3) {
            image[i] = 255 - image[i];
        }
    }
    
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    // Make sure you make sufficient number of threads for bytes
    int blocksPerGrid = (4 * width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
