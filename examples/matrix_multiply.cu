#include "vgpu/api.hpp"
#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matrixMultiply(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;  // Matrix size
    const size_t size = N * N * sizeof(float);

    // Create host matrices
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialize matrices...

    try {
        // Request a virtual GPU with 1GB memory
        auto gpu = vgpu::GPU::create(1024 * 1024 * 1024);
        if (!gpu) {
            std::cerr << "Failed to create virtual GPU\n";
            return 1;
        }

        // Allocate device memory
        float *d_A, *d_B, *d_C;
        gpu->allocateBuffer(&d_A, N * N);
        gpu->allocateBuffer(&d_B, N * N);
        gpu->allocateBuffer(&d_C, N * N);

        // Copy data to device
        gpu->copyToDevice(d_A, h_A, N * N);
        gpu->copyToDevice(d_B, h_B, N * N);

        // Launch kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                    (N + blockDim.y - 1) / blockDim.y);
        
        gpu->launchKernel(matrixMultiply, gridDim, blockDim, 
                         d_A, d_B, d_C, N);

        // Copy result back
        gpu->copyToHost(h_C, d_C, N * N);

        // Cleanup
        gpu->freeBuffer(d_A);
        gpu->freeBuffer(d_B);
        gpu->freeBuffer(d_C);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // Cleanup host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
} 