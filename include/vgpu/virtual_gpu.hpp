#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace vgpu {

class VirtualGPU {
public:
    VirtualGPU(size_t memory_size, int device_id = 0);
    ~VirtualGPU();

    // Delete copy constructor and assignment operator
    VirtualGPU(const VirtualGPU&) = delete;
    VirtualGPU& operator=(const VirtualGPU&) = delete;

    // Basic operations
    cudaError_t allocateMemory(void** ptr, size_t size);
    cudaError_t freeMemory(void* ptr);
    cudaError_t memcpyHostToDevice(void* dst, const void* src, size_t count);
    cudaError_t memcpyDeviceToHost(void* dst, const void* src, size_t count);
    
    // Kernel execution
    template<typename Func, typename... Args>
    cudaError_t launchKernel(Func kernel, dim3 gridDim, dim3 blockDim, Args... args);

    // Status and info
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;
    int getDeviceId() const;
    cudaStream_t getStream() const;

private:
    cudaStream_t stream_;
    size_t total_memory_;
    size_t used_memory_;
    int device_id_;
    void* memory_start_;  // Base pointer for this vGPU's memory region
    std::unordered_map<void*, size_t> allocations_;  // Track memory allocations
};

} // namespace vgpu 