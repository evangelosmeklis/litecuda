#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <mutex>

namespace vgpu {

class MemoryManager {
public:
    MemoryManager(size_t total_gpu_memory);
    ~MemoryManager();

    // Memory allocation and management
    cudaError_t allocateChunk(size_t size, void** ptr);
    cudaError_t freeChunk(void* ptr);
    
    // Memory tracking
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;

private:
    std::mutex mutex_;
    size_t total_memory_;
    size_t available_memory_;
    std::unordered_map<void*, size_t> allocated_chunks_;
    void* base_ptr_;
};

} // namespace vgpu 