#include "vgpu/memory_manager.hpp"
#include <stdexcept>

namespace vgpu {

MemoryManager::MemoryManager(size_t total_gpu_memory) 
    : total_memory_(total_gpu_memory)
    , available_memory_(total_gpu_memory)
    , base_ptr_(nullptr) {
    // Allocate the total GPU memory pool
    cudaError_t err = cudaMalloc(&base_ptr_, total_memory_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory pool: " + 
                               std::string(cudaGetErrorString(err)));
    }
}

MemoryManager::~MemoryManager() {
    if (base_ptr_) {
        cudaFree(base_ptr_);
    }
}

cudaError_t MemoryManager::allocateChunk(size_t size, void** ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (size > available_memory_) {
        return cudaErrorMemoryAllocation;
    }

    // Calculate the offset from base pointer
    size_t offset = total_memory_ - available_memory_;
    *ptr = static_cast<char*>(base_ptr_) + offset;

    available_memory_ -= size;
    allocated_chunks_[*ptr] = size;
    return cudaSuccess;
}

cudaError_t MemoryManager::freeChunk(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_chunks_.find(ptr);
    if (it == allocated_chunks_.end()) {
        return cudaErrorInvalidValue; // Handle invalid free attempts
    }

    available_memory_ += it->second;
    allocated_chunks_.erase(it);
    return cudaSuccess;
}

size_t MemoryManager::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_memory_;
}

size_t MemoryManager::getTotalMemory() const {
    return total_memory_;
}

}