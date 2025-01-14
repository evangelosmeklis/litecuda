#include "vgpu/virtual_gpu.hpp"
#include <stdexcept>

namespace vgpu {

VirtualGPU::VirtualGPU(size_t memory_size, int device_id)
    : total_memory_(memory_size)
    , used_memory_(0)
    , device_id_(device_id)
    , memory_start_(nullptr) {
    
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + 
                               std::string(cudaGetErrorString(err)));
    }
    
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + 
                               std::string(cudaGetErrorString(err)));
    }
}

VirtualGPU::~VirtualGPU() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

cudaError_t VirtualGPU::allocateMemory(void** ptr, size_t size) {
    if (used_memory_ + size > total_memory_) {
        return cudaErrorMemoryAllocation;
    }
    
    cudaError_t err = cudaMalloc(ptr, size);
    if (err == cudaSuccess) {
        used_memory_ += size;
    }
    return err;
}

cudaError_t VirtualGPU::freeMemory(void* ptr) {
    // Note: In a real implementation, we'd want to track the size of each allocation
    cudaError_t err = cudaFree(ptr);
    return err;
}

cudaError_t VirtualGPU::memcpyHostToDevice(void* dst, const void* src, size_t count) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream_);
}

cudaError_t VirtualGPU::memcpyDeviceToHost(void* dst, const void* src, size_t count) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream_);
}

size_t VirtualGPU::getAvailableMemory() const {
    return total_memory_ - used_memory_;
}

size_t VirtualGPU::getTotalMemory() const {
    return total_memory_;
}

int VirtualGPU::getDeviceId() const {
    return device_id_;
}

cudaStream_t VirtualGPU::getStream() const {
    return stream_;
}

} // namespace vgpu 