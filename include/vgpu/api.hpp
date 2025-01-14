#pragma once

#include "vgpu/vgpu_manager.hpp"
#include <functional>
#include <memory>

namespace vgpu {

// Main API class for users
class GPU {
public:
    // Get a virtual GPU instance
    static std::shared_ptr<GPU> create(size_t memory_size) {
        auto& manager = VGPUManager::getInstance();
        auto vgpu = manager.createVirtualGPU(memory_size);
        if (!vgpu) return nullptr;
        return std::make_shared<GPU>(vgpu);
    }

    // Memory management
    template<typename T>
    bool allocateBuffer(T** ptr, size_t count) {
        return vgpu_->allocateMemory((void**)ptr, count * sizeof(T)) == cudaSuccess;
    }

    template<typename T>
    bool freeBuffer(T* ptr) {
        return vgpu_->freeMemory(ptr) == cudaSuccess;
    }

    // Data transfer
    template<typename T>
    bool copyToDevice(T* dst, const T* src, size_t count) {
        return vgpu_->memcpyHostToDevice(dst, src, count * sizeof(T)) == cudaSuccess;
    }

    template<typename T>
    bool copyToHost(T* dst, const T* src, size_t count) {
        return vgpu_->memcpyDeviceToHost(dst, src, count * sizeof(T)) == cudaSuccess;
    }

    // Kernel execution
    template<typename Func, typename... Args>
    bool launchKernel(Func kernel, dim3 gridDim, dim3 blockDim, Args... args) {
        return vgpu_->launchKernel(kernel, gridDim, blockDim, args...) == cudaSuccess;
    }

    // Resource info
    size_t getAvailableMemory() const { return vgpu_->getAvailableMemory(); }
    size_t getTotalMemory() const { return vgpu_->getTotalMemory(); }

private:
    explicit GPU(std::shared_ptr<VirtualGPU> vgpu) : vgpu_(vgpu) {}
    std::shared_ptr<VirtualGPU> vgpu_;
};

} // namespace vgpu 