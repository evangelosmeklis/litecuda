#include "vgpu/vgpu_manager.hpp"
#include <cuda_runtime.h>

namespace vgpu {

VGPUManager& VGPUManager::getInstance() {
    static VGPUManager instance;
    return instance;
}

VGPUManager::VGPUManager() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Using first GPU for simplicity
    
    // Initialize with 80% of available GPU memory
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    memory_manager_ = std::make_unique<MemoryManager>(free_memory * 0.8);
    scheduler_ = std::make_unique<Scheduler>();
    scheduler_->start();
}

VGPUManager::~VGPUManager() {
    scheduler_->stop();
}

std::shared_ptr<VirtualGPU> VGPUManager::createVirtualGPU(size_t memory_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (memory_size > memory_manager_->getAvailableMemory()) {
        return nullptr;
    }
    
    auto vgpu = std::make_shared<VirtualGPU>(memory_size);
    active_vgpus_.push_back(vgpu);
    return vgpu;
}

void VGPUManager::destroyVirtualGPU(std::shared_ptr<VirtualGPU> vgpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = std::find(active_vgpus_.begin(), active_vgpus_.end(), vgpu);
    if (it != active_vgpus_.end()) {
        active_vgpus_.erase(it);
    }
}

size_t VGPUManager::getTotalGPUMemory() const {
    return memory_manager_->getTotalMemory();
}

size_t VGPUManager::getAvailableGPUMemory() const {
    return memory_manager_->getAvailableMemory();
}

int VGPUManager::getNumActiveVGPUs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(active_vgpus_.size());
}

} // namespace vgpu 