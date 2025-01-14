#pragma once

#include "virtual_gpu.hpp"
#include "scheduler.hpp"
#include "memory_manager.hpp"
#include <memory>
#include <vector>

namespace vgpu {

class VGPUManager {
public:
    static VGPUManager& getInstance();

    // vGPU management
    std::shared_ptr<VirtualGPU> createVirtualGPU(size_t memory_size);
    void destroyVirtualGPU(std::shared_ptr<VirtualGPU> vgpu);

    // Resource monitoring
    size_t getTotalGPUMemory() const;
    size_t getAvailableGPUMemory() const;
    int getNumActiveVGPUs() const;

private:
    VGPUManager();
    ~VGPUManager();

    // Singleton pattern
    VGPUManager(const VGPUManager&) = delete;
    VGPUManager& operator=(const VGPUManager&) = delete;

    std::unique_ptr<MemoryManager> memory_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::vector<std::shared_ptr<VirtualGPU>> active_vgpus_;
    std::mutex mutex_;
};

} // namespace vgpu 