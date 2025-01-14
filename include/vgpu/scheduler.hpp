#pragma once

#include "virtual_gpu.hpp"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace vgpu {

struct Task {
    VirtualGPU* vgpu;
    std::function<void()> work;
    int priority;
    
    bool operator<(const Task& other) const {
        return priority < other.priority;
    }
};

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    void start();
    void stop();
    
    // Task management
    void submitTask(VirtualGPU* vgpu, std::function<void()> task, int priority = 0);
    void waitForCompletion(VirtualGPU* vgpu);

private:
    void schedulerLoop();
    
    std::priority_queue<Task> task_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool running_;
    std::thread scheduler_thread_;
};

} // namespace vgpu 