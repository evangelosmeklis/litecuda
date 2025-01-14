#include "vgpu/scheduler.hpp"
#include <chrono>

namespace vgpu {

Scheduler::Scheduler() : running_(false) {}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        running_ = true;
        scheduler_thread_ = std::thread(&Scheduler::schedulerLoop, this);
    }
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    cv_.notify_one();
    
    if (scheduler_thread_.joinable()) {
        scheduler_thread_.join();
    }
}

void Scheduler::submitTask(VirtualGPU* vgpu, std::function<void()> task, int priority) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        task_queue_.push(Task{vgpu, task, priority});
    }
    cv_.notify_one();
}

void Scheduler::waitForCompletion(VirtualGPU* vgpu) {
    cudaStreamSynchronize(vgpu->getStream());
}

void Scheduler::schedulerLoop() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { 
                return !running_ || !task_queue_.empty(); 
            });
            
            if (!running_ && task_queue_.empty()) {
                break;
            }
            
            if (!task_queue_.empty()) {
                task = task_queue_.top();
                task_queue_.pop();
            }
        }
        
        if (task.work) {
            task.work();
        }
    }
}

} // namespace vgpu 