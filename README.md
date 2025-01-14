
# Lightweight GPU Virtualization Framework

## Overview

This project is a lightweight C++ framework that simulates GPU virtualization on a single physical GPU. The framework enables multiple virtual "GPU instances" to share GPU resources such as memory and computation in an isolated manner. Each virtual instance can execute CUDA kernels independently, with a basic scheduler managing access to the GPU.

## Features
- **Context Isolation**: Isolate execution contexts for multiple virtual GPU instances using CUDA streams.
- **Simple Scheduler**: Implement a round-robin scheduler to manage time slices for each virtual GPU.
- **Memory Management**: Partition GPU memory into chunks and allocate each chunk to a virtual GPU instance.
- **Basic API**: Provide a C++ API for creating virtual GPU instances, submitting tasks, and managing memory.
- **Resource Monitoring**: Monitor GPU usage, including memory allocation and compute load.

## Technologies Used
- **CUDA**: For GPU programming and memory management.
- **C++**: For implementing the core framework and API.
- **CUDA Streams and Events**: For context isolation and task synchronization.

## How It Works
1. **Context Management**:
   - Each virtual GPU instance uses a unique CUDA stream for task isolation.
   - Memory allocation is managed through a custom allocator to ensure isolation between instances.

2. **Task Scheduling**:
   - A simple round-robin scheduler determines which virtual GPU gets access to the physical GPU.
   - Tasks are queued and executed in time slices.

3. **Memory Partitioning**:
   - GPU memory is divided into fixed-size chunks and assigned to virtual GPUs.
   - Optional shared memory regions can be configured for inter-instance communication.

4. **API**:
   - Create virtual GPU instances.
   - Allocate and deallocate memory.
   - Launch kernels with isolated execution contexts.

## Future Enhancements
- Priority-based and resource-aware scheduling.
- Multi-node support for distributed GPU virtualization.
- Integration with containerized environments (e.g., Docker, Kubernetes).

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the AGPL3 License.

## Originally written by Evangelos Meklis