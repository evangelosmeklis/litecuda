cmake_minimum_required(VERSION 3.15)
project(gpu_virtualization CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/memory_manager.cpp
    src/virtual_gpu.cpp
    src/scheduler.cpp
    src/vgpu_manager.cpp
)

# Create library
add_library(vgpu STATIC ${SOURCES})
target_link_libraries(vgpu ${CUDA_LIBRARIES})

# Examples
add_executable(matrix_multiply examples/matrix_multiply.cu)
target_link_libraries(matrix_multiply vgpu)

# For VSCode, create a compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 