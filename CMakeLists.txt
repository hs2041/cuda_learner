cmake_minimum_required(VERSION 3.0.0)
project(cuda_lessons VERSION 0.1.0)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})

# message("${OpenCL_LIBRARY}")

add_executable(cuda_lessons main.cpp)
