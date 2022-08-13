cmake_minimum_required(VERSION 3.0.0)
project(cuda_lessons VERSION 0.1.0)


# add_compile_options(-Wall -Wextra -pedantic -Werror)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

add_compile_options(-w)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-lineinfo")
# add_compile_options(-lineinfo)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

find_package(OpenCL REQUIRED)
find_package(GTest REQUIRED)

include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(gtest src/gtest.cpp)
target_link_libraries(gtest ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)

add_executable(3_vector_addition src/3_vector_addition.cu)
target_link_libraries(3_vector_addition ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
add_executable(4_cuda_info src/4_cuda_info.cu)
target_link_libraries(4_cuda_info ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)

