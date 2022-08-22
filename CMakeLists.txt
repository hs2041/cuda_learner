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
target_link_libraries(3_vector_addition -lcublas -lcurand) # Linking cublas library

add_executable(4_cuda_info src/4_cuda_info.cu)
target_link_libraries(4_cuda_info ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)

option(VIDEOS_ENABLED "Demo of videos" ON)

if(VIDEOS_ENABLED)

    add_executable(1_add_unified src/videos/1_add_unified.cu)
    target_link_libraries(1_add_unified ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(1_add_unified -lcublas -lcurand) # Linking cublas library

    add_executable(2_mat_mul src/videos/2_mat_mul.cu)
    target_link_libraries(2_mat_mul ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(2_mat_mul -lcublas -lcurand) # Linking cublas library

    add_executable(3_tiled_mat_mul src/videos/3_tiled_mat_mul.cu)
    target_link_libraries(3_tiled_mat_mul ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(3_tiled_mat_mul -lcublas -lcurand) # Linking cublas library
    
    add_executable(4_cublas_add src/videos/4_cublas_add.cu)
    target_link_libraries(4_cublas_add ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(4_cublas_add -lcublas -lcurand) # Linking cublas library

    add_executable(5_cublas_matmul src/videos/5_cublas_matmul.cu)
    target_link_libraries(5_cublas_matmul ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(5_cublas_matmul -lcublas -lcurand) # Linking cublas library

    add_executable(6_reduction_base src/videos/6_reduction_base.cu)
    target_link_libraries(6_reduction_base ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(6_reduction_base -lcublas -lcurand) # Linking cublas library

    add_executable(6_reduction_2 src/videos/6_reduction_2.cu)
    target_link_libraries(6_reduction_2 ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(6_reduction_2 -lcublas -lcurand) # Linking cublas library

    add_executable(6_reduction_3 src/videos/6_reduction_3.cu)
    target_link_libraries(6_reduction_3 ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(6_reduction_3 -lcublas -lcurand) # Linking cublas library

    add_executable(6_reduction_4_coop src/videos/6_reduction_4_coop.cu)
    target_link_libraries(6_reduction_4_coop ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(6_reduction_4_coop -lcublas -lcurand) # Linking cublas library

    add_executable(7_cooperative_groups src/videos/7_cooperative_groups.cu)
    target_link_libraries(7_cooperative_groups ${OpenCL_LIBRARY} ${GTEST_LIBRARIES} pthread)
    target_link_libraries(7_cooperative_groups -lcublas -lcurand) # Linking cublas library

endif(VIDEOS_ENABLED)