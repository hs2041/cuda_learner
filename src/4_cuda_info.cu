#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gtest/gtest.h>
#include <tuple>
#include <chrono>
#include <functional>
#include <chrono>
#include <string>

int main()
{
    int count;
    cudaGetDeviceCount(&count);
    std::cout << "Device count: " << count << std::endl;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "threads per block: " << props.maxThreadsPerBlock << std::endl;
    std::cout << "multi-processor count: " << props.multiProcessorCount << std::endl;
    std::cout << "clock rate: " << props.clockRate << std::endl;

    std::cout << "max thread dim 0: " << props.maxThreadsDim[0] << std::endl;
    std::cout << "max thread dim 1: " << props.maxThreadsDim[1] << std::endl;
    std::cout << "max thread dim 2: " << props.maxThreadsDim[2] << std::endl;

    std::cout << "max grid dim 0: " << props.maxGridSize[0] << std::endl;
    std::cout << "max grid dim 1: " << props.maxGridSize[1] << std::endl;
    std::cout << "max grid dim 2: " << props.maxGridSize[2] << std::endl;

    return 0;

}