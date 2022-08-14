

## Unified memory management

Unified Memory is a single memory address space accessible from any processor in a system. Unified Memory in CUDA makes memory acces easy by providing a single memory space accessible by all GPUs and CPUs in your system. To allocate data in unified memory, call cudaMallocManaged(), which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to cudaFree().

Just one more thing: the CPU needs to wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this one can call call cudaDeviceSynchronize() before doing the final error checking on the CPU.

## Time Profiling 

Use: clock_t clock() object inside kernel functions for profiling. 

Using cudamallocmanaged, memprefetchasync and cudadevicesynchronize 

## Tiled matrix multiplication

Use this [link](https://penny-xu.github.io/blog/tiled-matrix-multiplication) to understand