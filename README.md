# CUDA LEARNING
 These are my notes while learning CUDA  



## Table of Contents

1. [Introduction](doc/1_intro.md)
2. [Basic Concepts](doc/2_concepts.md)
3. [Driver info](doc/3_info.md)
4. [CUDA Libraries](doc/4_libraries.md)
## Cuda Architecture
1. Shared memory across gpu 
2. Block level memory  
3. Individual cache for each thread 
4. 

Compiler used: nvcc

Threads are in a block which are in a grid 
Inter-block syncing is possible but intra-block is hard 

Streaming Multiprocessors (SMs)

Compute capability versions (3.0 is common version)

SIMT architecture (Single Instance multiple threads)
Each thread runs the same instruction

Thread execution is done in groups: __Warps__ which is a group of 32 threads 

## Running a kernel 

* Blocks are assigned to available SMs 
* Each block is split into warps which are scheduled and run on SMs
* Multiple warps/blocks run on each SM


## Performance optimization

### Visual Profiler 

__nvvp__ is the official profiler for CUDA programming. To enable profiling of your executable, compile it with the *lineinfo* flag then use the nvprof executable for profiling.

### Shared Memory model
1. Global device memory  
2. Memory shared across SM 
3. Threa specific memory (cache)
4. Block specific memory ( \__shared__ )
5. 

## Library Specific Features

1. cudastreams (## Read about this one)
2. Cooperative groups allow you to implement synchronisation at multiple levels (https://developer.nvidia.com/blog/cooperative-groups/)
3. Unified memory and prefetch
4. clock() for profiling (You can find how much time each individual gpu core takes for execution)
5. CudaEvents  

## Concurrency fundamental to CUDA

1. Multiple concurrent kernel execution is possible (if one kernel is not fully utilitising gpu)
2. Code execution on host can be done in parallel to kernel execution.
3. Memory transfer can be handled such that the processors are busy with computation at the time of transfer 
4. Note that modulo operator is expensive, try to avoid using it
5. Shared memory bank conflicts
6. Race condition
7. pragma unroll command in the kernel allows you to unroll an entire loop into a set of seperate instructions. This allows you to get rid of the loop check and variable increment condition.
8. Concurrent writes are not directly possible in CUDA. Atmoic functions can be used for such cases. (https://stackoverflow.com/questions/11773141/what-are-all-the-atomic-operations-in-cuda). There is a compare and swap atomic operation which one can use to update the value of a variable.
9. The concept of bank conflict (https://stackoverflow.com/questions/3841877/what-is-a-bank-conflict-doing-cuda-opencl-programming)
10. Understand SIMT properly. A single instruction for a warp is called (a set of 32 threads) together.
11. Constant memory in GPU cache using the \__constant__ keyword and cudamemcopytoSymbol
12. Dynamic allocated arrays in GPU shared memory (https://stackoverflow.com/questions/5531247/allocating-shared-memory)



## Handling non-perfect input size 
Using a simple condition is the easiest way to get rid of this. Padding is another option. 
### Memory access
Adjacent memory access can be coalesced into a single wide access.

## OpenACC
It can easily transform the code from one GPU make to another (AMD and nvidia). The code style used is similar to OpenMP (pragmas). Basically, it looks like OpenMP for GPU programming. OpenACC is generally used with a PGI compiler. 

## Optimization
Optimizing the code considering the principles of GPU programming (SIMT, memory transfer, memory access, synchronisation, register and shared memory access, grid and block size, SIMT calls are based on warp size, prefetch memory, loop unrolling, avoid accessing same memory multiple times, ) in mind leads to better result.

## Excercises To Do

1. Transpose a matrix using a single matrix argument
2. 

