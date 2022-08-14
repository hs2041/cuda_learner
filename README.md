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

1. Cooperative groups (## Read about this one)
2. cudastreams (## Read about this one)
3. Unified memory and prefetch
4. clock() for profiling

## Concurrency fundamental to CUDA

1. Multiple concurrent kernel execution is possible (if one kernel is not fully utilitising gpu)
2. Code execution on host can be done in parallel to kernel execution.
3. Memory transfer can be handled such that the processors are busy with computation at the time of transfer 
4. 

### Memory access
Adjacent memory access can be coalesced into a single wide access.

# Execercises To Do

1. Transpose a matrix using a single matrix argument
2. 

