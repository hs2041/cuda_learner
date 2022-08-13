
## Cuda Architecture
1. Shared memory across gpu 
2. Block level memory  
3. Individual cache for each thread 
1. 

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


# Execercises To Do

1. Transpose a matrix using a single matrix argument
2. 

# Library Specific Features

1. Cooperative groups (## Read about this one)
