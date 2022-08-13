# cuda_learner
 
## Cuda Architecture
1. Shared memory across gpu 
2. Block level memory  
3. Invdividual cache for each thread 
1. 

Comiler used: nvcc

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

