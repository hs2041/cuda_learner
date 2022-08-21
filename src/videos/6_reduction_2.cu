// This program performs sum reduction with an optimization
// that removes shared memory bank conflicts
// By: Nick from CoffeeBeforeArch

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <chrono>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int *v, int *v_r) {
  __shared__ int partial_sum[SHMEM_SIZE];

	// Calculate thread ID
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// // Load elements AND do first add of reduction
	// // Vector now 2x as long as number of threads, so scale i
	// int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	// // Store first partial result instead of just the elements
	partial_sum[threadIdx.x] = v[tid];
	//  + v[i + blockDim.x];
	__syncthreads();


	// for(int i = 1; i <= blockDim.x; i = i*2)
	// // Start at 1/2 block stride and divide by two each iteration
	for (int s = blockDim.x / 2; s > 0; s = s/2) {
	// 	// Each thread does work unless it is further than the stride
		if (threadIdx.x < s) {
			partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
		}
		__syncthreads();
	}

	// Let the thread 0 for this block write it's result to main memory
	// Result is inexed by this block
	if (threadIdx.x == 0) {
		v_r[blockIdx.x] = partial_sum[0];
	}
}

void initialize_vector(int *v, int n) {
	for (int i = 0; i < n; i++) {
		v[i] = 1;//rand() % 10;
	}
}

int main() {
	// Vector size
	int n = 1 << 16;
	size_t bytes = n * sizeof(int);

	// Original vector and result vector
	int *h_v, *h_v_r;
	int *d_v, *d_v_r;

	// Allocate memory
	h_v = (int*)malloc(bytes);
	h_v_r = (int*)malloc(bytes);
	cudaMalloc(&d_v, bytes);
	cudaMalloc(&d_v_r, bytes);

	// Initialize vector
	initialize_vector(h_v, n);

	// Copy to device
	cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

	// TB Size
	int TB_SIZE = SIZE;

	// Grid Size (cut in half) (No padding)
	int GRID_SIZE = n / TB_SIZE ;

	auto start = std::chrono::high_resolution_clock::now();

	// Call kernel
	sum_reduction << <GRID_SIZE, TB_SIZE >> > (d_v, d_v_r);

	sum_reduction << <1, TB_SIZE >> > (d_v_r, d_v_r);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout<< "Time taken by function 2 (matrixMul): "<< duration.count() << " microseconds" << std::endl;

	// Copy to host;
	cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

	std::cout<<"Accumulated result is "<<h_v_r[0]<<std::endl;

	printf("COMPLETED SUCCESSFULLY\n");

	return 0;
}