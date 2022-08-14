
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

__global__ void matrixMul(const int *a, const int *b, int *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

// Check result on the CPU
void verify_result(int *a, int *b, int *c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * N + j];
      }

      // Check against the CPU result
      assert(tmp == c[i * N + j]);
    }
  }
}

void create_vec(int *A,long long  int n)
{
    // A = new float[n];
    for(long long int i=0;i<n;i++)
        A[i] = rand() % 1000;
}


int main() {
  int id = cudaGetDevice(&id);
  // Matrix size of 1024 x 1024;
  auto start = std::chrono::high_resolution_clock::now();
  int N = 1 << 10;

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  int *h_a;
  int *h_b;
  int *h_c;

  cudaMallocManaged(&h_a, N*N*sizeof(int));
  cudaMallocManaged(&h_b, N*N*sizeof(int));
  cudaMallocManaged(&h_c, N*N*sizeof(int));
    
// Initialize matrices
// Very nicely done
//   std::generate(h_a.begin(), h_a.end(), []() { return rand() % 10000; });
//   std::generate(h_b.begin(), h_b.end(), []() { return rand() % 10000; });

  create_vec(h_a,N*N);
  create_vec(h_a,N*N);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  cudaMemPrefetchAsync(h_a, N*N*sizeof(int), id);
  cudaMemPrefetchAsync(h_b, N*N*sizeof(int), id);
  // Launch kernel
  matrixMul<<<blocks, threads>>>(h_a, h_b, h_c, N);
  cudaDeviceSynchronize();

  cudaMemPrefetchAsync(h_c, N*N*sizeof(int), cudaCpuDeviceId);

  // Check result
  auto stop = std::chrono::high_resolution_clock::now();
  verify_result(h_a, h_b, h_c, N);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  cudaFree(h_a);
  cudaFree(h_b);
  cudaFree(h_c);

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout<< "Time taken by function: "<< duration.count() << " microseconds" << std::endl;


  return 0;
}