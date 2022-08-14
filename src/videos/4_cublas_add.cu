
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <gtest/gtest.h>


// Check result on the CPU
void verify_result(float *a, float *b, float *c, int N) {
  
  for (int i = 0; i < N; i++) {
    assert(c[i] == 4.0*a[i]+b[i]);
    }
}

void create_vec(float *A,long long  int n)
{
    for(long long int i=0;i<n;i++)
        A[i] = (float)i;
}

//print an array of size N
void print_array(float *A, long long int n) {
  for(long long int i=0;i<n;i++)
    std::cout << A[i] << std::endl;
}


int main() {
  int id = cudaGetDevice(&id);
  auto start = std::chrono::high_resolution_clock::now();
  int n = 1<<12;

  // Size (in bytes) of matrix
  size_t bytes = n * sizeof(float);

  // Host vectors
  float *h_a, *d_a;
  float *h_b, *d_b;
  float *h_c, *d_c;

  h_a = (float *)malloc(bytes);
  h_b = (float *)malloc(bytes);
  h_c = (float *)malloc(bytes);

  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
    
  create_vec(h_a,n);
  create_vec(h_b,n);

  // print_vector(h_a,n);

  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  
  cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
  cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

  const float scale = 4.0f;

  cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

  cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

  verify_result(h_a, h_b, h_c, n);
  std::cout<<"verify_result: "<<std::endl;

  // print_array(h_c,n);

  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // std::cout<< "Time taken by function: "<< duration.count() << " microseconds" << std::endl;


  return 0;
}