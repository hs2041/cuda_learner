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
#include "cublas_v2.h"

__global__ 
void vec_add(float *A, float *B, float *C, long long int n)
{
    long long int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}

void create_vec(float *A,long long  int n)
{
    // A = new float[n];
    srand (static_cast <unsigned> (time(0)));
    for(long long int i=0;i<n;i++)
        A[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/1000000.0));
        // A[i] = i;
}

void checker(float *a, float *b, float *c2, long long int n)
{
    for(long long int i=0;i<n;i++)
    {
        ASSERT_EQ(c2[i], a[i]+b[i]);
    }
}

void gpu_add (float *A, float *B, float *C, long long int n)
{
    long long int size = n*sizeof(float);

    // float *d_a, *d_b, *d_c;

    vec_add<<<std::ceil(n/256.0),256>>>(A, B, C, n);     

    cudaDeviceSynchronize();
    // cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    // cudaFree(d_a);
    // cudaFree(d_b);    
    // cudaFree(d_c);
    
}

template <typename T>
void print_array( T* array,long long  int n)
{
    for(long long int i=0;i<n;i++)
        std::cout<<array[i]<<" ";
    std::cout<<std::endl;
}

void get_time(std::function<void(float*, float*, float*, long long int)> func, float* A, float* B, float* C, long long int n) 
{

    auto start = std::chrono::high_resolution_clock::now();
    func(A,B,C,n);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout<< "Time taken by function: "<< duration.count() << " microseconds" << std::endl;
}

void initialize(long long int n)
{
    int id = cudaGetDevice(&id);

    float *A;
    // A = (float *)malloc(n*sizeof(float));
    float *B;
    // B = (float *)malloc(n*sizeof(float));
    float *C;
    // C1 = (float *)malloc(n*sizeof(float));

    cudaMallocManaged(&A, n*sizeof(float));
    cudaMallocManaged(&B, n*sizeof(float));
    cudaMallocManaged(&C, n*sizeof(float));
    
    create_vec(A,n);
    create_vec(B,n);
    
    cudaMemPrefetchAsync(A, n*sizeof(float), id);
    cudaMemPrefetchAsync(B, n*sizeof(float), id);
    
    std::function<void(float*, float*, float*, long long int)>gpufun = gpu_add;

    get_time(gpufun,A,B,C,n);

    // gpu_add(A,B,C,n);

    // checker(A, B, C, n);
    // delete A;
}

int main(int argc, char** argv) {

    long long int n = std::stoi(argv[1]);

    initialize(n);
    return 0;

}
