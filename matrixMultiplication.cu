#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cublasLt.h>
#include <cublas_v2.h>
#include "cublas_matmul.cuh"
#include "GpuMatrix.h"
#include "custom_matmul_1.cuh"
#include "gemm_policy.h"

// Define function signature for kernels
template <typename T>
using KernelFn = void(*)(int, int, int, T*, T*, T*);

// Kernel registry per type
template <typename T>
std::vector<std::pair<std::string, KernelFn<T>>> get_kernels();

template <>
std::vector<std::pair<std::string, KernelFn<bf16>>> get_kernels<bf16>() {
    return {
        {"cuBLAS bf16 GEMM", runCublasMatmulBF16},
        {"custom bf16 GEMM", runCustomMatmul<GemmPolicyBF16>}
		
    };
}


// Function that runs the various matmul implementations & compares
//N, M, K: Matrix sizes for A(N,M) * B(M,K) = C(N,K)
//iterations: total iterations to average over for each kernel
template <typename T>
void compare_matmul(int N, int M, int K, int iterations) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GpuMatrix<T> A(N, M);
    GpuMatrix<T> B(M, K);
    GpuMatrix<T> C(N, K);

    A.fill_normal();
    B.fill_normal();

    auto kernels = get_kernels<T>();

    for (auto& kv : kernels) {
        const auto& kernel_name = kv.first;
        auto kernel_fn = kv.second;
        float total_ms = 0.0f;

        for (int i = 0; i < iterations; i++) {
            cudaEventRecord(start);

            kernel_fn(N, M, K, A.data(), B.data(), C.data());

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        std::cout  << kernel_name << " Average execution time: " << (total_ms / iterations) << " ms\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {

    initCublasLt();

    compare_matmul<bf16>(4096, 4096, 4096, 1);

}
