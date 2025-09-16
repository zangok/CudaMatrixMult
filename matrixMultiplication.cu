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
#include "custom_matmul_2.cuh"
#include "mult_policy.h"

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
        {"custom bf16 GEMM", runCustomMatmul<GemmPolicyBF16>},
		{"custom bf16 GEMM v2", runCustomMatmul2<GemmPolicyBF16>},
		
    };
}

template <typename T>
void validate_results(const GpuMatrix<T>& C_ref, const GpuMatrix<T>& C_test, const std::string& kernel_name) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    auto h_ref = C_ref.to_host();   // Copy from device to host
    auto h_test = C_test.to_host();

    for (size_t i = 0; i < h_ref.size(); i++) {
        float ref = static_cast<float>(h_ref[i]);
        float test = static_cast<float>(h_test[i]);
		//std::cout << "Ref: " << ref << ", Test: " << test << "\n";
        float abs_err = fabs(ref - test);
        float rel_err = (fabs(ref) > 1e-6f) ? abs_err / fabs(ref) : abs_err;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    std::cout << kernel_name
        << " | Max Abs Err: " << max_abs_err
        << " | Max Rel Err: " << max_rel_err
        << (max_rel_err < 1e-2f ? "valid" : "not valid")
        << "\n";
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
    GpuMatrix<T> C_ref(N, K);

    A.fill_normal();
    B.fill_normal();

    auto kernels = get_kernels<T>();

    auto ref_kernel = kernels.at(0).second;
    ref_kernel(N, M, K, A.data(), B.data(), C_ref.data());

	// Warmup
    /*
    const auto& kernel_pair_warmup = kernels.at(1);
    const auto& kernel_name_warmup = kernel_pair_warmup.first;
    auto kernel_fn_warmup = kernel_pair_warmup.second;

    kernel_fn_warmup(N, M, K, A.data(), B.data(), C.data());
    */

	// Benchmark each kernel
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

        validate_results(C_ref, C, kernel_name);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main() {

    initCublasLt();

    compare_matmul<bf16>(512, 512, 512, 1);

}
