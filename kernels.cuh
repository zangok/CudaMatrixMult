// kernels.h
#pragma once
#include <vector>
#include <string>
#include "GpuMatrix.h"
#include "gemm_policy.h"
#include "cublas_matmul.cuh"
#include "custom_matmul_1.cuh"
#include "custom_matmul_2.cuh"

// Kernel function signature
template <typename T>
using KernelFn = void(*)(int, int, int, T*, T*, T*, float, float);

template <typename T>
struct KernelEntry {
    std::string name;
    KernelFn<T> fn;
};



inline std::vector<KernelEntry<bf16>> get_bf16_kernels() {
    return {
        KernelEntry<bf16>{"cuBLAS bf16 GEMM", runCublasLtBF16Gemm},
        KernelEntry<bf16>{"custom bf16 GEMM 1", runCustomMatmul<GemmPolicyBF16>},
        KernelEntry<bf16>{"custom bf16 GEMM 2", runCustomMatmul2<GemmPolicyBF16>},
    };
}