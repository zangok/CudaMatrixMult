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
#include "custom_matmul_3.cuh"
#include "mult_policy.h"
#include "kernels.cuh"
#include "benchmark.h"


int main() {

    auto kernels = get_bf16_kernels();
    auto results = benchmark_matmul<bf16>(2048, 2048, 2048, 1.0f, 0.0f, 1, kernels);

    for (auto& r : results) {
        std::cout << r.kernel_name
            << " | Avg: " << r.avg_ms << "ms"
            << " | GFLOPs: " << r.gflops
            << " | Valid: " << r.validation.valid
            << " | MaxAbs: " << r.validation.max_abs_err
            << " | MaxRel: " << r.validation.max_rel_err
            << "\n";
    }

    CUDA_CHECK(cudaGetLastError());
}
