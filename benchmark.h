// benchmark.h
#pragma once
#include <vector>
#include "GpuMatrix.h"
#include "kernels.cuh"
#include "validation.h"

template <typename T>
struct BenchmarkResult {
    std::string kernel_name;
    float avg_ms;
    double gflops;
    ValidationResult<T> validation;
};

template <typename T>
std::vector<BenchmarkResult<T>> benchmark_matmul(
    int N, int M, int K, float alpha, float beta, int iterations,
    const std::vector<KernelEntry<T>>& kernels
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    GpuMatrix<T> A(N, M), B(M, K), C(N, K), C_ref(N, K);
    A.fill_normal();
    B.fill_normal();

    kernels.front().fn(N, M, K, A.data(), B.data(), C_ref.data(), alpha, beta);

    std::vector<BenchmarkResult<T>> results;

    for (auto& k : kernels) {
        float total_ms = 0.0f;
        for (int i = 0; i < iterations; i++) {
            cudaEventRecord(start);
            k.fn(N, M, K, A.data(), B.data(), C.data(), alpha, beta);
            cudaDeviceSynchronize();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        double flops = 2.0 * N * M * K;
        double gflops = (flops / (total_ms / iterations / 1000.0)) / 1e9;
        results.push_back({ k.name, total_ms / iterations, gflops, validate_results(C_ref, C) });
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return results;
}
