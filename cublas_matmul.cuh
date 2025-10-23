#pragma once
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdexcept>
#include "cublas_utils.cuh"

//RAII Wrapper for cuBLASLt
struct CublasLtContext {
    cublasLtHandle_t handle;
    void* workspace = nullptr;
    size_t workspace_size = 32 * 1024 * 1024; 

    CublasLtContext() {
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    ~CublasLtContext() {
        if (workspace) cudaFree(workspace);
        cublasLtDestroy(handle);
    }

    CublasLtContext(const CublasLtContext&) = delete;
    CublasLtContext& operator=(const CublasLtContext&) = delete;
};

// cuBLASLt GEMM Runner
inline void runCublasLtBF16Gemm(int M, int N, int K,
    bf16* A, bf16* B, bf16* C,
    float alpha = 1.0f, float beta = 0.0f)
{
    static thread_local CublasLtContext ctx;
    // 1. Create operation descriptor
    cublasLtMatmulDesc_t opDesc;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_16F));

    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Scale type
    cublasDataType_t scale_type = CUDA_R_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type, sizeof(scale_type)));

    // 2. Create matrix layouts
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, M, K, M));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, K, N, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16BF, M, N, M));

    // 3. Create preference object
    cublasLtMatmulPreference_t preference;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ctx.workspace_size,
        sizeof(ctx.workspace_size)
    ));

    // 4. Pick algorithm heuristically
    cublasLtMatmulHeuristicResult_t heuristic;
    int returnedResults = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        ctx.handle, opDesc,
        layoutA, layoutB, layoutC, layoutC,
        preference,
        1, &heuristic, &returnedResults
    ));
    if (returnedResults == 0) {
        throw std::runtime_error("No suitable cuBLASLt algorithm found.");
    }

    // 5. Run GEMM
    CUBLAS_CHECK(cublasLtMatmul(
        ctx.handle,
        opDesc,
        &alpha,
        A, layoutA,
        B, layoutB,
        &beta,
        C, layoutC,
        C, layoutC, // Output
        &heuristic.algo,
        ctx.workspace, ctx.workspace_size,
        0
    ));

    // 6. Clean up
    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(preference));
    CUBLAS_CHECK(cublasLtMatmulDescDestroy(opDesc));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutA));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutB));
    CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(layoutC));
}
