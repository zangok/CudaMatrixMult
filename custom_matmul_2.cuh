#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gemm_policy.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
/*
// New implementation of custom matmul with optimizations
// Idea is to iteratively improve using nsight compute
// 1. Memeory load /store optimizations (coalescing, shared memory, etc.) currently 70% speedup
// 2. Also realized casting and * isn't modular enough for different types & mult instructions.
template <typename GemmPolicy>
__global__ void customMatmulKernel2(
    int M, int N, int K,
    const typename GemmPolicy::a_value_type* __restrict__ A,
    const typename GemmPolicy::b_value_type* __restrict__ B,
    typename GemmPolicy::c_value_type* __restrict__ C)
{
    // Global thread indices
    int row = blockIdx.y * GemmPolicy::TILE_DIM + threadIdx.y;
    int col = blockIdx.x * GemmPolicy::TILE_DIM + threadIdx.x;

    // Shared memory tiles
    __shared__ typename GemmPolicy::a_value_type Asub[GemmPolicy::TILE_DIM * GemmPolicy::TILE_DIM];
    __shared__ typename GemmPolicy::b_value_type Bsub[GemmPolicy::TILE_DIM * GemmPolicy::TILE_DIM];

    typename GemmPolicy::compute_type sum = 0.0f;

    // Loop over tiles along K
    for (int tile_idx = 0; tile_idx < K; tile_idx += GemmPolicy::TILE_DIM)
    {
        // Load A tile (row-major)
        int a_row = row;
        int a_col = tile_idx + threadIdx.x;
        if (a_row < M && a_col < K)
            Asub[threadIdx.y * GemmPolicy::TILE_DIM + threadIdx.x] = A[a_row * K + a_col];
        else
            Asub[threadIdx.y * GemmPolicy::TILE_DIM + threadIdx.x] = static_cast<typename GemmPolicy::a_value_type>(0);

        // Load B tile (row-major)
        int b_row = tile_idx + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bsub[threadIdx.y * GemmPolicy::TILE_DIM + threadIdx.x] = B[b_row * N + b_col];
        else
            Bsub[threadIdx.y * GemmPolicy::TILE_DIM + threadIdx.x] = static_cast<typename GemmPolicy::b_value_type>(0);

        __syncthreads();

        // Multiply A_tile * B_tile
        int max_k = min(GemmPolicy::TILE_DIM, K - tile_idx);
        for (int k = 0; k < max_k; k++)
        {
            float a_val = static_cast<float>(Asub[threadIdx.y * GemmPolicy::TILE_DIM + k]);
            float b_val = static_cast<float>(Bsub[k * GemmPolicy::TILE_DIM + threadIdx.x]);
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N)
        C[row * N + col] = static_cast<typename GemmPolicy::c_value_type>(sum);
}

// A generic host function to launch the kernel. It takes a policy as a template argument.
template <typename GemmPolicy>
void runCustomMatmul2(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	dim3 blockSize(GemmPolicy::TILE_DIM, GemmPolicy::TILE_DIM);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

	customMatmulKernel2<GemmPolicy><<<gridSize, blockSize>>>(M, N, K, A, B, C);

	CUDA_CHECK(cudaGetLastError());
}*/