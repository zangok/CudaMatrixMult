#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gemm_policy.h"
#include "mult_policy.h"

// New implementation of custom matmul with optimizations
// Idea is to iteratively improve using nsight compute
// 1. Memeory load /store optimizations (coalescing, shared memory, etc.) currently 70% speedup
// 2. Also realized casting and * isn't modular enough for different types & mult instructions.
// Added Multpolicy for various multipilication strategies
template <typename GemmPolicy>
void __launch_bounds__(GemmPolicy::max_threads_per_block) __global__ customMatmulKernel2(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < M && col < N) {
		typename GemmPolicy::compute_type sum = 0.0f;
		for (int k = 0; k < K; k++) {
			GemmPolicy::multiply_accumulate(A[row * K + k], B[k * N + col], sum);
		}
		C[row * N + col] = static_cast<typename GemmPolicy::c_value_type>(sum);
	}
}

// A generic host function to launch the kernel. It takes a policy as a template argument.
template <typename GemmPolicy>
void runCustomMatmul2(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	dim3 blockSize(GemmPolicy::TILE_DIM, GemmPolicy::TILE_DIM);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

	customMatmulKernel<GemmPolicy> << <gridSize, blockSize >> > (M, N, K, A, B, C);
	CUDA_CHECK(cudaGetLastError());
}