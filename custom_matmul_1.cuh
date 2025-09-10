#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gemm_policy.h"

// takes all its configuration from the policy.
template <typename GemmPolicy>
void __launch_bounds__(GemmPolicy::max_threads_per_block) __global__ customMatmulKernel(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < M && col < N) {
		typename GemmPolicy::compute_type sum = 0.0f;
		for (int k = 0; k < K; k++) {
			sum += static_cast<typename GemmPolicy::compute_type>(A[row * K + k]) * static_cast<typename GemmPolicy::compute_type>(B[k * N + col]);
		}
		C[row * N + col] = static_cast<typename GemmPolicy::c_value_type>(sum);
	}
}

// A generic host function to launch the kernel. It takes a policy as a template argument.
template <typename GemmPolicy>
void runCustomMatmul(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	dim3 blockSize(GemmPolicy::TILE_DIM, GemmPolicy::TILE_DIM);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

	customMatmulKernel<GemmPolicy> << <gridSize, blockSize >> > (M, N, K, A, B, C);
	CUDA_CHECK(cudaGetLastError());
}