#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gemm_policy.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
//L1TEX Local Store/load pattern issues
// Seems to be from Bsub set 
template <typename GemmPolicy>
void __launch_bounds__(GemmPolicy::max_threads_per_block) __global__ customMatmulKernel3(int M, int N, int K,
	typename const GemmPolicy::a_value_type* __restrict__ A,
	typename const GemmPolicy::b_value_type* __restrict__ B,
	typename GemmPolicy::c_value_type* __restrict__ C) {

	// value to be added to C for this thread
	int c_row = blockIdx.y * blockDim.y + threadIdx.y;
	int c_col = blockIdx.x * blockDim.x + threadIdx.x;

	// This is the position within the tile
	int thread_row = c_row % GemmPolicy::TILE_DIM;
	int thread_col = c_row % GemmPolicy::TILE_DIM;

	__shared__ typename GemmPolicy::a_value_type Asub[GemmPolicy::TILE_DIM * GemmPolicy::TILE_DIM];
	__shared__ typename GemmPolicy::b_value_type Bsub[GemmPolicy::TILE_DIM * GemmPolicy::TILE_DIM];

	typename GemmPolicy::compute_type sum = 0.0f;

	for (int k_tile_idx = 0; k_tile_idx < K; k_tile_idx += GemmPolicy::TILE_DIM) {


		
		Asub[thread_row + thread_col] = A[c_row];

		Bsub[thread_col] = B[c_col];

		__syncthreads();

		for (int idx = 0; idx < GemmPolicy::TILE_DIM; idx++) {
			GemmPolicy::multiply_accumulate(
				static_cast<typename GemmPolicy::compute_type>(Asub[thread_row * GemmPolicy::TILE_DIM + idx]),
				static_cast<typename GemmPolicy::compute_type>(Bsub[idx * GemmPolicy::TILE_DIM + thread_col]),
				sum);
		}
		__syncthreads();
	}

	int out_row = c_row + thread_row;
	int out_col = c_col + thread_col;

	if (out_row < M && out_col < N) {
		C[out_row * N + out_col] = static_cast<typename GemmPolicy::c_value_type>(sum);
	}
}

// A generic host function to launch the kernel. It takes a policy as a template argument.
template <typename GemmPolicy>
void runCustomMatmul3(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C) {

	dim3 blockSize(GemmPolicy::TILE_DIM, GemmPolicy::TILE_DIM);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

	customMatmulKernel3<GemmPolicy> << <gridSize, blockSize >> > (M, N, K, A, B, C);
	CUDA_CHECK(cudaGetLastError());
}