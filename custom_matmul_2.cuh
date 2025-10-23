#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "gemm_policy.h"

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// New implementation of custom matmul with optimizations
// Idea is to iteratively improve using nsight compute
// 1. Memeory load /store optimizations (coalescing, shared memory, etc.) currently 70% speedup
// 2. Also realized casting and * isn't modular enough for different types & mult instructions.

// As we are using col major infexing, coalesced memory access done by reading inside columns
// e.i. traverse rows in inner column loop and transpose the access pattern for shared memory
// A(row, col) = A[col * M + row]
// B(row, col) = B[col * K + row]
// C(row, col) = C[col * M + row]

// Note: An earlier version of this kernel had the memory access
// seperated into loading in shared memory but still had uncoalesced access in
// typename GemmPolicy::c_value_type* C_ptr = C + (out_col * M + out_row);

//Also changed to multiply_accumulate with fma 

//With these changes, GFLOPs: 2.94089 vs GFLOPs: 6.6306 for bf16 on 2056x2056 on matmul1
template <typename GemmPolicy>
__global__ void customMatmulKernel2(int M, int N, int K,
    typename GemmPolicy::a_value_type* A,
    typename GemmPolicy::b_value_type* B,
    typename GemmPolicy::c_value_type* C,
    typename GemmPolicy::compute_type alpha,
    typename GemmPolicy::compute_type beta) {

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int load_row = threadIdx.y;
    const int load_col = threadIdx.x;

    const int out_row = block_row * GemmPolicy::TILE_DIM + load_row;
    const int out_col = block_col * GemmPolicy::TILE_DIM + load_col;

    __shared__ typename GemmPolicy::compute_type A_tile[GemmPolicy::TILE_DIM][GemmPolicy::TILE_DIM];
    __shared__ typename GemmPolicy::compute_type B_tile[GemmPolicy::TILE_DIM][GemmPolicy::TILE_DIM];

    typename GemmPolicy::compute_type acc = 0.0f;

    int tiles = CEIL_DIV(K, GemmPolicy::TILE_DIM);

    for (int i = 0; i < tiles; i++) {
        int a_row = block_row * GemmPolicy::TILE_DIM + load_col;
        int a_col = i * GemmPolicy::TILE_DIM + load_row;

        int b_row = i * GemmPolicy::TILE_DIM + load_col;
        int b_col = block_col * GemmPolicy::TILE_DIM + load_row;

        // --- A tile (column-major) ---
        if (a_row < M && a_col < K)
            A_tile[load_row][load_col] = GemmPolicy::to_compute_type(A[a_row + a_col * M]);
        else
            A_tile[load_row][load_col] = 0;

        // --- B tile (column-major) ---
        if (b_row < K && b_col < N)
            B_tile[load_row][load_col] = GemmPolicy::to_compute_type(B[b_row + b_col * K]);
        else
            B_tile[load_row][load_col] = 0;

		__syncthreads();
        
        // --- Compute C tile ---
        #pragma unroll
        for (int k = 0; k < GemmPolicy::TILE_DIM; k++) {
            int global_k = i * GemmPolicy::TILE_DIM + k;
            if (global_k < K) {
                //acc += GemmPolicy::multiply(A_tile[load_row][k], B_tile[k][load_col]);
                GemmPolicy::multiply_accumulate(A_tile[k][load_row], B_tile[load_col][k], acc);
            }
        }
		__syncthreads();
    }

    GemmPolicy::store_c_value_type(C, acc, out_row, out_col, M, N, load_row, load_col);
}


// A generic host function to launch the kernel. It takes a policy as a template argument.
template <typename GemmPolicy>
void runCustomMatmul2(int M, int N, int K,
	typename GemmPolicy::a_value_type* A,
	typename GemmPolicy::b_value_type* B,
	typename GemmPolicy::c_value_type* C,
	typename GemmPolicy::compute_type alpha,
	typename GemmPolicy::compute_type beta) {

	dim3 blockSize(GemmPolicy::TILE_DIM, GemmPolicy::TILE_DIM);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

	customMatmulKernel2<GemmPolicy> << <gridSize, blockSize >> > (M, N, K, A, B, C, alpha, beta);

    cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError());
}