#pragma once

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>

struct GemmPolicyBF16 {
	
	using a_value_type = __nv_bfloat16;
	using b_value_type = __nv_bfloat16;
	using c_value_type = __nv_bfloat16;
	using compute_type = float;
	static constexpr cudaDataType_t a_type = CUDA_R_16BF;
	static constexpr cudaDataType_t b_type = CUDA_R_16BF;
	static constexpr cudaDataType_t c_type = CUDA_R_16BF;
	static constexpr cudaDataType_t compute_type_enum = CUDA_R_32F;

	static constexpr int max_threads_per_block = 1024;
	static constexpr int TILE_DIM = 16;
};