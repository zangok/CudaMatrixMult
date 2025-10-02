#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// This is the base GEMM Policy to be derived
// It defines the names for the types and constants that all
// specific policies must provide.
// Based on cutlass github docs / layer of abstraction for later finetunning 
// could abstract out more, but for now this is sufficient.
template <typename Derived>
struct GemmPolicy {
    static constexpr int TILE_DIM = 32;
    static constexpr int max_threads_per_block = 1024;

    using a_value_type = float;
    using b_value_type = float;
    using c_value_type = float;
    using compute_type = float;

    __host__ __device__ static compute_type multiply_accumulate(
        const a_value_type&, const b_value_type&, compute_type&)
    {
        static_assert(sizeof(Derived) == -1,
            "Derived policy must implement multiply_accumulate");
    }

    __host__ __device__ static compute_type multiply(
        const a_value_type&, const b_value_type&)
    {
        static_assert(sizeof(Derived) == -1,
            "Derived policy must implement multiply");
    }
};

// ===========================================================
//  Specific Policy: BF16
// ===========================================================
struct GemmPolicyBF16 : public GemmPolicy<GemmPolicyBF16> {
    using a_value_type = __nv_bfloat16;
    using b_value_type = __nv_bfloat16;
    using c_value_type = __nv_bfloat16;
    using compute_type = float;

    __host__ __device__ static compute_type multiply_accumulate(
        const a_value_type& a, const b_value_type& b, compute_type& sum)
    {
        float a_float = __bfloat162float(a);
        float b_float = __bfloat162float(b);

        sum += a_float * b_float;

        return sum; 
    }

    __host__ __device__ static compute_type multiply(
        const a_value_type& a, const b_value_type& b)
    {
        return __bfloat162float(a) * __bfloat162float(b);
    }
};