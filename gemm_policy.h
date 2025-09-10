#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// This is the generic policy "template".
// It defines the names for the types and constants that all
// specific policies must provide.
// Based on usage of template < class blas> as seen in cublas docs 
struct GemmPolicy {
    using a_value_type = void;
    using b_value_type = void;
    using c_value_type = void;
    using compute_type = void;

    static const int TILE_DIM = 32;
    static const int max_threads_per_block = 1024;
};


// --- Specific Policies ---
//Could seperate into other files depending on use case

// Policy for Bfloat16
struct GemmPolicyBF16 : public GemmPolicy {
    using a_value_type = __nv_bfloat16;
    using b_value_type = __nv_bfloat16;
    using c_value_type = __nv_bfloat16;
    using compute_type = float;

    static const int TILE_DIM = 32;
    static const int max_threads_per_block = 1024;
};

// Policy for Float16
// can use later just to compare?
struct GemmPolicyFP16 : public GemmPolicy {
    using a_value_type = __half;
    using b_value_type = __half;
    using c_value_type = __half;
    using compute_type = float;

    static const int TILE_DIM = 64; 
    static const int max_threads_per_block = 1024;
};