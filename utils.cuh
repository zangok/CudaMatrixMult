#pragma once
#include <cuda_bf16.h>

#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <library_types.h>

typedef __nv_bfloat16 bf16;

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)
