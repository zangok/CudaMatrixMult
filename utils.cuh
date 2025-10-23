#pragma once
#include <cuda_bf16.h>

#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <iostream>
typedef __nv_bfloat16 bf16;

// CUDA API error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr,                                              \
                "CUDA error %d (%s) at %s:%d\n",                              \
                static_cast<int>(err_), cudaGetErrorString(err_),             \
                __FILE__, __LINE__);                                          \
            throw std::runtime_error("CUDA error");                           \
        }                                                                     \
    } while (0)