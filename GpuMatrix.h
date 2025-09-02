
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>

#include <stdexcept>
#include <string>
#include <sstream>
#include "utils.cuh"

// A simple RAII-style wrapper for a 2D matrix allocated on the GPU.
template <typename T>
class GpuMatrix {
public:
    // Constructor
    GpuMatrix(size_t numRows, size_t numCols)
        : numRows_(numRows), numCols_(numCols) {
        if (numRows_ == 0 || numCols_ == 0) {
            throw std::invalid_argument("Matrix dimensions must be greater than zero.");
        }
        size_t numElements = numRows_ * numCols_;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data_), sizeof(T) * numElements));
        std::cout << "Successfully allocated GPU memory for " << numElements << " elements." << std::endl;
    }

    // Destructor
    ~GpuMatrix() {
        if (d_data_) {
            cudaFree(d_data_);
            d_data_ = nullptr;
            std::cout << "Successfully freed GPU memory." << std::endl;
        }
    }

    // The Rule of Five: Prevent accidental copies that would lead to a double-free error.
    GpuMatrix(const GpuMatrix&) = delete;
    GpuMatrix& operator=(const GpuMatrix&) = delete;

    // Move constructor and move assignment operator for efficient transfers
    GpuMatrix(GpuMatrix&& other) noexcept
        : d_data_(other.d_data_), numRows_(other.numRows_), numCols_(other.numCols_) {
        other.d_data_ = nullptr;
        other.numCols_ = 0;
        other.numRows_ = 0;
    }

    GpuMatrix& operator=(GpuMatrix&& other) noexcept {
        if (this != &other) {
            if (d_data_) {
                cudaFree(d_data_);
            }
            d_data_ = other.d_data_;
            numRows_ = other.numRows_;
            numCols_ = other.numCols_;

            other.d_data_ = nullptr;
            other.numRows_ = 0;
            other.numCols_ = 0;
        }
        return *this;
    }

    // Fills the matrix with normal(mean, standard deviation)
    void fill_normal(float mean = 0.0f, float stddev = 1.0f) {
        size_t N = totalElements();
        std::vector<T> h_data(N);

        // Random generator on host
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);

        for (size_t i = 0; i < N; ++i) {
            float f = dist(gen);
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                h_data[i] = __float2bfloat16(f);
            }
            else {
                h_data[i] = static_cast<T>(f);
            }
        }

        cudaMemcpy(d_data_, h_data.data(), sizeof(T) * N, cudaMemcpyHostToDevice);
    }

    // Returns a raw device pointer to the matrix data
    T* data() const { return d_data_; }

    // Accessor methods
    size_t numRows() const { return numRows_; }
    size_t numCols() const { return numCols_; }
    size_t totalElements() const { return numRows_ * numCols_; }

private:
    T* d_data_ = nullptr; // Device pointer
    size_t numRows_;
    size_t numCols_;
};