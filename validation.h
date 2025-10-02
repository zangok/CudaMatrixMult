#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include "GpuMatrix.h"

template <typename T>
struct ValidationResult {
    float max_abs_err;
    float max_rel_err;
    bool valid;
};


template <typename T>
ValidationResult<T> validate_results(const GpuMatrix<T>& C_ref, const GpuMatrix<T>& C_test) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    auto h_ref = C_ref.to_host();   // Copy from device to host
    auto h_test = C_test.to_host();

    for (size_t i = 0; i < h_ref.size(); i++) {
        float ref = static_cast<float>(h_ref[i]);
        float test = static_cast<float>(h_test[i]);
        //std::cout << "Ref: " << ref << ", Test: " << test << "\n";
        float abs_err = fabs(ref - test);
        float rel_err = (fabs(ref) > 1e-6f) ? abs_err / fabs(ref) : abs_err;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
    }

    /*std::cout << kernel_name
        << " | Max Abs Err: " << max_abs_err
        << " | Max Rel Err: " << max_rel_err
        << (max_rel_err < 1e-2f ? "valid" : "not valid")
        << "\n";*/

    return { max_abs_err, max_rel_err, max_rel_err < 1e-2f };

}
