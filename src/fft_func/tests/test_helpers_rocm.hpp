#pragma once

/**
 * @file test_helpers_rocm.hpp
 * @brief Test helpers for ROCm tests using hipMallocManaged (unified memory)
 *
 * Unified memory (hipMallocManaged) allows CPU to fill data and GPU to read
 * it directly without explicit hipMemcpy. Use for small test datasets only.
 *
 * Usage:
 *   void* ptr = AllocateManagedForTest(n * sizeof(std::complex<float>));
 *   auto* data = static_cast<std::complex<float>*>(ptr);
 *   // fill data on CPU...
 *   auto input = MakeManagedInput(ptr, beam_count, n_point);
 *   // call ProcessMagnitude(input.data, params, input.gpu_memory_bytes)
 *   hipFree(ptr);  // caller is responsible
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-11
 */

#if ENABLE_ROCM

#include <core/interface/input_data.hpp>

#include <hip/hip_runtime.h>

#include <complex>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace test_helpers_rocm {

/// Allocate unified memory (CPU fill + GPU read without hipMemcpy). Caller must hipFree.
inline void* AllocateManagedForTest(size_t bytes) {
    void* ptr = nullptr;
    hipError_t e = hipMallocManaged(&ptr, bytes);
    if (e != hipSuccess) {
        throw std::runtime_error("AllocateManagedForTest: hipMallocManaged failed: " +
                                  std::string(hipGetErrorString(e)));
    }
    return ptr;
}

/// Build InputData<void*> for complex<float> managed buffer (beam_count * n_point complex)
inline drv_gpu_lib::InputData<void*> MakeManagedInput(
    void* ptr, uint32_t beam_count, uint32_t n_point)
{
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = static_cast<size_t>(beam_count) * n_point * sizeof(std::complex<float>);
    return out;
}

/// Build InputData<void*> for float magnitudes managed buffer (beam_count * n_point float)
inline drv_gpu_lib::InputData<void*> MakeManagedMagnitudeInput(
    void* ptr, uint32_t beam_count, uint32_t n_point)
{
    drv_gpu_lib::InputData<void*> out;
    out.data = ptr;
    out.antenna_count = beam_count;
    out.n_point = n_point;
    out.gpu_memory_bytes = static_cast<size_t>(beam_count) * n_point * sizeof(float);
    return out;
}

}  // namespace test_helpers_rocm

#endif  // ENABLE_ROCM
