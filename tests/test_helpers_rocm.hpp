#pragma once

// ============================================================================
// test_helpers_rocm — утилиты для ROCm-тестов spectrum
//
// ЧТО:    Хелперы для тестов: выделение unified memory (hipMallocManaged),
//         построение InputData<void*> для complex<float> и float magnitude.
// ЗАЧЕМ:  Устраняет дублирование setup-кода в каждом test_*_rocm.hpp.
// ПОЧЕМУ: hipMallocManaged позволяет заполнять буфер на CPU без hipMemcpy;
//         caller управляет временем жизни (hipFree) — RAII не нужен в тестах.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_helpers_rocm.hpp
 * @brief Хелперы ROCm тестов — unified memory (hipMallocManaged) для малых датасетов
 * @note Test fixture, не публичный API. Используется из test_*_rocm.hpp через include.
 *       Caller отвечает за hipFree выделенного ptr (RAII не оборачиваем — простота тестов).
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
