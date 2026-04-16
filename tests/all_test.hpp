#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for lch_farrow module
 *
 * main.cpp calls this file - NOT individual tests directly.
 *
 * Тесты:
 *  - test_lch_farrow        : функциональные тесты (OpenCL)
 *  - test_lch_farrow_rocm   : функциональные тесты (ROCm)
 *
 * Бенчмарки (раскомментировать для запуска):
 *  - test_lch_farrow_benchmark      : OpenCL (GpuBenchmarkBase)
 *  - test_lch_farrow_benchmark_rocm : ROCm   (GpuBenchmarkBase)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18, обновлено 2026-03-01
 */

#if ENABLE_ROCM
#include "test_lch_farrow_rocm.hpp"
#include "test_lch_farrow_benchmark_rocm.hpp"
#include "test_fft_cpu_reference_rocm.hpp"
#endif

namespace lch_farrow_all_test {

inline void run() {
#if ENABLE_ROCM
    test_lch_farrow_rocm::run();
    test_fft_cpu_reference::run();
    // test_lch_farrow_benchmark_rocm::run();
#endif
}

}  // namespace lch_farrow_all_test
