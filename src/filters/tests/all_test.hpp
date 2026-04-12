#pragma once

/**
 * @file all_test.hpp
 * @brief Test registry for filters module
 *
 * Called from src/main.cpp:
 *   #include "modules/filters/tests/all_test.hpp"
 *   filters_all_test::run();
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "test_filters_rocm.hpp"
#if ENABLE_ROCM
#include "test_filters_benchmark_rocm.hpp"
#include "test_moving_average_rocm.hpp"
#include "test_kalman_rocm.hpp"
#include "test_kaufman_rocm.hpp"
#endif

namespace filters_all_test {

inline void run() {
#if ENABLE_ROCM
  test_filters_rocm::run();

  // BENCHMARK: FirFilterROCm + IirFilterROCm (ROCm, GpuBenchmarkBase)
//  test_filters_benchmark_rocm::run();

  test_moving_average_rocm::run();   // Task_20: SMA/EMA/MMA/DEMA/TEMA
  test_kalman_rocm::run();           // Task_21: 1D Kalman filter
  test_kaufman_rocm::run();           // Task_22: KAMA (Kaufman adaptive MA)
#endif
}

}  // namespace filters_all_test
