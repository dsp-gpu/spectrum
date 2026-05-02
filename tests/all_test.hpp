#pragma once

// ============================================================================
// spectrum_all_test — агрегатор тестов модуля spectrum
//
// ЧТО:    Единая точка подключения всех test_*.hpp модуля spectrum.
//         Функция run() последовательно запускает весь набор тестов.
// ЗАЧЕМ:  main.cpp вызывает только этот файл — не отдельные test_*.hpp.
//         Закомментированный include = выключенный тест без правки main.cpp.
// ПОЧЕМУ: Паттерн all_test.hpp — из правила 15-cpp-testing.md.
//
// История: Создан: 2026-02-23
// ============================================================================

/**
 * @file all_test.hpp
 * @brief Агрегатор всех тестов модуля spectrum.
 * @note Не публичный API. Вызывается из main.cpp.
 */

#if ENABLE_ROCM
#include "test_lch_farrow_rocm.hpp"
#include "test_lch_farrow_benchmark_rocm.hpp"
#include "test_fft_cpu_reference_rocm.hpp"
#include "test_gate3_fft_profiler_v2.hpp"
#endif

namespace lch_farrow_all_test {

inline void run() {
#if ENABLE_ROCM
    test_lch_farrow_rocm::run();
    test_fft_cpu_reference::run();
    // test_lch_farrow_benchmark_rocm::run();
    test_gate3_fft_profiler_v2::run();
#endif
}

}  // namespace lch_farrow_all_test
