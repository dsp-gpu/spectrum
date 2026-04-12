#pragma once

/**
 * @file all_test.hpp
 * @brief Тесты модуля fft_func — ветка main (ROCm/hipFFT)
 *
 * Ветка: main (Linux / AMD GPU / ROCm)
 * Все тесты используют ROCm/hipFFT.
 *
 * main.cpp вызывает этот файл — НЕ отдельные тесты напрямую.
 * Включить/закомментировать нужные тесты здесь.
 */

// ─── ROCm / hipFFT тесты ─────────────────────────────────────────────────────
#include "test_fft_processor_rocm.hpp"
#include "test_complex_to_mag_phase_rocm.hpp"
#include "test_process_magnitude_rocm.hpp"
#include "test_fft_matrix_rocm.hpp"
#include "test_spectrum_maxima_rocm.hpp"
// #include "test_fft_benchmark_rocm.hpp"          // benchmark — долго
// #include "test_fft_maxima_benchmark_rocm.hpp"    // benchmark — долго

namespace fft_func_all_test {

inline void run() {
    // FFTProcessorROCm: hipFFT-based FFT
    test_fft_processor_rocm::run();

    // ComplexToMagPhaseROCm: complex → mag+phase
    test_complex_to_mag_phase_rocm::run();

    // ProcessMagnitude: magnitude-only, normalization, managed memory
    test_process_magnitude_rocm::run();

    // FFT Matrix: beams × nFFT table
    test_fft_matrix_rocm::run();

    // SpectrumMaximaFinder ROCm
    test_spectrum_maxima_rocm::run();

    // Benchmarks (раскомментировать для запуска):
    // test_fft_benchmark_rocm::run();
    // test_fft_maxima_benchmark_rocm::run();
}

}  // namespace fft_func_all_test
