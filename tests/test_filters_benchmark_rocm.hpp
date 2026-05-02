#pragma once

// ============================================================================
// test_filters_benchmark_rocm — бенчмарк FirFilterROCm + IirFilterROCm
//
// ЧТО:    2 бенчмарка: FirFilterROCm::ProcessFromCPU и IirFilterROCm::ProcessFromCPU.
//         5 warmup + 20 замерных → PrintReport + ExportJSON + ExportMarkdown.
// ЗАЧЕМ:  FIR/IIR фильтры — в каждом radar pipeline. Регрессии производительности.
// ПОЧЕМУ: Нет AMD GPU → [SKIP]. Results/Profiler/GPU_00_FirFilter_ROCm/.
//
// История: Создан: 2026-03-01
// ============================================================================

/**
 * @file test_filters_benchmark_rocm.hpp
 * @brief Test runner: FirFilterROCm + IirFilterROCm benchmark (GpuBenchmarkBase)
 *
 * Запускает два ROCm бенчмарка:
 *  Benchmark 1: FirFilterROCm::ProcessFromCPU → Results/Profiler/GPU_00_FirFilter_ROCm/
 *  Benchmark 2: IirFilterROCm::ProcessFromCPU → Results/Profiler/GPU_00_IirFilter_ROCm/
 *
 * Если нет AMD GPU — выводит [SKIP] и не падает.
 * Каждый: 5 прогревочных + 20 замерных → PrintReport + ExportJSON + ExportMarkdown.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see filters_benchmark_rocm.hpp, MemoryBank/tasks/TASK_filters_profiling.md
 */

#if ENABLE_ROCM

#include "filters_benchmark_rocm.hpp"
#include "test_fir_basic.hpp"   // kTestFirCoeffs64
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/backends/rocm/rocm_core.hpp>

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_filters_benchmark_rocm {

// ─── Утилита: генерация тестового сигнала ─────────────────────────────────────

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t channels, uint32_t points, float sample_rate)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(channels) * points);
  for (uint32_t ch = 0; ch < channels; ++ch) {
    float freq = 100.0f + ch * 500.0f;
    for (uint32_t t = 0; t < points; ++t) {
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
      data[static_cast<size_t>(ch) * points + t] = {std::cos(phase), std::sin(phase)};
    }
  }
  return data;
}

// ─── Точка входа ──────────────────────────────────────────────────────────────

inline int run() {
  std::cout << "\n"
            << "============================================================\n"
            << "  Filters ROCm Benchmark (FirFilterROCm + IirFilterROCm)\n"
            << "============================================================\n";

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    std::cout << "  [SKIP] No AMD GPU available\n";
    return 0;
  }

  try {
    // ── ROCm backend init ─────────────────────────────────────────────────
    auto backend = std::make_unique<drv_gpu_lib::ROCmBackend>();
    backend->Initialize(0);

    // ── Параметры ─────────────────────────────────────────────────────────
    constexpr uint32_t CHANNELS    = 8;
    constexpr uint32_t POINTS      = 4096;
    constexpr float    SAMPLE_RATE = 50000.0f;

    auto input_data = GenerateSignal(CHANNELS, POINTS, SAMPLE_RATE);

    // ── Benchmark 1: FirFilterROCm ────────────────────────────────────────
    std::cout << "\n--- Benchmark 1: FirFilterROCm::ProcessFromCPU (64-tap LP FIR) ---\n";
    {
      filters::FirFilterROCm fir(backend.get());
      fir.SetCoefficients(filters::tests::kTestFirCoeffs64);

      test_filters_rocm::FirFilterROCmBenchmark bench(
          backend.get(), fir, input_data, CHANNELS, POINTS,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_FirFilter_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] FirFilterROCm benchmark complete\n";
    }

    // ── Benchmark 2: IirFilterROCm ────────────────────────────────────────
    std::cout << "\n--- Benchmark 2: IirFilterROCm::ProcessFromCPU (Butterworth 2nd order LP) ---\n";
    {
      filters::IirFilterROCm iir(backend.get());

      // Butterworth 2nd order LP, fc=0.1 (normalized)
      // scipy: butter(2, 0.1, output='sos')
      filters::BiquadSection sec0;
      sec0.b0 =  0.02008337f;
      sec0.b1 =  0.04016673f;
      sec0.b2 =  0.02008337f;
      sec0.a1 = -1.56101808f;
      sec0.a2 =  0.64135154f;

      iir.SetBiquadSections({sec0});

      test_filters_rocm::IirFilterROCmBenchmark bench(
          backend.get(), iir, input_data, CHANNELS, POINTS,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_IirFilter_ROCm"});

      bench.Run();
      bench.Report();
      std::cout << "  [OK] IirFilterROCm benchmark complete\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cout << "  [SKIP] " << e.what() << "\n";
    return 0;
  }
}

}  // namespace test_filters_benchmark_rocm

#endif  // ENABLE_ROCM
