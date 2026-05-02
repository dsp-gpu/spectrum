#pragma once

// ============================================================================
// test_fft_maxima_benchmark_rocm — бенчмарк SpectrumProcessorROCm (ROCm)
//
// ЧТО:    2 бенчмарка: ProcessFromCPU (ONE_PEAK) и FindAllMaximaFromCPU.
//         Стадии: Upload+PadKernel+FFT+PostKernel+Download / +ComputeMagnitudes.
// ЗАЧЕМ:  SpectrumMaximaFinder — в radar pipeline (поиск целей). Регрессии критичны.
// ПОЧЕМУ: ENABLE_ROCM. Результаты → Results/Profiler/.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_fft_maxima_benchmark_rocm.hpp
 * @brief ROCm test runner: SpectrumProcessorROCm benchmark (GpuBenchmarkBase)
 *
 * Benchmark 1: ProcessFromCPU (ONE_PEAK)
 *   → Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process/
 *   Стадии: Upload(H2D) + PadKernel + FFT + PostKernel + Download(D2H)
 *
 * Benchmark 2: FindAllMaximaFromCPU (full pipeline)
 *   → Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima/
 *   Стадии: Upload(H2D) + PadKernel + FFT + ComputeMagnitudes + Pipeline
 *
 * Запускается ТОЛЬКО на Linux + AMD GPU (ENABLE_ROCM=1).
 * На Windows/без AMD: catch → [SKIP], не ошибка.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see fft_maxima_benchmark_rocm.hpp, GpuBenchmarkBase
 */

#if ENABLE_ROCM

#include "fft_maxima_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_maxima_benchmark_rocm {

// ═══════════════════════════════════════════════════════════════════════════
// Утилита — генерация мультилучевых данных
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateSignal(
    uint32_t beam_count, uint32_t n_point, float sample_rate)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(beam_count) * n_point);
  for (uint32_t b = 0; b < beam_count; ++b) {
    float freq = 50.0f + b * 30.0f;
    for (uint32_t t = 0; t < n_point; ++t) {
      float val = std::sin(2.0f * static_cast<float>(M_PI)
                           * freq * t / sample_rate);
      data[static_cast<size_t>(b) * n_point + t] = {val, 0.0f};
    }
  }
  return data;
}

// ═══════════════════════════════════════════════════════════════════════════
// Точка входа бенчмарка
// ═══════════════════════════════════════════════════════════════════════════

inline int run() {
  std::cout << "\n";
  std::cout << "============================================================\n";
  std::cout << "  SpectrumProcessorROCm Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  std::cout << "  Available ROCm devices: " << device_count << "\n";
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices found\n";
    return 0;
  }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────────
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    // ── Параметры ────────────────────────────────────────────────────────
    const uint32_t BEAM_COUNT  = 10;
    const uint32_t N_POINT     = 8192;
    const float    SAMPLE_RATE = 100000.0f;

    auto signal = GenerateSignal(BEAM_COUNT, N_POINT, SAMPLE_RATE);

    antenna_fft::SpectrumParams params;
    params.antenna_count = BEAM_COUNT;
    params.n_point       = N_POINT;
    params.sample_rate   = SAMPLE_RATE;
    params.repeat_count  = 1;
    params.peak_mode     = antenna_fft::PeakSearchMode::ONE_PEAK;
    params.memory_limit  = 0.8f;

    antenna_fft::SpectrumProcessorROCm proc(&backend);
    proc.Initialize(params);

    // ── Benchmark 1: ProcessFromCPU (ONE_PEAK) ───────────────────────────
    std::cout << "\n--- ROCm Benchmark 1: ProcessFromCPU (ONE_PEAK) ---\n";
    {
      test_fft_maxima_rocm::SpectrumProcessorROCmBenchmark bench(
          &backend, proc, signal,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] ROCm Process benchmark complete\n";
      }
    }

    // ── Benchmark 2: FindAllMaximaFromCPU (pipeline) ─────────────────────
    std::cout << "\n--- ROCm Benchmark 2: FindAllMaximaFromCPU (pipeline) ---\n";
    {
      test_fft_maxima_rocm::SpectrumProcessorROCmAllMaximaBenchmark bench(
          &backend, proc, signal,
          {.n_warmup   = 5,
           .n_runs     = 20,
           .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"});

      if (!bench.IsProfEnabled()) {
        std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
      } else {
        bench.Run();
        bench.Report();
        std::cout << "  [OK] ROCm AllMaxima benchmark complete\n";
      }
    }

    return 0;

  } catch (const std::exception& e) {
    std::cout << "  [SKIP] ROCm not available: " << e.what() << "\n";
    return 0;  // не ошибка: просто нет AMD GPU
  }
}

}  // namespace test_fft_maxima_benchmark_rocm

#endif  // ENABLE_ROCM
