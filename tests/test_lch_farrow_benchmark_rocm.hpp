#pragma once

// ============================================================================
// test_lch_farrow_benchmark_rocm — бенчмарк LchFarrowROCm (GpuBenchmarkBase)
//
// ЧТО:    5 warmup + 20 замерных → ProfilingFacade. 8 антенн × 4096 точек,
//         delays={0.3..7.9}мкс, sample_rate=1MHz.
// ЗАЧЕМ:  LchFarrow — polyphase+Farrow fractional resampling для antenna delays.
//         Регрессии производительности критичны для radar pipeline.
// ПОЧЕМУ: ENABLE_ROCM. GpuBenchmarkBase. Результаты → Results/Profiler/.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_lch_farrow_benchmark_rocm.hpp
 * @brief Бенчмарк LchFarrowROCm через GpuBenchmarkBase (ROCm/HIP)
 *
 * Запускает:
 *  - 5 прогревочных итераций (без замеров — warmup сбрасывается)
 *  - 20 замерных итераций → GPUProfiler (min/max/avg автоматически)
 *  - bench.Report() → profiler.PrintReport() + ExportJSON + ExportMarkdown
 *
 * Параметры:
 *  - antennas  = 8, points = 4096, sample_rate = 1 MHz
 *  - delays    = {0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9} мкс
 *
 * Профилируемые стейджи: Upload_input, Upload_delay, Kernel
 * Результаты: Results/Profiler/GPU_00_LchFarrow_ROCm/
 *
 * Пропускается если нет ROCm-устройств (ENABLE_ROCM не определён или нет AMD GPU).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see lch_farrow_benchmark_rocm.hpp, GpuBenchmarkBase
 */

#if ENABLE_ROCM

#include "lch_farrow_benchmark_rocm.hpp"
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/backends/rocm/rocm_core.hpp>

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_lch_farrow_benchmark_rocm {

// ═══════════════════════════════════════════════════════════════════════════
// Утилита — генерация CW сигнала (flat: antennas × points)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateBenchmarkData(
    uint32_t antennas, uint32_t points, float fs, float freq)
{
  std::vector<std::complex<float>> data(
      static_cast<size_t>(antennas) * points);
  for (uint32_t a = 0; a < antennas; ++a) {
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / fs;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      data[a * points + n] = std::complex<float>(
          std::cos(phase), std::sin(phase));
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
  std::cout << "  LchFarrowROCm Benchmark (GpuBenchmarkBase, HIP)\n";
  std::cout << "============================================================\n";

  // Проверка наличия ROCm-устройств
  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  std::cout << "  Available ROCm devices: " << device_count << "\n";
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices found\n";
    return 0;
  }

  try {
    // ── ROCm backend ─────────────────────────────────────────────────
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    // ── Параметры бенчмарка ──────────────────────────────────────────
    const uint32_t antennas    = 8;
    const uint32_t points      = 4096;
    const float    sample_rate = 1e6f;
    const float    freq        = 50e3f;

    std::vector<float> delays = {0.3f, 1.7f, 2.1f, 3.5f, 4.0f, 5.3f, 6.7f, 7.9f};

    auto input_data = GenerateBenchmarkData(antennas, points, sample_rate, freq);

    // ── Создать LchFarrowROCm ────────────────────────────────────────
    lch_farrow::LchFarrowROCm proc(&backend);
    proc.SetDelays(delays);
    proc.SetSampleRate(sample_rate);

    // ── Создать бенчмарк ─────────────────────────────────────────────
    test_lch_farrow_rocm::LchFarrowBenchmarkROCm bench(
        &backend, proc, input_data, antennas, points,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "../Results/Profiler/lch_farrow"});

    // ── Запуск ───────────────────────────────────────────────────────
    if (!bench.IsProfEnabled()) {
      std::cout << "  [SKIP] is_prof=false in configGPU.json\n";
    } else {
      bench.Run();     // warmup(5) + measure(20) → GPUProfiler
      bench.Report();  // profiler.PrintReport() + ExportJSON + ExportMarkdown
      std::cout << "  [OK] Benchmark complete\n";
    }

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "  FATAL: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace test_lch_farrow_benchmark_rocm

#endif  // ENABLE_ROCM
