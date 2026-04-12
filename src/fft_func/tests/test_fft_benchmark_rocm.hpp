#pragma once

/**
 * @file test_fft_benchmark_rocm.hpp
 * @brief Бенчмарк FFTProcessorROCm через GpuBenchmarkBase
 *
 * Запускает:
 *  - 5 прогревочных FFT (без замеров — данные warmup сбрасываются)
 *  - 20 замерных FFT → GPUProfiler (min/max/avg автоматически)
 *  - profiler.PrintReport() + ExportJSON + ExportMarkdown
 *
 * Результаты в: Results/Profiler/GPU_00_FFT_ROCm/
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see fft_processor_benchmark_rocm.hpp, GpuBenchmarkBase
 */

#if ENABLE_ROCM

#include "fft_processor_benchmark_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_benchmark_rocm {

// ═══════════════════════════════════════════════════════════════════════════
// Утилита — генерация мультилучевых данных
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> GenerateBenchmarkData(
    size_t beam_count, size_t n_point, float sample_rate,
    float base_freq, float freq_step)
{
  std::vector<std::complex<float>> data(beam_count * n_point);
  for (size_t b = 0; b < beam_count; ++b) {
    float freq = base_freq + b * freq_step;
    for (size_t i = 0; i < n_point; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      data[b * n_point + i] = std::complex<float>(
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
  std::cout << "  FFTProcessorROCm Benchmark (GpuBenchmarkBase)\n";
  std::cout << "============================================================\n";

  // Проверка наличия ROCm-устройств
  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  std::cout << "  Available ROCm devices: " << device_count << "\n";
  if (device_count == 0) {
    std::cout << "  [SKIP] No ROCm devices found\n";
    return 0;
  }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────
    drv_gpu_lib::ROCmBackend backend;
    backend.Initialize(0);

    // ── Параметры бенчмарка ───────────────────────────────────────────
    fft_processor::FFTProcessorParams params;
    params.beam_count  = 64;
    params.n_point     = 1024;
    params.sample_rate = 1e6f;
    params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

    auto input_data = GenerateBenchmarkData(
        params.beam_count, params.n_point,
        params.sample_rate, 100e3f, 10e3f);

    // ── Создать процессор и бенчмарк ──────────────────────────────────
    fft_processor::FFTProcessorROCm proc(&backend);

    test_fft_processor_rocm::FFTProcessorBenchmarkROCm bench(
        &backend, proc, params, input_data,
        {.n_warmup   = 5,
         .n_runs     = 20,
         .output_dir = "Results/Profiler/GPU_00_FFT_ROCm"});

    // ── Запуск ────────────────────────────────────────────────────────
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

}  // namespace test_fft_benchmark_rocm

#endif  // ENABLE_ROCM
