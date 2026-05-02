#pragma once

// ============================================================================
// fft_maxima_benchmark_rocm — базовые классы бенчмарков SpectrumProcessorROCm
//
// ЧТО:    SpectrumProcessorROCmBenchmark + SpectrumProcessorROCmAllMaximaBenchmark —
//         наследники GpuBenchmarkBase для замера производительности spectrum maxima.
// ЗАЧЕМ:  Отделяет benchmark-логику (warmup + measure + Export) от production-кода.
//         Используется из test_fft_maxima_benchmark_rocm.hpp как runner.
// ПОЧЕМУ: Паттерн GpuBenchmarkBase — единый способ профилирования GPU-операций
//         с ProfilingFacade (правило 06).
//
// История: Создан: 2026-03-01
// ============================================================================

/**
 * @file fft_maxima_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для SpectrumProcessorROCm (GpuBenchmarkBase).
 * @note Не публичный API. Запускается через test_fft_maxima_benchmark_rocm.hpp.
 */

#if ENABLE_ROCM

#include <spectrum/processors/spectrum_processor_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <complex>
#include <vector>

namespace test_fft_maxima_rocm {

// ─── Benchmark 1: ProcessFromCPU (ONE_PEAK) ───────────────────────────────────

class SpectrumProcessorROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (ROCm) — для GPUProfiler
   * @param proc    Ссылка на SpectrumProcessorROCm (не владеет)
   * @param input_data Входные данные (фиксированы между запусками)
   * @param cfg     Параметры бенчмарка
   */
  SpectrumProcessorROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_Process"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_Process", cfg),
      proc_(proc), input_data_(input_data) {}

protected:
  /// Warmup: ProcessFromCPU без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    proc_.ProcessFromCPU(input_data_);
  }

  /// Замер: ProcessFromCPU с ROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.ProcessFromCPU(input_data_, &events);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "spectrum/fft", events);
  }

private:
  antenna_fft::SpectrumProcessorROCm&  proc_;
  std::vector<std::complex<float>>     input_data_;
};

// ─── Benchmark 2: FindAllMaximaFromCPU (full pipeline) ────────────────────────

class SpectrumProcessorROCmAllMaximaBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (ROCm) — для GPUProfiler
   * @param proc    Ссылка на SpectrumProcessorROCm (не владеет)
   * @param input_data Входные данные (фиксированы между запусками)
   * @param cfg     Параметры бенчмарка
   */
  SpectrumProcessorROCmAllMaximaBenchmark(
      drv_gpu_lib::IBackend* backend,
      antenna_fft::SpectrumProcessorROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_SpectrumMaxima_ROCm_AllMaxima"})
    : GpuBenchmarkBase(backend, "SpectrumMaxima_ROCm_AllMaxima", cfg),
      proc_(proc), input_data_(input_data) {}

protected:
  /// Warmup: FindAllMaximaFromCPU без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU, 1, 0);
  }

  /// Замер: FindAllMaximaFromCPU с ROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU, 1, 0,
                               &events);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "spectrum/fft", events);
  }

private:
  antenna_fft::SpectrumProcessorROCm&  proc_;
  std::vector<std::complex<float>>     input_data_;
};

}  // namespace test_fft_maxima_rocm

#endif  // ENABLE_ROCM
