#pragma once

// ============================================================================
// filters_benchmark_rocm — базовые классы бенчмарков FirFilterROCm / IirFilterROCm
//
// ЧТО:    FirFilterROCmBenchmark + IirFilterROCmBenchmark — наследники GpuBenchmarkBase.
//         Stages: Upload(H2D) + Kernel.
// ЗАЧЕМ:  Отделяет benchmark-логику (warmup + measure + Export) от production-кода фильтров.
//         Используется из test_filters_benchmark_rocm.hpp как runner.
// ПОЧЕМУ: Паттерн GpuBenchmarkBase — единый способ профилирования GPU-операций (правило 06).
//
// История: Создан: 2026-03-01
// ============================================================================

/**
 * @file filters_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для FirFilterROCm и IirFilterROCm (GpuBenchmarkBase).
 * @note Не публичный API. Запускается через test_filters_benchmark_rocm.hpp.
 */

#if ENABLE_ROCM

#include <spectrum/filters/fir_filter_rocm.hpp>
#include <spectrum/filters/iir_filter_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <complex>
#include <vector>
#include <cstdint>

namespace test_filters_rocm {

// ─── Benchmark 1: FirFilterROCm::ProcessFromCPU ───────────────────────────────

class FirFilterROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend    IBackend (ROCm) — для GPUProfiler
   * @param filter     Ссылка на FirFilterROCm (не владеет, коэффициенты уже загружены)
   * @param input_data Входные данные (фиксированы между прогонами)
   * @param channels   Количество каналов
   * @param points     Количество отсчётов на канал
   * @param cfg        Параметры бенчмарка
   */
  FirFilterROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      filters::FirFilterROCm& filter,
      const std::vector<std::complex<float>>& input_data,
      uint32_t channels,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_FirFilter_ROCm"})
    : GpuBenchmarkBase(backend, "FirFilter_ROCm", cfg),
      filter_(filter), input_data_(input_data),
      channels_(channels), points_(points) {}

protected:
  /// Warmup — ProcessFromCPU без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_);
    hipFree(result.data);
  }

  /// Замер — ProcessFromCPU с ROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    filters::ROCmProfEvents events;
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_, &events);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "spectrum/filters", events);
    hipFree(result.data);
  }

private:
  filters::FirFilterROCm&              filter_;
  std::vector<std::complex<float>>     input_data_;
  uint32_t                             channels_;
  uint32_t                             points_;
};

// ─── Benchmark 2: IirFilterROCm::ProcessFromCPU ───────────────────────────────

class IirFilterROCmBenchmark : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend    IBackend (ROCm) — для GPUProfiler
   * @param filter     Ссылка на IirFilterROCm (не владеет, секции уже загружены)
   * @param input_data Входные данные (фиксированы между прогонами)
   * @param channels   Количество каналов
   * @param points     Количество отсчётов на канал
   * @param cfg        Параметры бенчмарка
   */
  IirFilterROCmBenchmark(
      drv_gpu_lib::IBackend* backend,
      filters::IirFilterROCm& filter,
      const std::vector<std::complex<float>>& input_data,
      uint32_t channels,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {
          .n_warmup   = 5,
          .n_runs     = 20,
          .output_dir = "Results/Profiler/GPU_00_IirFilter_ROCm"})
    : GpuBenchmarkBase(backend, "IirFilter_ROCm", cfg),
      filter_(filter), input_data_(input_data),
      channels_(channels), points_(points) {}

protected:
  /// Warmup — ProcessFromCPU без timing (prof_events = nullptr)
  void ExecuteKernel() override {
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_);
    hipFree(result.data);
  }

  /// Замер — ProcessFromCPU с ROCmProfEvents → ProfilingFacade::BatchRecord
  void ExecuteKernelTimed() override {
    filters::ROCmProfEvents events;
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_, &events);
    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "spectrum/filters", events);
    hipFree(result.data);
  }

private:
  filters::IirFilterROCm&              filter_;
  std::vector<std::complex<float>>     input_data_;
  uint32_t                             channels_;
  uint32_t                             points_;
};

}  // namespace test_filters_rocm

#endif  // ENABLE_ROCM
