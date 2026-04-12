#pragma once

/**
 * @file filters_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для FirFilterROCm и IirFilterROCm (GpuBenchmarkBase)
 *
 * FirFilterROCmBenchmark  → ProcessFromCPU:  Upload(H2D) + Kernel
 * IirFilterROCmBenchmark  → ProcessFromCPU:  Upload(H2D) + Kernel
 *
 * Запускается ТОЛЬКО на Linux + AMD GPU (ENABLE_ROCM=1).
 * На Windows/без AMD: compile-only, не выполняется.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/tasks/TASK_filters_profiling.md
 */

#if ENABLE_ROCM

#include "filters/fir_filter_rocm.hpp"
#include "filters/iir_filter_rocm.hpp"
#include "DrvGPU/services/gpu_benchmark_base.hpp"

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

  /// Замер — ProcessFromCPU с ROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    filters::ROCmProfEvents events;
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
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

  /// Замер — ProcessFromCPU с ROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    filters::ROCmProfEvents events;
    auto result = filter_.ProcessFromCPU(input_data_, channels_, points_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
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
