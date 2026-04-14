#pragma once

/**
 * @file fft_maxima_benchmark_rocm.hpp
 * @brief ROCm benchmark-классы для SpectrumProcessorROCm (GpuBenchmarkBase)
 *
 * SpectrumProcessorROCmBenchmark         → ProcessFromCPU:
 *   Upload(H2D) + PadKernel + FFT + PostKernel + Download(D2H)
 *
 * SpectrumProcessorROCmAllMaximaBenchmark → FindAllMaximaFromCPU:
 *   Upload(H2D) + PadKernel + FFT + ComputeMagnitudes + Pipeline
 *
 * Запускается ТОЛЬКО на Linux + AMD GPU (ENABLE_ROCM=1).
 * На Windows/без AMD: compile-only, не выполняется.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include <spectrum/processors/spectrum_processor_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>

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

  /// Замер: ProcessFromCPU с ROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.ProcessFromCPU(input_data_, &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
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

  /// Замер: FindAllMaximaFromCPU с ROCmProfEvents → RecordROCmEvent → GPUProfiler
  void ExecuteKernelTimed() override {
    antenna_fft::ROCmProfEvents events;
    proc_.FindAllMaximaFromCPU(input_data_,
                               antenna_fft::OutputDestination::CPU, 1, 0,
                               &events);
    for (auto& [name, data] : events)
      RecordROCmEvent(name, data);
  }

private:
  antenna_fft::SpectrumProcessorROCm&  proc_;
  std::vector<std::complex<float>>     input_data_;
};

}  // namespace test_fft_maxima_rocm

#endif  // ENABLE_ROCM
