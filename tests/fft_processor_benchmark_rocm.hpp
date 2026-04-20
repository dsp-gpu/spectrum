#pragma once

/**
 * @file fft_processor_benchmark_rocm.hpp
 * @brief FFTProcessorBenchmarkROCm — наследник GpuBenchmarkBase для FFTProcessorROCm
 *
 * Пример использования:
 * @code
 *   FFTProcessorROCm proc(backend);
 *   FFTProcessorBenchmarkROCm bench(backend, proc, params, input_data);
 *   bench.Run();     // warmup(5) + measure(20) → GPUProfiler
 *   bench.Report();  // profiler.PrintReport() + ExportJSON + ExportMarkdown
 * @endcode
 *
 * FFTProcessorROCm — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование через опциональный prof_events (ROCmProfEvents*):
 *  - ExecuteKernel()      → ProcessComplex(data, params)         — без событий
 *  - ExecuteKernelTimed() → ProcessComplex(data, params, &events) — с hipEvent_t
 *    → ProfilingFacade::BatchRecord(events) — одним вызовом в profiler v2
 *
 * Stages: Upload (H2D), Pad (kernel), FFT (hipfftExecC2C), Download (D2H)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, MemoryBank/specs/Profil_GPU.md
 */

#if ENABLE_ROCM

#include <spectrum/fft_processor_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>
#include <core/services/profiling/profiling_facade.hpp>

#include <complex>
#include <vector>

namespace test_fft_processor_rocm {

class FFTProcessorBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @brief Конструктор
   * @param backend IBackend (ROCm) — для инициализации GPUProfiler
   * @param proc Ссылка на чистый FFTProcessorROCm (не владеет)
   * @param params Параметры FFT (фиксированы на весь бенчмарк)
   * @param input_data Входные данные (фиксированы между запусками)
   * @param cfg Параметры бенчмарка (n_warmup=5, n_runs=20)
   */
  FFTProcessorBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      fft_processor::FFTProcessorROCm& proc,
      const fft_processor::FFTProcessorParams& params,
      const std::vector<std::complex<float>>& input_data,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "Results/Profiler/GPU_00_FFT_ROCm"})
    : GpuBenchmarkBase(backend, "FFTProcessorROCm", cfg),
      proc_(proc),
      params_(params),
      input_data_(input_data) {}

protected:
  /**
   * @brief Warmup — запуск FFT БЕЗ timing
   *
   * ProcessComplex() без prof_events → GPU прогревается (JIT, clock ramp-up).
   * Нулевой overhead.
   */
  void ExecuteKernel() override {
    proc_.ProcessComplex(input_data_, params_);  // prof_events = nullptr
  }

  /**
   * @brief Замер — запуск FFT С timing → ProfilingFacade::BatchRecord
   *
   * ProcessComplex() с prof_events → собирает ROCmProfilingData:
   *   Upload (H2D), Pad (kernel), FFT (hipfftExecC2C), Download (D2H)
   * Все события передаются одним вызовом BatchRecord() в новый
   * ProfilingFacade (profiler v2). Одно сообщение в queue вместо N —
   * меньше contention (W1 из Round 3 ревью).
   */
  void ExecuteKernelTimed() override {
    fft_processor::ROCmProfEvents events;
    proc_.ProcessComplex(input_data_, params_, &events);

    drv_gpu_lib::profiling::ProfilingFacade::GetInstance()
        .BatchRecord(gpu_id_, "spectrum/fft", events);
  }

private:
  fft_processor::FFTProcessorROCm&      proc_;
  fft_processor::FFTProcessorParams     params_;
  std::vector<std::complex<float>>      input_data_;
};

}  // namespace test_fft_processor_rocm

#endif  // ENABLE_ROCM
