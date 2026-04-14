#pragma once

/**
 * @file lch_farrow_benchmark_rocm.hpp
 * @brief LchFarrowBenchmarkROCm — наследник GpuBenchmarkBase для LchFarrowROCm
 *
 * LchFarrowROCm — ЧИСТЫЙ production-класс (ноль кода профилирования).
 * Профилирование через опциональный prof_events (ROCmProfEvents*):
 *  - ExecuteKernel()      → ProcessFromCPU(...) — без событий (warmup)
 *  - ExecuteKernelTimed() → ProcessFromCPU(..., &events) — с hipEvent_t
 *    → RecordROCmEvent() для каждого события → GPUProfiler
 *
 * Stages (ProcessFromCPU):
 *  - Upload_input : hipMemcpyHtoDAsync (входной сигнал → GPU)
 *  - Upload_delay : hipMemcpyHtoDAsync (delay_us → GPU)
 *  - Kernel       : lch_farrow_delay (hipModuleLaunchKernel)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-01
 * @see GpuBenchmarkBase, Doc_Addition/GPU_Profiling_Mechanism.md
 */

#if ENABLE_ROCM

#include <spectrum/lch_farrow_rocm.hpp>
#include <core/services/gpu_benchmark_base.hpp>

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cstdint>

namespace test_lch_farrow_rocm {

class LchFarrowBenchmarkROCm : public drv_gpu_lib::GpuBenchmarkBase {
public:
  /**
   * @param backend    IBackend (ROCm) — для GPUProfiler
   * @param proc       Ссылка на LchFarrowROCm (не владеет)
   * @param input_data Входные данные (CPU flat: antennas × points) — фиксированы
   * @param antennas   Число антенн
   * @param points     Число отсчётов на антенну
   * @param cfg        Параметры бенчмарка (n_warmup, n_runs, output_dir)
   */
  LchFarrowBenchmarkROCm(
      drv_gpu_lib::IBackend* backend,
      lch_farrow::LchFarrowROCm& proc,
      const std::vector<std::complex<float>>& input_data,
      uint32_t antennas,
      uint32_t points,
      GpuBenchmarkBase::Config cfg = {.n_warmup   = 5,
                                      .n_runs     = 20,
                                      .output_dir = "../Results/Profiler/lch_farrow"})
    : GpuBenchmarkBase(backend, "LchFarrowROCm", cfg),
      proc_(proc),
      input_data_(input_data),
      antennas_(antennas),
      points_(points) {}

protected:
  /**
   * @brief Warmup — запуск БЕЗ timing (прогрев GPU: JIT, clock ramp-up)
   *
   * ProcessFromCPU() без prof_events → ноль overhead.
   * result.data (GPU буфер) освобождается сразу.
   */
  void ExecuteKernel() override {
    auto result = proc_.ProcessFromCPU(input_data_, antennas_, points_);
    if (result.data) hipFree(result.data);
  }

  /**
   * @brief Замер — запуск С timing → RecordROCmEvent → GPUProfiler
   *
   * ProcessFromCPU() с prof_events → собирает ROCmProfilingData:
   *   Upload_input (H2D), Upload_delay (H2D), Kernel (lch_farrow_delay)
   * Каждое событие записывается через RecordROCmEvent().
   * GPUProfiler копит все вызовы → min/max/avg автоматически.
   */
  void ExecuteKernelTimed() override {
    lch_farrow::ROCmProfEvents events;
    auto result = proc_.ProcessFromCPU(input_data_, antennas_, points_, &events);
    if (result.data) hipFree(result.data);

    for (auto& [name, data] : events) {
      RecordROCmEvent(name, data);
    }
  }

private:
  lch_farrow::LchFarrowROCm&            proc_;
  std::vector<std::complex<float>>      input_data_;
  uint32_t                              antennas_;
  uint32_t                              points_;
};

}  // namespace test_lch_farrow_rocm

#endif  // ENABLE_ROCM
