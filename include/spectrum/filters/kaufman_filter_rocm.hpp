#pragma once

// ============================================================================
// KaufmanFilterROCm — GPU Kaufman Adaptive Moving Average (KAMA, ROCm/HIP)
//
// ЧТО:    Адаптивное скользящее среднее: автоматически меняет скорость
//         сглаживания по Efficiency Ratio (ER):
//           - Trend сигнал (ER≈1) → быстрая реакция (fast EMA = 2/(2+1) = 2/3)
//           - Noise сигнал (ER≈0) → почти заморожен (slow EMA = 2/(30+1) = 2/31)
//         SC = (ER × (fast_sc − slow_sc) + slow_sc)². Применяется к Re/Im
//         комплексного сигнала независимо.
//
// ЗАЧЕМ:  В радар-задачах сигнал-шум меняется во времени: на короткой
//         дистанции SNR высок (видно цель — нужна быстрая реакция), на
//         дальней — низкий (только шум — нужен почти отрезающий фильтр).
//         Фиксированный EMA даёт компромисс «всегда чуть-чуть мажет», KAMA
//         подстраивается per-sample.
//
// ПОЧЕМУ: - Per-N kernel cache (compiled_window_size_) — kernel ре-компилится
//           через hiprtc с -DN_WINDOW=<er_period> при смене er_period.
//           Внутри ring-buffer фиксированной длины, hot-path не имеет
//           runtime-проверок длины окна.
//         - fast_sc_ / slow_sc_ precomputed в SetParams() — не вычислять
//           2/(N+1) внутри kernel при каждом sample.
//         - cached_input_buf_ — re-use H2D buffer'а если размер совпал
//           (избавляет от ~0.5ms hipMalloc/hipFree на каждый ProcessFromCPU).
//         - Grid 1D, 1 thread = 1 channel — последовательный цикл по
//           samples внутри канала (рекурсивно, parallelism только по каналам).
//         - block_size_ конфигурируемый (default 256) — стандарт RDNA4.
//
// Использование:
//   filters::KaufmanFilterROCm kf(rocm_backend);
//   kf.SetParams(/*er_period=*/10, /*fast=*/2, /*slow=*/30);
//   auto result = kf.ProcessFromCPU(data, channels, points);
//
// История:
//   - Создан:  2026-03-01 (адаптивный KAMA для radar SNR-aware фильтрации)
//   - Изменён: 2026-04-15 (миграция в DSP-GPU/spectrum, GpuContext Ref03)
// ============================================================================

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/types/filter_params.hpp>

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cstdint>

namespace filters {

/**
 * @class KaufmanFilterROCm
 * @brief GPU Kaufman Adaptive Moving Average — ER-адаптивное сглаживание.
 *
 * @note Move-only. Требует #if ENABLE_ROCM (CPU-only сборки получают stub).
 * @note Параллелизм только по каналам (рекурсия по времени внутри канала).
 * @note Per-N kernel cache: смена er_period вызывает recompile через hiprtc.
 * @note cached_input_buf_ переиспользуется при одинаковых размерах.
 * @see filters::KalmanFilterROCm — линейный фильтр для тех же задач
 * @see filters::MovingAverageFilterROCm — простое EMA без адаптации
 * @ingroup grp_filters
 */
class KaufmanFilterROCm {
public:
  explicit KaufmanFilterROCm(drv_gpu_lib::IBackend* backend,
                              unsigned int block_size = 256);
  ~KaufmanFilterROCm();

  // No copy
  KaufmanFilterROCm(const KaufmanFilterROCm&) = delete;
  KaufmanFilterROCm& operator=(const KaufmanFilterROCm&) = delete;

  // Move
  KaufmanFilterROCm(KaufmanFilterROCm&& other) noexcept;
  KaufmanFilterROCm& operator=(KaufmanFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void SetParams(const KaufmanParams& params);
  void SetParams(uint32_t er_period, uint32_t fast_period = 2,
                 uint32_t slow_period = 30);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points);

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels, uint32_t points);

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points) const;

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  const KaufmanParams& GetParams() const { return params_; }
  bool IsReady() const { return compiled_; }

private:
  void EnsureKernel();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;

  KaufmanParams params_;  ///< er_period, fast_period, slow_period
  // fast_sc_ и slow_sc_ — предельные EMA-константы для SC-интерполяции:
  // SC = (ER*(fast_sc-slow_sc)+slow_sc)^2. Precomputed в SetParams()
  // чтобы не вычислять 2/(N+1) внутри kernel при каждом sample.
  float fast_sc_ = 2.0f / 3.0f;   ///< 2/(fast_period+1), default fast=2 → 2/3
  float slow_sc_ = 2.0f / 31.0f;  ///< 2/(slow_period+1), default slow=30 → 2/31

  // Кешированный input-буфер — переиспользуется при одинаковом размере
  void*  cached_input_buf_  = nullptr;  ///< GPU-буфер [channels * points] float2, принадлежит объекту
  size_t cached_input_size_ = 0;        ///< Размер cached_input_buf_ в байтах

  unsigned int block_size_         = 256;  ///< Threads per block (1 thread per channel)
  uint32_t     compiled_window_size_ = 0;  ///< N_WINDOW used in last EnsureKernel(); 0 = not compiled
};

}  // namespace filters

#else  // !ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <spectrum/types/filter_params.hpp>

#include <stdexcept>
#include <vector>
#include <complex>
#include <cstdint>

namespace filters {

class KaufmanFilterROCm {
public:
  explicit KaufmanFilterROCm(drv_gpu_lib::IBackend*, unsigned int = 256) {}
  ~KaufmanFilterROCm() = default;

  void SetParams(const KaufmanParams&) {
    throw std::runtime_error("KaufmanFilterROCm: ROCm not enabled");
  }
  void SetParams(uint32_t, uint32_t = 2, uint32_t = 30) {
    throw std::runtime_error("KaufmanFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t) {
    throw std::runtime_error("KaufmanFilterROCm: ROCm not enabled");
  }
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("KaufmanFilterROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) const {
    throw std::runtime_error("KaufmanFilterROCm: ROCm not enabled");
  }

  const KaufmanParams& GetParams() const {
    static KaufmanParams empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
