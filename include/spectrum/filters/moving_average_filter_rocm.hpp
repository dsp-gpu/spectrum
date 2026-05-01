#pragma once

// ============================================================================
// MovingAverageFilterROCm — GPU скользящие средние (ROCm/HIP)
//
// ЧТО:    Семейство скользящих средних на GPU: SMA, EMA, MMA, DEMA, TEMA.
//         Применяется к комплексному multi-channel IQ сигналу
//         [channels × points × complex<float>]. Тип фильтра выбирается через
//         SetParams(MAType, window_size).
//
// ЗАЧЕМ:  Базовый строительный блок ЦОС: сглаживание, denoising, тренд-detection.
//         GPU-версия нужна для multi-channel радара (1024+ каналов параллельно)
//         где CPU становится bottleneck'ом. Поддержка нескольких типов в одном
//         классе — вместо плодить FirFilter под каждый MA.
//
// ПОЧЕМУ: - Один класс на 5 типов MA (а не 5 классов) — у них одинаковый
//           pipeline (input → kernel → output), отличается только формула
//           внутри kernel. Выбор kernel host-side по MAType (zero overhead).
//         - alpha_ precomputed в SetParams() — для EMA = 2/(N+1), MMA = 1/N;
//           внутри kernel — простое умножение без деления.
//         - SMA window ограничен 128 — больше требует ring-buffer в shared
//           memory, превышает доступный объём для batch_size=256.
//         - compiled_sma_window_ — re-compile kernel через hiprtc при смене
//           window_size для SMA (template parameter N_WINDOW).
//         - cached_input_buf_ — re-use H2D buffer (избавляет от ~0.5ms
//           hipMalloc/hipFree на каждый ProcessFromCPU).
//         - Grid 1D, 1 thread = 1 channel — рекурсивный pass по samples.
//
// Использование:
//   filters::MovingAverageFilterROCm ma(rocm_backend);
//   ma.SetParams(MAType::EMA, /*window_size=*/10);
//   auto result = ma.ProcessFromCPU(data, channels, points);
//
// История:
//   - Создан:  2026-03-01 (5 типов MA в одном классе)
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
 * @class MovingAverageFilterROCm
 * @brief GPU скользящие средние: SMA / EMA / MMA / DEMA / TEMA в одном классе.
 *
 * @note Move-only. Требует #if ENABLE_ROCM (CPU-only сборки получают stub).
 * @note Параллелизм только по каналам (рекурсия по времени внутри канала).
 * @note SMA: max window_size = 128 (shared memory limit).
 * @note Per-N kernel cache для SMA: смена window_size вызывает recompile.
 * @note cached_input_buf_ переиспользуется при одинаковых размерах.
 * @see filters::KaufmanFilterROCm — адаптивная альтернатива (KAMA)
 * @ingroup grp_filters
 */
class MovingAverageFilterROCm {
public:
  explicit MovingAverageFilterROCm(drv_gpu_lib::IBackend* backend,
                                   unsigned int block_size = 256);
  ~MovingAverageFilterROCm();

  // No copy
  MovingAverageFilterROCm(const MovingAverageFilterROCm&) = delete;
  MovingAverageFilterROCm& operator=(const MovingAverageFilterROCm&) = delete;

  // Move
  MovingAverageFilterROCm(MovingAverageFilterROCm&& other) noexcept;
  MovingAverageFilterROCm& operator=(MovingAverageFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void SetParams(const MovingAverageParams& params);
  void SetParams(MAType type, uint32_t window_size);

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

  MAType   GetType()       const { return ma_type_; }
  uint32_t GetWindowSize() const { return window_size_; }
  bool     IsReady()       const { return compiled_; }

private:
  void EnsureKernels();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;

  MAType   ma_type_     = MAType::EMA;  ///< Текущий тип скользящей средней
  uint32_t window_size_ = 10;           ///< N — размер окна (для SMA: max 128)
  float    alpha_        = 2.0f / 11.0f;  ///< Сглаживающий коэффициент: EMA=2/(N+1), MMA=1/N. Precomputed в SetParams()

  // Кешированный input-буфер на GPU — переиспользуется если размер совпадает,
  // чтобы избежать дорогого hipMalloc/hipFree (~0.5 мс) на каждый вызов ProcessFromCPU
  void*  cached_input_buf_  = nullptr;  ///< GPU-буфер [channels * points] float2, принадлежит объекту
  size_t cached_input_size_ = 0;        ///< Текущий размер cached_input_buf_ в байтах

  unsigned int block_size_          = 256;  ///< Threads per block для hipModuleLaunchKernel (1 thread per channel)
  uint32_t     compiled_sma_window_ = 0;   ///< N_WINDOW used in last EnsureKernels(); 0 = not compiled
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

class MovingAverageFilterROCm {
public:
  explicit MovingAverageFilterROCm(drv_gpu_lib::IBackend*, unsigned int = 256) {}
  ~MovingAverageFilterROCm() = default;

  void SetParams(const MovingAverageParams&) {
    throw std::runtime_error("MovingAverageFilterROCm: ROCm not enabled");
  }
  void SetParams(MAType, uint32_t) {
    throw std::runtime_error("MovingAverageFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t) {
    throw std::runtime_error("MovingAverageFilterROCm: ROCm not enabled");
  }
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("MovingAverageFilterROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) const {
    throw std::runtime_error("MovingAverageFilterROCm: ROCm not enabled");
  }

  MAType   GetType()       const { return MAType::EMA; }
  uint32_t GetWindowSize() const { return 0; }
  bool     IsReady()       const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
