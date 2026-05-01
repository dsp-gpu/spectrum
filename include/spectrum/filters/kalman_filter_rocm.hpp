#pragma once

// ============================================================================
// KalmanFilterROCm — GPU 1D скалярный фильтр Калмана (ROCm/HIP)
//
// ЧТО:    Применяет независимый скалярный фильтр Калмана к Re и Im частям
//         комплексного сигнала [channels × points × complex<float>].
//         Один поток обрабатывает один канал последовательно (predict-update
//         loop по samples). State (x, P) переменные хранятся в регистрах
//         потока, не в global memory.
//
// ЗАЧЕМ:  В радар-пайплайне Калман нужен для:
//           - сглаживания f_beat (range estimation),
//           - фильтрации амплитуды по антеннам,
//           - углового tracking'а.
//         CPU-реализация на 1024+ каналах = bottleneck, GPU параллелит по
//         каналам, оставляя последовательность по времени внутри канала.
//
// ПОЧЕМУ: - Скалярный 1D Kalman (а не векторный) — Re и Im обрабатываются
//           независимо. Это упрощение оправдано: для f_beat хватает
//           скалярного state-space, а векторный требовал бы 4×4 матрицы и
//           регистрового spill'а.
//         - Grid 1D, 1 thread = 1 channel — рекурсия x[n] = f(x[n-1])
//           параллелится только между каналами, не внутри.
//         - cached_input_buf_ — переиспользование GPU-буфера для одинаковых
//           размеров: hipMalloc/hipFree (~0.5ms каждый) убраны из hot-path
//           ProcessFromCPU. Освобождается в ReleaseGpuResources / dtor.
//         - block_size_ конфигурируемый (default 256) — на коротких channels
//           уменьшение даёт лучшую occupancy для radar-сценариев.
//
// Использование:
//   filters::KalmanFilterROCm kf(rocm_backend);
//   kf.SetParams(/*Q=*/1e-3f, /*R=*/0.1f, /*x0=*/0.0f, /*P0=*/25.0f);
//   auto result = kf.ProcessFromCPU(data, channels, points);
//   // result.data — void*, caller обязан hipFree
//
// История:
//   - Создан:  2026-03-01 (1D Kalman для radar range estimation)
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
 * @class KalmanFilterROCm
 * @brief GPU 1D скалярный фильтр Калмана (Re/Im независимо), one-thread-per-channel.
 *
 * @note Move-only. Требует #if ENABLE_ROCM (CPU-only сборки получают stub).
 * @note Параллелизм только по каналам (рекурсия по времени внутри канала).
 * @note cached_input_buf_ переиспользуется при одинаковых размерах (hot-path optimization).
 * @note Не thread-safe (один экземпляр = один владелец GPU-ресурсов).
 * @see filters::KaufmanFilterROCm — адаптивная альтернатива (KAMA)
 * @ingroup grp_filters
 */
class KalmanFilterROCm {
public:
  explicit KalmanFilterROCm(drv_gpu_lib::IBackend* backend,
                             unsigned int block_size = 256);
  ~KalmanFilterROCm();

  // No copy
  KalmanFilterROCm(const KalmanFilterROCm&) = delete;
  KalmanFilterROCm& operator=(const KalmanFilterROCm&) = delete;

  // Move
  KalmanFilterROCm(KalmanFilterROCm&& other) noexcept;
  KalmanFilterROCm& operator=(KalmanFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void SetParams(const KalmanParams& params);
  void SetParams(float Q, float R, float x0 = 0.0f, float P0 = 25.0f);

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

  const KalmanParams& GetParams() const { return params_; }
  bool IsReady() const { return compiled_; }

private:
  void EnsureCompiled();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  bool compiled_ = false;

  KalmanParams params_;  ///< Текущие параметры Q, R, x0, P0

  // Кешированный input-буфер — переиспользуется при одинаковом размере,
  // чтобы избежать hipMalloc/hipFree (~0.5 мс) на каждый ProcessFromCPU
  void*  cached_input_buf_  = nullptr;  ///< GPU-буфер [channels * points] float2, принадлежит объекту
  size_t cached_input_size_ = 0;        ///< Размер cached_input_buf_ в байтах

  unsigned int block_size_ = 256;  ///< Threads per block (1 thread per channel)
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

class KalmanFilterROCm {
public:
  explicit KalmanFilterROCm(drv_gpu_lib::IBackend*, unsigned int = 256) {}
  ~KalmanFilterROCm() = default;

  void SetParams(const KalmanParams&) {
    throw std::runtime_error("KalmanFilterROCm: ROCm not enabled");
  }
  void SetParams(float, float, float = 0.0f, float = 25.0f) {
    throw std::runtime_error("KalmanFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t) {
    throw std::runtime_error("KalmanFilterROCm: ROCm not enabled");
  }
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("KalmanFilterROCm: ROCm not enabled");
  }
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) const {
    throw std::runtime_error("KalmanFilterROCm: ROCm not enabled");
  }

  const KalmanParams& GetParams() const {
    static KalmanParams empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
