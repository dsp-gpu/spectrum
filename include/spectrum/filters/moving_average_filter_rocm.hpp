#pragma once

/**
 * @file moving_average_filter_rocm.hpp
 * @brief MovingAverageFilterROCm - GPU moving average filters (ROCm/HIP)
 *
 * Supports: SMA, EMA, MMA, DEMA, TEMA
 * Input: complex<float> (float2_t) — multi-channel IQ signals
 * Grid: 1D — one thread per channel, sequential loop over points
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

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

/// @ingroup grp_filters
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
