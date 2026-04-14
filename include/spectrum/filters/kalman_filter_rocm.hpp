#pragma once

/**
 * @file kalman_filter_rocm.hpp
 * @brief KalmanFilterROCm - GPU 1D scalar Kalman filter (ROCm/HIP)
 *
 * Applies independent scalar Kalman filters to Re and Im parts.
 * Grid: 1D — one thread per channel, sequential predict-update loop.
 *
 * Application: smoothing f_beat (radar range estimation),
 * amplitude filtering, angular tracking.
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
