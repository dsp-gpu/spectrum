#pragma once

/**
 * @file kaufman_filter_rocm.hpp
 * @brief KaufmanFilterROCm - GPU Kaufman Adaptive Moving Average (ROCm/HIP)
 *
 * KAMA adapts smoothing speed automatically:
 *   Trend signal (ER≈1) → fast reaction
 *   Noise signal (ER≈0) → slow reaction (almost frozen)
 *
 * Grid: 1D — one thread per channel, sequential loop.
 * Ring buffer size = er_period, compiled via hiprtc -DN_WINDOW=<er_period> (lazy, per-N cache).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "interface/gpu_context.hpp"
#include "types/filter_params.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <cstdint>

namespace filters {

/// @ingroup grp_filters
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

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "types/filter_params.hpp"

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
