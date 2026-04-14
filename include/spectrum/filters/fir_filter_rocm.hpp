#pragma once

/**
 * @file fir_filter_rocm.hpp
 * @brief FirFilterROCm - GPU FIR convolution filter (ROCm/HIP)
 *
 * ROCm port of FirFilter (OpenCL). Same algorithm, HIP runtime:
 * - hiprtc for kernel compilation
 * - void* device pointers instead of cl_mem
 * - hipStream_t instead of cl_command_queue
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * Usage:
 * @code
 * FirFilterROCm fir(rocm_backend);
 * fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});
 *
 * auto result = fir.Process(gpu_input, channels, points);
 * // result.data is void* (HIP device pointer), caller must hipFree()
 * @endcode
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/types/filter_params.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

/// Список событий профилирования ROCm (имя стадии + ROCmProfilingData)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/// @ingroup grp_filters
class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~FirFilterROCm();

  // No copy
  FirFilterROCm(const FirFilterROCm&) = delete;
  FirFilterROCm& operator=(const FirFilterROCm&) = delete;

  // Move
  FirFilterROCm(FirFilterROCm&& other) noexcept;
  FirFilterROCm& operator=(FirFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void LoadConfig(const std::string& json_path);
  void SetCoefficients(const std::vector<float>& coeffs);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Apply FIR filter on GPU (ROCm)
   * @param input_ptr HIP device pointer with [channels * points] complex float
   * @param channels Number of parallel channels
   * @param points Samples per channel
   * @return InputData<void*> with filtered signal (caller must hipFree result.data)
   * @note input_ptr is NOT freed by this method
   */
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief Apply FIR filter from CPU data (upload + process + keep on GPU)
   */
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief CPU reference implementation (for validation)
   */
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points);

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  uint32_t GetNumTaps() const { return static_cast<uint32_t>(coefficients_.size()); }
  const std::vector<float>& GetCoefficients() const { return coefficients_; }
  bool IsReady() const { return compiled_ && !coefficients_.empty(); }

private:
  void EnsureCompiled();
  void UploadCoefficients();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  std::vector<float> coefficients_;
  bool compiled_ = false;

  // GPU buffer for coefficients (persistent)
  void* coeff_buf_ = nullptr;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace filters

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <spectrum/types/filter_params.hpp>

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend*) {}
  ~FirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  void SetCoefficients(const std::vector<float>&) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  uint32_t GetNumTaps() const { return 0; }
  const std::vector<float>& GetCoefficients() const {
    static std::vector<float> empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
