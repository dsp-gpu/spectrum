#pragma once

/**
 * @file iir_filter_rocm.hpp
 * @brief IirFilterROCm - GPU IIR biquad cascade filter (ROCm/HIP)
 *
 * ROCm port of IirFilter (OpenCL). Same algorithm, HIP runtime:
 * - hiprtc for kernel compilation
 * - void* device pointers instead of cl_mem
 * - hipStream_t instead of cl_command_queue
 *
 * Compiles ONLY with ENABLE_ROCM=1 (Linux + AMD GPU).
 *
 * Usage:
 * @code
 * IirFilterROCm iir(rocm_backend);
 * iir.SetBiquadSections({{0.02f, 0.04f, 0.02f, -1.56f, 0.64f}});
 *
 * auto result = iir.Process(gpu_input, channels, points);
 * // result.data is void* (HIP device pointer), caller must hipFree()
 * @endcode
 *
 * @note GPU IIR is efficient ONLY with many channels (>= 8).
 *       Single-channel IIR is faster on CPU!
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "interface/gpu_context.hpp"
#include "types/filter_params.hpp"
#include "services/profiling_types.hpp"

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

/// Список событий профилирования ROCm (имя стадии + ROCmProfilingData)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/// @ingroup grp_filters
class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~IirFilterROCm();

  // No copy
  IirFilterROCm(const IirFilterROCm&) = delete;
  IirFilterROCm& operator=(const IirFilterROCm&) = delete;

  // Move
  IirFilterROCm(IirFilterROCm&& other) noexcept;
  IirFilterROCm& operator=(IirFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void LoadConfig(const std::string& json_path);
  void SetBiquadSections(const std::vector<BiquadSection>& sections);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Apply IIR biquad cascade on GPU (ROCm)
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
   * @brief Apply IIR filter from CPU data (upload + process + keep on GPU)
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

  uint32_t GetNumSections() const { return static_cast<uint32_t>(sections_.size()); }
  const std::vector<BiquadSection>& GetSections() const { return sections_; }
  bool IsReady() const { return compiled_ && !sections_.empty(); }

private:
  void EnsureCompiled();
  void UploadSosMatrix();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  std::vector<BiquadSection> sections_;
  bool compiled_ = false;

  // GPU buffer for SOS matrix (persistent)
  void* sos_buf_ = nullptr;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace filters

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include "interface/i_backend.hpp"
#include "interface/input_data.hpp"
#include "types/filter_params.hpp"

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend*) {}
  ~IirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  void SetBiquadSections(const std::vector<BiquadSection>&) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  uint32_t GetNumSections() const { return 0; }
  const std::vector<BiquadSection>& GetSections() const {
    static std::vector<BiquadSection> empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
