#pragma once

// ============================================================================
// LchFarrowROCm — fractional-delay интерполятор Лагранжа 48×5 (ROCm/HIP)
//
// ЧТО:    ROCm-порт OpenCL-варианта LchFarrow. Тот же алгоритм (5-точечная
//         интерполяция Лагранжа, матрица 48×5), но через HIP runtime:
//         hipModule + hipStream + void* device pointers вместо cl_program +
//         cl_command_queue + cl_mem. Plus ProcessFromCPU (загрузка + обработка
//         за один вызов) как удобство для Python-биндингов.
//
// ЗАЧЕМ:  Main-ветка DSP-GPU работает на Linux + AMD + ROCm 7.2+ (правило
//         09-rocm-only). LchFarrow (OpenCL) — legacy nvidia-ветки, не работает
//         на RDNA4. LchFarrowROCm — production-вариант для всех современных
//         AMD GPU. API специально совместим с LchFarrow для лёгкой замены.
//
// ПОЧЕМУ: - GpuContext (Layer 1 Ref03) → переиспользует skомpiled module
//           через KernelCacheService между вызовами Process. EnsureCompiled
//           делает hipModuleLoad один раз, дальше hot-path без overhead.
//         - Move-only (deleted copy) — GPU-ресурсы (matrix_buf_, delay_buf_,
//           hipModule_t) уникальны на инстанс.
//         - kBlockSize=256 — оптимум для warp=64 на RDNA4 (правило 13).
//         - Stub-секция #else (Windows без ROCm) — все методы throw, так что
//           код, инклюдящий этот заголовок, компилируется кросс-платформенно
//           (только бросает в runtime). Это для Python-биндингов: одна сборка.
//         - Read-only profiling: ROCmProfEvents — лист пар (имя, ROCmProfilingData),
//           caller передаёт указатель если нужно профилирование, иначе
//           nullptr → zero overhead (без замеров hipEvent).
//
// Использование:
//   lch_farrow::LchFarrowROCm processor(rocm_backend);
//   processor.SetDelays({0.0f, 1.5f, 3.0f});
//   processor.SetSampleRate(1e6f);
//
//   auto result = processor.Process(gpu_input, antennas, points);
//   // result.data — void* (HIP device pointer); caller вызывает hipFree.
//
//   // Или одним вызовом из CPU:
//   auto result = processor.ProcessFromCPU(cpu_data, antennas, points);
//
// История:
//   - Создан: 2026-02-23 (порт OpenCL-варианта на ROCm для main-ветки)
// ============================================================================

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <utility>

namespace lch_farrow {

/// ROCm profiling events collected during Process() / ProcessFromCPU() (optional)
/// Pass non-null to benchmark; null = production (zero overhead)
using ROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class LchFarrowROCm
 * @brief ROCm/HIP fractional-delay процессор (Lagrange 48×5).
 *
 * @note Move-only: GPU-ресурсы уникальны.
 * @note Требует #if ENABLE_ROCM. На non-ROCm платформах — stub (см. ниже).
 * @note API совместим с lch_farrow::LchFarrow для прозрачной замены backend'а.
 * @see lch_farrow::LchFarrow (legacy OpenCL)
 * @see drv_gpu_lib::GpuContext (Layer 1 Ref03)
 * @ingroup grp_lch_farrow
 */
class LchFarrowROCm {
public:
  explicit LchFarrowROCm(drv_gpu_lib::IBackend* backend);
  ~LchFarrowROCm();

  // No copy
  LchFarrowROCm(const LchFarrowROCm&) = delete;
  LchFarrowROCm& operator=(const LchFarrowROCm&) = delete;

  // Move
  LchFarrowROCm(LchFarrowROCm&& other) noexcept;
  LchFarrowROCm& operator=(LchFarrowROCm&& other) noexcept;

  /// Set per-antenna delays in microseconds
  void SetDelays(const std::vector<float>& delay_us);

  /// Set sample rate (Hz)
  void SetSampleRate(float sample_rate);

  /// Set noise parameters (amplitude and seed; 0 = no noise)
  void SetNoise(float noise_amplitude, float norm_val = 0.7071067811865476f,
                uint32_t noise_seed = 0);

  /**
   * @brief Load Lagrange matrix from JSON file
   * Uses built-in 48x5 matrix by default.
   */
  void LoadMatrix(const std::string& json_path);

  /**
   * @brief Apply fractional delay on GPU (ROCm)
   * @param input_ptr  HIP device pointer with complex signal [antennas * points]
   * @param antennas   Number of antennas/channels
   * @param points     Samples per antenna
   * @param prof_events Optional: collect ROCm events for profiling (null = production)
   *   - "Upload_delay" : hipMemcpyHtoDAsync (delay_us array)
   *   - "Kernel"       : lch_farrow_delay (hipModuleLaunchKernel)
   * @return InputData<void*> with delayed signal (caller must hipFree result.data)
   * @note input_ptr is NOT freed by this method
   */
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t antennas, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief Apply fractional delay from CPU data (upload + process + keep on GPU)
   * @param data       Flat complex signal: antennas * points elements
   * @param antennas   Number of antennas
   * @param points     Samples per antenna
   * @param prof_events Optional: collect ROCm events for profiling (null = production)
   *   - "Upload_input" : hipMemcpyHtoDAsync (input signal)
   *   - "Upload_delay" : hipMemcpyHtoDAsync (delay_us array)    [via Process()]
   *   - "Kernel"       : lch_farrow_delay                        [via Process()]
   * @return InputData<void*> with delayed signal on GPU
   */
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t antennas, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief Apply fractional delay on CPU (reference)
   * Identical to LchFarrow::ProcessCpu.
   */
  std::vector<std::vector<std::complex<float>>> ProcessCpu(
      const std::vector<std::vector<std::complex<float>>>& input,
      uint32_t antennas, uint32_t points);

  // Getters
  const std::vector<float>& GetDelays() const { return delay_us_; }
  float GetSampleRate() const { return sample_rate_; }

private:
  void EnsureCompiled();
  void UploadMatrix();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;  ///< Ref03: compilation, stream, disk cache

  // Parameters
  std::vector<float> delay_us_;
  float sample_rate_ = 1e6f;
  float noise_amplitude_ = 0.0f;
  float norm_val_ = 0.7071067811865476f;
  uint32_t noise_seed_ = 0;

  // Lagrange matrix 48x5 (240 floats)
  std::vector<float> lagrange_matrix_;
  bool matrix_loaded_ = false;
  bool compiled_ = false;

  // GPU buffer for matrix (persistent, small — 240 floats)
  void* matrix_buf_ = nullptr;

  // GPU buffer for delay_us (persistent, resized on demand)
  void* delay_buf_ = nullptr;
  size_t delay_buf_size_ = 0;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace lch_farrow

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace lch_farrow {

class LchFarrowROCm {
public:
  explicit LchFarrowROCm(drv_gpu_lib::IBackend*) {}
  ~LchFarrowROCm() = default;

  void SetDelays(const std::vector<float>&) {}
  void SetSampleRate(float) {}
  void SetNoise(float, float = 0.0f, uint32_t = 0) {}
  void LoadMatrix(const std::string&) {
    throw std::runtime_error("LchFarrowROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t,
      void* = nullptr) {
    throw std::runtime_error("LchFarrowROCm: ROCm not enabled");
  }
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t,
      void* = nullptr) {
    throw std::runtime_error("LchFarrowROCm: ROCm not enabled");
  }
  std::vector<std::vector<std::complex<float>>> ProcessCpu(
      const std::vector<std::vector<std::complex<float>>>&, uint32_t, uint32_t) {
    throw std::runtime_error("LchFarrowROCm: ROCm not enabled");
  }

  const std::vector<float>& GetDelays() const {
    static std::vector<float> empty;
    return empty;
  }
  float GetSampleRate() const { return 0.0f; }
};

}  // namespace lch_farrow

#endif  // ENABLE_ROCM
