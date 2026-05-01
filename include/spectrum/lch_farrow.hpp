#pragma once

#if !ENABLE_ROCM  // OpenCL-only: ROCm-сборка использует LchFarrowROCm

// ============================================================================
// LchFarrow — fractional-delay интерполятор Лагранжа 48×5 (OpenCL)
//
// ЧТО:    Самостоятельный модуль для применения дробной задержки к comple-IQ
//         сигналу на GPU. Использует 5-точечную интерполяцию Лагранжа с
//         предвычисленной матрицей коэффициентов 48×5 (240 float).
//
// ЗАЧЕМ:  В радарных pipeline'ах задержки между антеннами — нецелое число
//         сэмплов (зависит от геометрии решётки и угла). Без fractional delay
//         beamforming работает только на сетке сэмплов → big quantization
//         loss. Lagrange 48×5 даёт sub-sample sub-degree точность с малыми
//         накладными расходами.
//
// ПОЧЕМУ: - 48 фаз × 5 точек: 48 интерполяционных позиций между сэмплами
//           (≈ 1/48 сэмпла разрешение) × 5-tap Лагранжа (4-й порядок).
//           Trade-off: точность vs scratch memory (240 float ≈ 960 байт).
//         - Для каждой антенны read_pos = n − delay_samples; если read_pos<0
//           (сигнал ещё не пришёл) → output=0. Это намеренно (causality).
//         - Этот файл — ТОЛЬКО OpenCL-вариант (legacy nvidia-ветка).
//           Под `#if !ENABLE_ROCM`. ROCm/HIP реализация в lch_farrow_rocm.hpp.
//         - Move-only (no copy): GPU-ресурсы (cl_program, cl_kernel, cl_mem)
//           уникальны на инстанс, копирование = double-release.
//
// Использование:
//   lch_farrow::LchFarrow processor(backend);
//   processor.SetDelays({0.0f, 1.5f, 3.0f});  // микросекунды, на антенну
//   processor.SetSampleRate(1e6f);
//   auto result = processor.Process(input_buf, antennas, points);
//   // result.data — cl_mem задержанного сигнала; caller освобождает
//   // через clReleaseMemObject().
//
// История:
//   - Создан: 2026-02-18 (legacy OpenCL-ветка)
// ============================================================================

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>

#include <CL/cl.h>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>
#include <utility>

namespace lch_farrow {

/// OpenCL profiling events collected during Process() (optional)
/// Pass non-null to benchmark; null = production (zero overhead)
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

/**
 * @class LchFarrow
 * @brief OpenCL fractional-delay процессор (Lagrange 48×5).
 *
 * @note Move-only: GPU-ресурсы уникальны на инстанс.
 * @note Не зависит от signal_generators — работает с уже сгенерированным сигналом.
 * @note Доступен только при `#if !ENABLE_ROCM`. ROCm-вариант: LchFarrowROCm.
 * @see lch_farrow::LchFarrowROCm
 */
class LchFarrow {
public:
  explicit LchFarrow(drv_gpu_lib::IBackend* backend);
  ~LchFarrow();

  // No copy
  LchFarrow(const LchFarrow&) = delete;
  LchFarrow& operator=(const LchFarrow&) = delete;

  // Move
  LchFarrow(LchFarrow&& other) noexcept;
  LchFarrow& operator=(LchFarrow&& other) noexcept;

  /// Set per-antenna delays in microseconds
  void SetDelays(const std::vector<float>& delay_us);

  /// Set sample rate (Hz)
  void SetSampleRate(float sample_rate);

  /// Set noise parameters (amplitude and seed; 0 = no noise)
  void SetNoise(float noise_amplitude, float norm_val = 0.7071067811865476f,
                uint32_t noise_seed = 0);

  /**
   * @brief Load Lagrange matrix from JSON file
   * @param json_path Path to JSON (format: { "data": [[...], ...] })
   * Uses built-in 48x5 matrix by default.
   */
  void LoadMatrix(const std::string& json_path);

  /**
   * @brief Apply fractional delay on GPU
   * @param input_buf  cl_mem with complex signal [antennas * points]
   * @param antennas   Number of antennas/channels
   * @param points     Samples per antenna
   * @param prof_events Optional: collect cl_events for profiling (null = production)
   *   - "Upload_delay" : clEnqueueWriteBuffer  (delay_us array)
   *   - "Kernel"       : lch_farrow_delay kernel
   * @return InputData<cl_mem> with delayed signal
   * @note Caller must release result.data via clReleaseMemObject()
   * @note input_buf is NOT released by this method
   */
  drv_gpu_lib::InputData<cl_mem> Process(
      cl_mem input_buf, uint32_t antennas, uint32_t points,
      ProfEvents* prof_events = nullptr);

  /**
   * @brief Apply fractional delay on CPU (reference)
   * @param input [antenna][sample] complex signal
   * @param antennas Number of antennas
   * @param points Samples per antenna
   * @return [antenna][sample] delayed signal
   */
  std::vector<std::vector<std::complex<float>>> ProcessCpu(
      const std::vector<std::vector<std::complex<float>>>& input,
      uint32_t antennas, uint32_t points);

  // Getters
  const std::vector<float>& GetDelays() const { return delay_us_; }
  float GetSampleRate() const { return sample_rate_; }

private:
  void CompileKernel();
  void UploadMatrix();
  void ReleaseGpuResources();

  drv_gpu_lib::IBackend* backend_ = nullptr;

  // Parameters
  std::vector<float> delay_us_;
  float sample_rate_ = 1e6f;
  float noise_amplitude_ = 0.0f;
  float norm_val_ = 0.7071067811865476f;
  uint32_t noise_seed_ = 0;

  // Lagrange matrix 48x5 (240 floats)
  std::vector<float> lagrange_matrix_;
  bool matrix_loaded_ = false;

  // OpenCL resources
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel kernel_ = nullptr;
  cl_mem matrix_buf_ = nullptr;

  // Persistent delay buffer (resized on demand)
  cl_mem delay_buf_ = nullptr;
  size_t delay_buf_size_ = 0;
};

} // namespace lch_farrow

#endif  // !ENABLE_ROCM
