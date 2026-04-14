#pragma once

#if !ENABLE_ROCM  // OpenCL-only: ROCm uses LchFarrowROCm

/**
 * @file lch_farrow.hpp
 * @brief LchFarrow - Lagrange interpolation fractional delay processor (48x5)
 *
 * Standalone module for applying fractional delay to complex signals on GPU.
 * Uses Lagrange 5-point interpolation with pre-computed 48x5 coefficient matrix.
 *
 * Algorithm:
 *   For each output sample n:
 *     read_pos = n - delay_samples
 *     if read_pos < 0: output[n] = 0
 *     else: output[n] = 5-point Lagrange interpolation of input at read_pos
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

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
 * @brief GPU fractional delay processor (Lagrange 48x5)
 *
 * Applies per-antenna fractional delay to an already-generated complex signal.
 * Independent from signal_generators module.
 *
 * @code
 * LchFarrow processor(backend);
 * processor.SetDelays({0.0f, 1.5f, 3.0f});
 * processor.SetSampleRate(1e6f);
 *
 * // From existing GPU buffer
 * auto result = processor.Process(input_buf, antennas, points);
 * // result.data is cl_mem (delayed signal)
 *
 * // Or from CPU data
 * auto cpu_result = processor.ProcessCpu(input_data, antennas, points);
 * @endcode
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
