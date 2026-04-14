#pragma once

/**
 * @file fft_processor_rocm.hpp
 * @brief FFTProcessorROCm — thin Facade for FFT using hipFFT + hiprtc kernels
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 *
 * ROCm port of FFTProcessor. Same public API, refactored internals:
 * - GpuContext for kernel compilation (pad_data + complex_to_mag_phase)
 * - BufferSet<4> for pipeline GPU buffers
 * - PadDataOp + MagPhaseOp for kernel launches
 * - hipFFT plan management with LRU-2 cache (stays in facade)
 *
 * PUBLIC API IS UNCHANGED — Python bindings work as before.
 *
 * IMPORTANT: This file compiles ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-03-14 (v2 Ref03 Facade)
 */

#if ENABLE_ROCM

#include <spectrum/fft_processor_types.hpp>
#include <spectrum/types/window_type.hpp>
#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/buffer_set.hpp>
#include <core/services/batch_manager.hpp>
#include <core/services/gpu_profiler.hpp>

// Op classes (Layer 5)
#include <spectrum/operations/pad_data_op.hpp>
#include <spectrum/operations/mag_phase_op.hpp>
#include <spectrum/operations/magnitude_op.hpp>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <complex>
#include <vector>
#include <cstdint>
#include <string>
#include <chrono>

namespace fft_processor {

/// ROCm profiling events: (name, ROCmProfilingData) pairs collected during processing
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/// @ingroup grp_fft_func
class FFTProcessorROCm {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  explicit FFTProcessorROCm(drv_gpu_lib::IBackend* backend);
  ~FFTProcessorROCm();

  // No copying
  FFTProcessorROCm(const FFTProcessorROCm&) = delete;
  FFTProcessorROCm& operator=(const FFTProcessorROCm&) = delete;

  // Move semantics
  FFTProcessorROCm(FFTProcessorROCm&& other) noexcept;
  FFTProcessorROCm& operator=(FFTProcessorROCm&& other) noexcept;

  // =========================================================================
  // Public API -- Complex output
  // =========================================================================

  std::vector<FFTComplexResult> ProcessComplex(
      const std::vector<std::complex<float>>& data,
      const FFTProcessorParams& params,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<FFTComplexResult> ProcessComplex(
      void* gpu_data,
      const FFTProcessorParams& params,
      size_t gpu_memory_bytes = 0);

  // =========================================================================
  // Public API -- Magnitude + Phase output
  // =========================================================================

  std::vector<FFTMagPhaseResult> ProcessMagPhase(
      const std::vector<std::complex<float>>& data,
      const FFTProcessorParams& params,
      ROCmProfEvents* prof_events = nullptr);

  std::vector<FFTMagPhaseResult> ProcessMagPhase(
      void* gpu_data,
      const FFTProcessorParams& params,
      size_t gpu_memory_bytes = 0);

  // =========================================================================
  // Public API -- Magnitudes directly to caller-provided GPU buffer (SNR_04)
  // =========================================================================

  /**
   * @brief Process FFT and write magnitudes directly to caller GPU buffer.
   *
   * Pipeline: PadDataOp(window) → hipfftExecC2C → MagnitudeOp(squared).
   * NO D2H copy — output goes straight to caller-provided GPU buffer.
   *
   * @param gpu_data            GPU input [beam_count × n_point × complex<float>]
   * @param gpu_out_magnitudes  GPU output [beam_count × nFFT × float] (caller owns)
   * @param params              FFT params (beam_count, n_point, sample_rate, ...)
   * @param squared             false = |X| (default), true = |X|² (square-law, no sqrt)
   * @param window              WindowType::None (default) / Hann / Hamming / Blackman
   *                            Для SNR-estimator использовать Hann (решает sinc sidelobes).
   * @param prof_events         Optional profiling events collector
   *
   * Используется SnrEstimatorOp — нужны магнитуды сразу на GPU, без D2H roundtrip.
   */
  void ProcessMagnitudesToGPU(
      void* gpu_data,
      void* gpu_out_magnitudes,
      const FFTProcessorParams& params,
      bool squared = false,
      WindowType window = WindowType::None,
      ROCmProfEvents* prof_events = nullptr);

  // =========================================================================
  // Information
  // =========================================================================

  FFTProfilingData GetProfilingData() const;
  uint32_t GetNFFT() const { return nFFT_; }

private:
  // =========================================================================
  // Utilities
  // =========================================================================

  static uint32_t NextPowerOf2(uint32_t n);
  void CalculateNFFT(const FFTProcessorParams& params);
  size_t CalculateBytesPerBeam(FFTOutputMode mode) const;

  // =========================================================================
  // GPU Resources
  // =========================================================================

  /// Ensure kernels compiled (lazy, one-time)
  void EnsureCompiled();

  /// Allocate/reuse pipeline buffers
  void AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode);

  /// Create/reuse hipFFT plan (LRU-2 cache)
  void CreateFFTPlan(size_t batch_beam_count);

  /// Release hipFFT plans
  void ReleasePlans();

  // =========================================================================
  // Data transfer
  // =========================================================================

  void UploadData(const std::complex<float>* data, size_t count);
  void CopyGpuData(void* src, size_t src_offset_bytes, size_t count);

  // =========================================================================
  // Read results
  // =========================================================================

  std::vector<FFTComplexResult> ReadComplexResults(
      size_t beam_count, size_t start_beam, float sample_rate);

  std::vector<FFTMagPhaseResult> ReadMagPhaseResults(
      size_t beam_count, size_t start_beam,
      float sample_rate, bool include_freq);

  // =========================================================================
  // Members — Ref03
  // =========================================================================

  drv_gpu_lib::GpuContext ctx_;  ///< Per-module context (kernels, stream)

  // Pipeline GPU buffers
  // In-place FFT: kFftBuf serves as both padded input AND FFT output
  // (hipfftExecC2C supports idata == odata), saving one buffer allocation.
  enum PipelineBuf : size_t {
    kInputBuf = 0,        ///< Raw input: batch × n_point × complex<float>
    kFftBuf,              ///< Zero-padded + in-place FFT result: batch × nFFT × complex<float>
    kMagPhaseInterleaved, ///< Interleaved {mag, phase}: batch × nFFT × float2
    kBufCount
  };
  drv_gpu_lib::BufferSet<kBufCount> bufs_;

  // Op instances (Layer 5)
  PadDataOp pad_op_;
  MagPhaseOp mag_phase_op_;
  MagnitudeOp mag_op_;  ///< SNR_04: magnitude-only kernel for ProcessMagnitudesToGPU

  bool compiled_ = false;

  // hipFFT plans — LRU-2 cache (stays in facade, not in Op)
  hipfftHandle plan_ = 0;
  bool plan_created_ = false;
  hipfftHandle plan_last_ = 0;
  size_t plan_last_batch_ = 0;

  // FFT parameters
  uint32_t nFFT_ = 0;
  uint32_t n_point_ = 0;
  size_t plan_batch_size_ = 0;
  bool has_mag_phase_buffers_ = false;
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
