#pragma once

/**
 * @file complex_to_mag_phase_rocm.hpp
 * @brief ComplexToMagPhaseROCm -- direct complex-to-magnitude+phase conversion on GPU
 *
 * Separate class for converting complex IQ data to magnitude+phase pairs.
 * Does NOT perform FFT -- pure mathematical conversion:
 *   magnitude = sqrt(re^2 + im^2)
 *   phase     = atan2(im, re)
 *
 * Supports:
 * - CPU data input (std::vector<complex<float>>)
 * - GPU data input (void* device pointer)
 * - CPU output (std::vector<MagPhaseResult>)
 * - GPU output (void* -- caller owns, must Free)
 * - Batch processing for large datasets (BatchManager)
 *
 * Key differences from FFTProcessorROCm:
 * - No hipFFT dependency (no FFT plan, no padding)
 * - Only 2 GPU buffers (input + output, no fft_input/fft_output)
 * - ProcessToGPU() methods for "stay on GPU" use case
 * - Lighter MagPhaseResult types (no nFFT, sample_rate, frequency)
 *
 * IMPORTANT: This file compiles ONLY with ENABLE_ROCM=1.
 * On Windows (no ROCm) this file is completely skipped.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "types/mag_phase_types.hpp"
#include "interface/i_backend.hpp"
#include "interface/gpu_context.hpp"
#include "services/buffer_set.hpp"
#include "services/batch_manager.hpp"
#include "operations/mag_phase_op.hpp"
#include "operations/magnitude_op.hpp"

#include <hip/hip_runtime.h>

#include <complex>
#include <vector>
#include <cstdint>
#include <string>

namespace fft_processor {

class ComplexToMagPhaseROCm {
public:
    // =========================================================================
    // Constructor / Destructor
    // =========================================================================

    /**
     * @brief Constructor
     * @param backend Pointer to IBackend (non-owning, must be ROCm backend)
     */
    explicit ComplexToMagPhaseROCm(drv_gpu_lib::IBackend* backend);

    ~ComplexToMagPhaseROCm();

    // No copying
    ComplexToMagPhaseROCm(const ComplexToMagPhaseROCm&) = delete;
    ComplexToMagPhaseROCm& operator=(const ComplexToMagPhaseROCm&) = delete;

    // Move semantics
    ComplexToMagPhaseROCm(ComplexToMagPhaseROCm&& other) noexcept;
    ComplexToMagPhaseROCm& operator=(ComplexToMagPhaseROCm&& other) noexcept;

    // =========================================================================
    // Public API -- CPU input -> CPU output
    // =========================================================================

    /**
     * @brief Convert complex data to magnitude+phase (CPU to CPU)
     * @param data Input: beam_count * n_point complex<float> values
     * @param params Conversion parameters (beam_count, n_point)
     * @return Vector of MagPhaseResult (one per beam)
     */
    std::vector<MagPhaseResult> Process(
        const std::vector<std::complex<float>>& data,
        const MagPhaseParams& params);

    // =========================================================================
    // Public API -- GPU input -> CPU output
    // =========================================================================

    /**
     * @brief Convert complex data to magnitude+phase (GPU to CPU)
     * @param gpu_data Device pointer to complex data (beam_count * n_point)
     * @param params Conversion parameters
     * @param gpu_memory_bytes Size of GPU buffer in bytes (0 = auto-calculate)
     * @return Vector of MagPhaseResult (one per beam)
     */
    std::vector<MagPhaseResult> Process(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

    // =========================================================================
    // Public API -- CPU input -> GPU output ("stay on GPU")
    // =========================================================================

    /**
     * @brief Convert and keep result on GPU
     * @param data Input: beam_count * n_point complex<float> values
     * @param params Conversion parameters
     * @return Device pointer to interleaved {mag, phase} pairs (float2_t layout)
     *         CALLER OWNS this pointer -- must call backend->Free() when done!
     *         Size = beam_count * n_point * 2 * sizeof(float)
     * @throws std::runtime_error if data does not fit in GPU memory
     */
    void* ProcessToGPU(
        const std::vector<std::complex<float>>& data,
        const MagPhaseParams& params);

    // =========================================================================
    // Public API -- GPU input -> GPU output (full GPU pipeline)
    // =========================================================================

    /**
     * @brief Convert GPU data, output stays on GPU (zero-copy path)
     * @param gpu_data Device pointer to complex data
     * @param params Conversion parameters
     * @param gpu_memory_bytes Size of GPU buffer in bytes (0 = auto-calculate)
     * @return Device pointer to interleaved {mag, phase} pairs (float2_t layout)
     *         CALLER OWNS this pointer -- must call backend->Free() when done!
     * @throws std::runtime_error if output buffer allocation fails
     */
    void* ProcessToGPU(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

    // =========================================================================
    // Public API -- Magnitude-only (no phase computed)
    // =========================================================================

    /**
     * @brief Convert GPU complex data to magnitude only (GPU in → CPU out)
     *
     * norm_coeff in params controls normalization:
     *   0.0f  → no normalization (inv_n = 1.0)
     *  -1.0f  → divide by n_point (inv_n = 1/n_point)
     *  >0.0f  → multiply by norm_coeff (inv_n = norm_coeff)
     *
     * @param gpu_data Device pointer to complex<float> data
     * @param params Conversion parameters (beam_count, n_point, norm_coeff)
     * @param gpu_memory_bytes Size of input GPU buffer in bytes (0 = auto)
     * @return Vector of MagnitudeResult (one per beam)
     */
    std::vector<MagnitudeResult> ProcessMagnitude(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

    /**
     * @brief Convert GPU complex data to magnitude only, result stays on GPU
     *
     * Zero-copy GPU→GPU path. Output buffer is plain float[beam_count * n_point].
     * CALLER OWNS the returned pointer — must hipFree when done.
     *
     * @param gpu_data Device pointer to complex<float> data
     * @param params Conversion parameters (beam_count, n_point, norm_coeff)
     * @param gpu_memory_bytes Size of input GPU buffer in bytes (0 = auto)
     * @return Device pointer to float[] magnitudes (beam_count * n_point floats)
     *         CALLER OWNS — must hipFree!
     */
    void* ProcessMagnitudeToGPU(
        void* gpu_data,
        const MagPhaseParams& params,
        size_t gpu_memory_bytes = 0);

    /**
     * @brief Convert GPU complex data to magnitude only, write to caller's buffer
     *
     * Zero allocations. Reads from gpu_complex_in, writes to gpu_magnitude_out.
     * Used by strategies to fill d_magnitudes_ directly after FFT.
     *
     * norm_coeff in params:
     *   0.0f  → no normalization (inv_n = 1.0)
     *  -1.0f  → divide by n_point
     *  >0.0f  → multiply by norm_coeff
     *
     * @param gpu_complex_in  Device pointer to complex<float> data [beam_count * n_point]
     * @param gpu_magnitude_out  Device pointer to float output [beam_count * n_point]
     * @param params  MagPhaseParams (beam_count, n_point, norm_coeff)
     */
    void ProcessMagnitudeToBuffer(void* gpu_complex_in, void* gpu_magnitude_out,
        const MagPhaseParams& params);

private:
    // =========================================================================
    // Internal methods
    // =========================================================================

    /// Ensure kernels compiled (lazy, one-time via GpuContext)
    void EnsureCompiled();

    /// Allocate/reuse internal GPU buffers
    void AllocateBuffers(size_t batch_beam_count);

    /// Upload CPU data to input buffer (H2D async)
    void UploadData(const std::complex<float>* data, size_t count);

    /// Copy GPU data to input buffer (D2D async)
    void CopyGpuData(void* src, size_t offset_bytes, size_t count);

    /// Read interleaved results from GPU and split into per-beam MagPhaseResult
    std::vector<MagPhaseResult> ReadResults(size_t beam_count, size_t start_beam);

    /// Calculate GPU memory required per beam (input + output)
    size_t CalculateBytesPerBeam() const;

    /// Compute inv_n from MagPhaseParams::norm_coeff
    static float ComputeInvN(const MagPhaseParams& params);

    // =========================================================================
    // Members — Ref03
    // =========================================================================

    drv_gpu_lib::GpuContext ctx_;  ///< Per-module context (compilation, stream, cache)

    // Internal GPU buffers (for CPU→CPU and GPU→CPU paths)
    enum Buf : size_t { kInput = 0, kOutput, kMagOnly, kBufCount };
    drv_gpu_lib::BufferSet<kBufCount> bufs_;

    // Op instances (Layer 5)
    MagPhaseOp mag_phase_op_;
    MagnitudeOp magnitude_op_;
    bool compiled_ = false;

    // State
    uint32_t n_point_ = 0;
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
