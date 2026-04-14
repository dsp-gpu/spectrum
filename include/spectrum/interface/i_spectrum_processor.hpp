#pragma once

/**
 * @file i_spectrum_processor.hpp
 * @brief Strategy interface for spectrum processing (ROCm/hipFFT)
 *
 * Part of SpectrumMaximaFinder refactoring - Strategy + Bridge pattern.
 * Each backend (OpenCL, ROCm) implements this interface.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <spectrum/interface/spectrum_maxima_types.h>
#include <spectrum/interface/spectrum_input_data.hpp>

#include <vector>
#include <complex>
#include <memory>
#include <cstdint>

namespace antenna_fft {

/**
 * @interface ISpectrumProcessor
 * @brief Abstract interface for GPU spectrum processing (FFT + peak search)
 *
 * Implementations:
 * - SpectrumProcessorROCm (hipFFT, hipDevicePtr)
 */
class ISpectrumProcessor {
public:
    virtual ~ISpectrumProcessor() = default;

    /**
     * @brief Initialize GPU resources for given params
     * @param params Spectrum parameters (antenna_count, n_point, nFFT, etc.)
     */
    virtual void Initialize(const SpectrumParams& params) = 0;

    /**
     * @brief Check if processor is initialized
     */
    virtual bool IsInitialized() const = 0;

    /**
     * @brief Process CPU data - OnePeak or TwoPeaks mode
     * Pipeline: Upload -> FFT -> PostKernel -> ReadResults
     */
    virtual std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) = 0;

    /**
     * @brief Process GPU data - OnePeak or TwoPeaks mode
     * Pipeline: GPU copy -> FFT -> PostKernel -> ReadResults
     * @param gpu_data OpenCL: cl_mem, ROCm: hipDeviceptr_t (passed as void*)
     * @param gpu_memory_bytes Actual buffer size on GPU
     */
    virtual std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) = 0;

    /**
     * @brief Process single batch from CPU data (called by Facade during batch orchestration)
     */
    virtual std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) = 0;

    /**
     * @brief Process single batch from GPU buffer
     */
    virtual std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) = 0;

    /**
     * @brief Full pipeline: CPU data -> FFT -> FindAllMaxima
     */
    virtual AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief Full pipeline: GPU data -> FFT -> FindAllMaxima
     */
    virtual AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief AllMaxima from CPU FFT data (no FFT - data already transformed)
     */
    virtual AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief FindAllMaxima in ready FFT data on GPU (detect -> scan -> compact)
     * @param fft_data GPU buffer with complex FFT result
     */
    virtual AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) = 0;

    /**
     * @brief Get driver/backend type
     */
    virtual DriverType GetDriverType() const = 0;

    /**
     * @brief Get profiling data (aggregated from GPUProfiler for this module)
     */
    virtual ProfilingData GetProfilingData() const = 0;

    /**
     * @brief Reallocate buffers for batch processing
     */
    virtual void ReallocateBuffersForBatch(size_t batch_antenna_count) = 0;

    /**
     * @brief Calculate bytes per antenna (for BatchManager)
     */
    virtual size_t CalculateBytesPerAntenna() const = 0;

    /**
     * @brief Compile post-kernel (lazy init)
     */
    virtual void CompilePostKernel() = 0;
};

} // namespace antenna_fft
