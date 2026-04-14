#pragma once

/**
 * @file i_all_maxima_pipeline.hpp
 * @brief Pipeline interface: Detect -> Scan -> Compact (FindAllMaxima)
 *
 * Separated from ISpectrumProcessor for Phase 3 refactoring.
 * Each backend implements: AllMaximaPipelineOpenCL, AllMaximaPipelineROCm.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <spectrum/interface/spectrum_maxima_types.h>

#include <cstdint>

namespace antenna_fft {

/**
 * @interface IAllMaximaPipeline
 * @brief Execute detect -> prefix sum -> compaction on magnitude data
 *
 * Input: GPU buffer with float magnitudes (beam_count * nFFT)
 * Output: AllMaximaResult with positions, magnitudes, frequencies
 */
class IAllMaximaPipeline {
public:
    virtual ~IAllMaximaPipeline() = default;

    /**
     * @brief Execute pipeline on pre-computed magnitudes
     * @param magnitudes_gpu GPU buffer with |FFT[i]| (float*, beam_count * nFFT elements)
     * @param fft_data_gpu GPU buffer with complex FFT (for frequency calculation)
     * @param max_maxima_per_beam Maximum number of maxima per beam (default 1000)
     */
    virtual AllMaximaResult Execute(
        void* magnitudes_gpu,
        void* fft_data_gpu,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 1,
        uint32_t search_end = 0,
        size_t max_maxima_per_beam = 1000) = 0;
};

} // namespace antenna_fft
