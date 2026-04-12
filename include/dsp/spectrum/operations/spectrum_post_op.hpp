#pragma once

/**
 * @file spectrum_post_op.hpp
 * @brief SpectrumPostOp — peak search kernel (ONE_PEAK / TWO_PEAKS)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from SpectrumProcessorROCm::ExecutePostKernel().
 *
 * Kernel: post_kernel_one_peak (4 MaxValue/beam) or post_kernel_two_peaks (8 MaxValue/beam)
 * Selection by PeakSearchMode at Execute() time (both compiled in GpuContext).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "types/spectrum_modes.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace antenna_fft {

class SpectrumPostOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SpectrumPost"; }

  /**
   * @brief Execute peak search kernel
   * @param fft_output     Device pointer to complex FFT data [beam_count × nFFT]
   * @param maxima_output  Device pointer to MaxValue results
   * @param beam_count     Number of beams
   * @param nFFT           FFT size
   * @param search_range   Range to search for peaks
   * @param sample_rate    Sample rate (Hz) for frequency calculation
   * @param peak_mode      ONE_PEAK or TWO_PEAKS
   */
  void Execute(void* fft_output, void* maxima_output,
               uint32_t beam_count, uint32_t nFFT,
               uint32_t search_range, float sample_rate,
               PeakSearchMode peak_mode) {
    unsigned int bc = beam_count;
    unsigned int nfft = nFFT;
    unsigned int sr_uint = search_range;

    void* args[] = {
      &fft_output, &maxima_output,
      &bc, &nfft, &sr_uint, &sample_rate
    };

    const char* pk_name = (peak_mode == PeakSearchMode::ONE_PEAK)
        ? "post_kernel_one_peak" : "post_kernel_two_peaks";

    hipError_t err = hipModuleLaunchKernel(
        kernel(pk_name),
        beam_count, 1, 1,
        kBlockSize, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("SpectrumPostOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
