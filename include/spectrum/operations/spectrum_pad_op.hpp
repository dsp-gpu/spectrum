#pragma once

/**
 * @file spectrum_pad_op.hpp
 * @brief SpectrumPadOp — zero-padding for SpectrumProcessor (with beam_offset)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from SpectrumProcessorROCm::ExecutePadKernel().
 *
 * Differs from fft_processor::PadDataOp by supporting beam_offset parameter
 * (used in batch processing where input_buffer starts at beam 0 but
 *  beam_offset indicates the absolute beam index for the kernel).
 *
 * Kernel: pad_data
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace antenna_fft {

class SpectrumPadOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SpectrumPad"; }

  /**
   * @brief Execute zero-padding: input → fft_input (with beam_offset)
   * @param input_buf     Device pointer to raw input [beam_count × n_point × complex]
   * @param fft_input_buf Device pointer to padded output [beam_count × nFFT × complex]
   * @param beam_count    Number of beams in this batch
   * @param n_point       Samples per beam (before padding)
   * @param nFFT          FFT size (after padding)
   * @param beam_offset   Offset for beam indexing (batch processing)
   */
  void Execute(void* input_buf, void* fft_input_buf,
               size_t beam_count, uint32_t n_point, uint32_t nFFT,
               uint32_t beam_offset = 0) {
    // Zero entire padded buffer first (async)
    hipMemsetAsync(fft_input_buf, 0,
        beam_count * nFFT * sizeof(float) * 2,
        stream());

    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int np = n_point;
    unsigned int nf = nFFT;
    unsigned int bo = beam_offset;

    unsigned int grid_x = (nFFT + 255) / 256;
    unsigned int grid_y = bc;

    void* args[] = { &input_buf, &fft_input_buf, &bc, &np, &nf, &bo };

    hipError_t err = hipModuleLaunchKernel(
        kernel("pad_data"),
        grid_x, grid_y, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("SpectrumPadOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
