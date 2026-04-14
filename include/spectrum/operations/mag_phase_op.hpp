#pragma once

/**
 * @file mag_phase_op.hpp
 * @brief MagPhaseOp — complex-to-magnitude/phase conversion kernel
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from FFTProcessorROCm::ExecuteMagPhaseKernel().
 *
 * Kernel: complex_to_mag_phase — converts FFT output to interleaved {mag, phase}.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14
 */

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

class MagPhaseOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MagPhase"; }

  /**
   * @brief Execute complex → mag/phase conversion
   * @param fft_output Device pointer to FFT result [total × complex<float>]
   * @param mag_phase_buf Device pointer to output [total × float2 {mag, phase}]
   * @param total_elements beam_count × nFFT
   */
  void Execute(void* fft_output, void* mag_phase_buf, size_t total_elements) {
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int block_size = 256;
    unsigned int grid_size = (total + block_size - 1) / block_size;

    void* args[] = { &fft_output, &mag_phase_buf, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("complex_to_mag_phase"),
        grid_size, 1, 1,
        block_size, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MagPhaseOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
