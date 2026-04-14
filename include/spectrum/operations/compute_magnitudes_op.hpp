#pragma once

/**
 * @file compute_magnitudes_op.hpp
 * @brief ComputeMagnitudesOp — complex → |z| magnitude conversion kernel
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from SpectrumProcessorROCm::ExecuteComputeMagnitudes().
 *
 * Kernel: compute_magnitudes — |z| = sqrt(re²+im²) per element (double-load).
 * Used by SpectrumProcessorROCm for median path and AllMaxima pipeline.
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

class ComputeMagnitudesOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "ComputeMagnitudes"; }

  /**
   * @brief Execute complex → magnitude conversion
   * @param fft_output Device pointer to complex FFT data [total × complex<float>]
   * @param magnitudes Device pointer to float output [total × float]
   * @param total_elements beam_count × nFFT
   */
  void Execute(void* fft_output, void* magnitudes, size_t total_elements) {
    if (!ctx_) {
      throw std::runtime_error("ComputeMagnitudesOp: not initialized");
    }

    unsigned int total = static_cast<unsigned int>(total_elements);
    // Double-load grid: each thread handles 2 elements
    unsigned int grid = (total + kBlockSize * 2 - 1) / (kBlockSize * 2);

    void* args[] = { &fft_output, &magnitudes, &total };

    hipError_t err = hipModuleLaunchKernel(
        kernel("compute_magnitudes"),
        grid, 1, 1,
        kBlockSize, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("ComputeMagnitudesOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }

private:
  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
