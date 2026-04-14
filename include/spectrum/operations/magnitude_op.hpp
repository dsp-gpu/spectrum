#pragma once

/**
 * @file magnitude_op.hpp
 * @brief MagnitudeOp — complex-to-magnitude conversion kernel (no phase)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from ComplexToMagPhaseROCm::ExecuteMagnitudeKernel().
 *
 * Kernels:
 *   - complex_to_magnitude          → |X| * inv_n            (default)
 *   - complex_to_magnitude_squared  → (re²+im²) * inv_n (SNR_02, ~7× faster, no sqrt)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22 (v1), 2026-04-09 (v2 — SNR_02 squared param)
 */

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

class MagnitudeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Magnitude"; }

  /**
   * @brief Execute complex → magnitude conversion
   * @param input          Device pointer to complex data [total × complex<float>]
   * @param output         Device pointer to float output [total × float]
   * @param total_elements beam_count × n_point
   * @param inv_n          Normalization factor (1.0 = no norm, 1/n_point, или custom)
   * @param squared        false = |X| (default, legacy), true = |X|² (no sqrt, ~7× faster)
   *
   * Выбор kernel делается на host-side по имени (zero runtime overhead):
   *   - squared=false → "complex_to_magnitude"
   *   - squared=true  → "complex_to_magnitude_squared"
   *
   * Для SNR-estimator (CFAR на power spectrum) передаётся squared=true —
   * не нужен sqrt, т.к. ratio |X_peak|²/mean(|X_ref|²) одинаково работает.
   */
  void Execute(void* input, void* output,
               size_t total_elements, float inv_n,
               bool squared = false) {
    unsigned int total = static_cast<unsigned int>(total_elements);
    unsigned int block_size = 256;
    unsigned int grid_size = (total + block_size - 1) / block_size;

    void* args[] = { &input, &output, &inv_n, &total };

    const char* kernel_name = squared
        ? "complex_to_magnitude_squared"
        : "complex_to_magnitude";

    hipError_t err = hipModuleLaunchKernel(
        kernel(kernel_name),
        grid_size, 1, 1,
        block_size, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("MagnitudeOp: " +
                                std::string(hipGetErrorString(err)));
    }
  }
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
