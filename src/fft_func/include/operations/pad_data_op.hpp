#pragma once

/**
 * @file pad_data_op.hpp
 * @brief PadDataOp — zero-padding kernel for FFT input (+ optional window function)
 *
 * Ref03 Layer 5: Concrete Operation.
 * Extracted from FFTProcessorROCm::ExecutePadKernel().
 *
 * Pipeline: memset(fft_input, 0) + (pad_data OR pad_data_windowed) kernel
 *
 * Kernels:
 *   - pad_data           — copies n_point samples per beam, rest already zeroed.
 *   - pad_data_windowed  — SNR_02b: pad + inline Hann/Hamming/Blackman window.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-14 (v1), 2026-04-09 (v2 — SNR_02b window param)
 */

#if ENABLE_ROCM

#include "services/gpu_kernel_op.hpp"
#include "interface/gpu_context.hpp"
#include "types/window_type.hpp"

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

class PadDataOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "PadData"; }

  /**
   * @brief Execute zero-padding (+ optional window): input_buf → fft_input_buf
   * @param input_buf     Device pointer to raw input [beam_count × n_point × complex]
   * @param fft_input_buf Device pointer to padded output [beam_count × nFFT × complex]
   * @param beam_count    Number of beams in this batch
   * @param n_point       Samples per beam (before padding)
   * @param nFFT          FFT size (after padding, power of 2)
   * @param window        Window function (default None = rectangular, legacy behavior)
   *
   * Default WindowType::None → вызывает старый kernel `pad_data` (идентично прежнему API).
   * Для SNR-estimator передаётся WindowType::Hann — решает sinc sidelobes (−27 dB bias!).
   */
  void Execute(void* input_buf, void* fft_input_buf,
               size_t beam_count, uint32_t n_point, uint32_t nFFT,
               WindowType window = WindowType::None) {
    // Zero entire padded buffer first (async) — одинаково для обоих kernel
    hipError_t err = hipMemsetAsync(
        fft_input_buf, 0,
        beam_count * nFFT * sizeof(float) * 2,  // complex<float> = 2 floats
        stream());
    if (err != hipSuccess) {
      throw std::runtime_error("PadDataOp memset: " +
                                std::string(hipGetErrorString(err)));
    }

    // 2D grid: X covers n_point samples, Y = beam_id
    unsigned int block_x = 256;
    unsigned int grid_x = static_cast<unsigned int>((n_point + block_x - 1) / block_x);
    unsigned int grid_y = static_cast<unsigned int>(beam_count);

    unsigned int np = n_point;
    unsigned int nf = nFFT;

    if (window == WindowType::None) {
      // Legacy path: старое API, существующие вызовы не меняются
      void* args[] = { &input_buf, &fft_input_buf, &np, &nf };

      err = hipModuleLaunchKernel(
          kernel("pad_data"),
          grid_x, grid_y, 1,
          block_x, 1, 1,
          0, stream(),
          args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("PadDataOp pad_data: " +
                                  std::string(hipGetErrorString(err)));
      }
    } else {
      // Windowed path: новый kernel с window function inline
      int w_type = static_cast<int>(window);
      void* args[] = { &input_buf, &fft_input_buf, &np, &nf, &w_type };

      err = hipModuleLaunchKernel(
          kernel("pad_data_windowed"),
          grid_x, grid_y, 1,
          block_x, 1, 1,
          0, stream(),
          args, nullptr);
      if (err != hipSuccess) {
        throw std::runtime_error("PadDataOp pad_data_windowed: " +
                                  std::string(hipGetErrorString(err)));
      }
    }
  }
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
