#pragma once

// ============================================================================
// PadDataOp — zero-padding входа FFT с опциональным window-function (Layer 5 Ref03)
//
// ЧТО:    Concrete Op: копирует n_point samples per beam в буфер размера nFFT
//         (степень двойки, требование hipFFT), остальное забивает нулями.
//         Опционально применяет inline window function (Hann/Hamming/Blackman)
//         в одном проходе — экономит second pass через memory.
//         Pipeline: hipMemsetAsync(fft_input, 0) + kernel pad_data[_windowed].
//
// ЗАЧЕМ:  hipFFT требует размер power-of-2; реальные радар-данные
//         (n_point=например 1100) нужно дополнить нулями до nFFT=2048.
//         Без windowing (rectangular) появляются sinc-sidelobes — для SNR
//         estimator это даёт смещение −27 dB, что критично. Inline window
//         в pad-kernel исправляет это без второго прохода.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один kernel-launch (плюс memset). Выбор
//           между pad_data и pad_data_windowed — host-side по WindowType.
//         - Default WindowType::None → старый kernel pad_data, legacy API
//           не ломается (важно для существующих тестов).
//         - hipMemsetAsync ВСЕГДА перед pad-kernel — если n_point < nFFT,
//           «хвост» должен быть строго zero. Делать через kernel дольше, чем
//           через DMA-style memset.
//         - 2D grid: X — n_point samples, Y — beam_id. block_x=256.
//         - Без BufferSet — Op stateless.
//         - Throws std::runtime_error если memset или kernel launch упал.
//
// Использование:
//   fft_processor::PadDataOp pad_op;
//   pad_op.Initialize(ctx);
//   // Старый API (rectangular window):
//   pad_op.Execute(input_buf, fft_input_buf, beam_count, n_point, nFFT);
//   // С Hann window для SNR-estimator:
//   pad_op.Execute(input_buf, fft_input_buf, beam_count, n_point, nFFT,
//                  WindowType::Hann);
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, выделено из FFTProcessorROCm)
//   - Изменён: 2026-04-09 (SNR_02b: добавлен window-параметр для pad_data_windowed)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/types/window_type.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

/**
 * @class PadDataOp
 * @brief Layer 5 Ref03 Op: zero-pad n_point → nFFT (+ опциональный window function).
 *
 * @note Stateless (наследует ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @note Default window=None = rectangular (legacy behavior).
 * @note Hann/Hamming/Blackman — нужны для SNR-estimator (без них sinc-bias −27 dB).
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see antenna_fft::SpectrumPadOp — аналогичный Op в namespace antenna_fft (с beam_offset)
 */
class PadDataOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "PadData"; }

  /**
   * @brief Zero-padding (+ optional window): input_buf → fft_input_buf.
   * @param input_buf     Device pointer на raw input [beam_count × n_point × complex<float>].
   * @param fft_input_buf Device pointer на padded output [beam_count × nFFT × complex<float>].
   * @param beam_count    Число beams в текущем batch'е.
   * @param n_point       Samples на beam (до padding'а).
   * @param nFFT          Размер FFT (после padding'а, степень двойки).
   * @param window        Window function (default None = rectangular, legacy).
   *
   * Default WindowType::None → старый kernel `pad_data` (legacy API не ломается).
   * Для SNR-estimator — WindowType::Hann: убирает sinc sidelobes (−27 dB bias!).
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
