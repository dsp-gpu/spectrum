#pragma once

// ============================================================================
// SpectrumPadOp — zero-padding с beam_offset для batch-обработки (Layer 5 Ref03)
//
// ЧТО:    Concrete Op: zero-pad n_point → nFFT для SpectrumProcessor с
//         поддержкой beam_offset. Запускает kernel `pad_data` после
//         hipMemsetAsync(0). Отличается от fft_processor::PadDataOp
//         параметром beam_offset.
//
// ЗАЧЕМ:  При batch-обработке (radar pipeline разбивает 1024 antennas на
//         батчи по 256) input_buffer внутри batch'а начинается с beam 0,
//         но kernel'у нужен абсолютный beam-index для правильного access
//         pattern'а к shared memory / per-beam state. beam_offset решает
//         это без копирования или ремаппинга буферов.
//
// ПОЧЕМУ: - Отдельный Op (вместо параметра в PadDataOp) — namespace
//           разделение: antenna_fft (SpectrumProcessor) vs fft_processor
//           (FFTProcessor). Они сосуществуют, исторически разные модули.
//         - Layer 5 Ref03 (Concrete Op): один файл = один Op = один kernel.
//         - hipMemsetAsync(0) перед kernel — обязателен, «хвост» beyond
//           n_point должен быть строго zero (требование hipFFT).
//         - 2D grid: X — nFFT samples (не n_point!), Y — beam_id.
//           block_x=256 — стандарт RDNA4.
//         - Без window-параметра (в отличие от fft_processor::PadDataOp) —
//           SpectrumProcessor использует windowing в отдельном Op (легче
//           эволюционировать независимо).
//         - Stateless — Op переиспользуется для разных батчей.
//
// Использование:
//   antenna_fft::SpectrumPadOp pad_op;
//   pad_op.Initialize(ctx);
//   // Batch на beam_offset = 256:
//   pad_op.Execute(input_batch, fft_input_buf, batch_count, n_point, nFFT,
//                  /*beam_offset=*/256);
//
// История:
//   - Создан:  2026-03-22 (Ref03 Layer 5, выделено из SpectrumProcessorROCm)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace antenna_fft {

/**
 * @class SpectrumPadOp
 * @brief Layer 5 Ref03 Op: zero-pad n_point → nFFT с поддержкой beam_offset.
 *
 * @note Stateless (наследует ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @note beam_offset нужен для batch-processing с правильным абсолютным beam-индексом.
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see fft_processor::PadDataOp — аналогичный Op в namespace fft_processor (с window)
 */
class SpectrumPadOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SpectrumPad"; }

  /**
   * @brief Zero-padding: input → fft_input (с поддержкой beam_offset).
   * @param input_buf     Device pointer на raw input [beam_count × n_point × complex<float>].
   * @param fft_input_buf Device pointer на padded output [beam_count × nFFT × complex<float>].
   * @param beam_count    Число beams в текущем batch'е.
   * @param n_point       Samples на beam (до padding'а).
   * @param nFFT          Размер FFT (после padding'а).
   * @param beam_offset   Смещение beam-индекса в абсолютных координатах (для batch-processing).
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
