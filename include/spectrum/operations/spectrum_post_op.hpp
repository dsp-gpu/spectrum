#pragma once

// ============================================================================
// SpectrumPostOp — поиск пика спектра ONE_PEAK / TWO_PEAKS (Layer 5 Ref03)
//
// ЧТО:    Concrete Op: пост-обработка FFT-результата — поиск top-K пиков по
//         амплитуде в каждом beam'е. Два режима:
//           - ONE_PEAK   — kernel post_kernel_one_peak  (4 MaxValue/beam)
//           - TWO_PEAKS  — kernel post_kernel_two_peaks (8 MaxValue/beam)
//         Оба kernel'а компилируются в GpuContext одновременно, выбор по
//         PeakSearchMode в Execute() — host-side (zero runtime overhead).
//
// ЗАЧЕМ:  В радаре часто нужен не полный AllMaxima, а top-1 или top-2 пика
//         per beam (basic target detection). Эти режимы быстрее AllMaxima
//         в ~5-10× за счёт компактного output (4-8 MaxValue вместо 1000)
//         и отсутствия prefix-sum stage.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один kernel-launch (выбор режима — выбор
//           kernel-имени, не runtime-проверки внутри kernel).
//         - 4 / 8 MaxValue per beam — выровнено под warp на RDNA4 (warp=64,
//           компактный access). Под N>2 пиков подходит AllMaxima pipeline.
//         - kBlockSize=256, grid = beam_count (1 block per beam) — каждый
//           beam обрабатывается независимо, классическая reduce-pattern.
//         - search_range — фильтрация bin'ов (типично игнорируем DC и
//           зеркальную часть для real-input).
//         - sample_rate — для пересчёта bin → frequency внутри kernel.
//         - Throws std::runtime_error при ошибке kernel launch.
//
// Использование:
//   antenna_fft::SpectrumPostOp post_op;
//   post_op.Initialize(ctx);
//   post_op.Execute(fft_output, maxima_buf,
//                   beam_count, nFFT, /*search_range=*/nFFT/2,
//                   sample_rate, PeakSearchMode::ONE_PEAK);
//
// История:
//   - Создан:  2026-03-22 (Ref03 Layer 5, выделено из SpectrumProcessorROCm)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/types/spectrum_modes.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace antenna_fft {

/**
 * @class SpectrumPostOp
 * @brief Layer 5 Ref03 Op: поиск top-1 / top-2 пиков спектра per beam.
 *
 * @note Stateless (наследует ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @note Для top-N с N > 2 — использовать AllMaximaPipelineROCm.
 * @note Оба kernel'а скомпилированы в GpuContext, выбор host-side.
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see IAllMaximaPipeline — для top-N с N > 2
 */
class SpectrumPostOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "SpectrumPost"; }

  /**
   * @brief Запустить kernel поиска пика (ONE_PEAK или TWO_PEAKS).
   * @param fft_output    Device pointer на complex FFT-данные [beam_count × nFFT].
   * @param maxima_output Device pointer на массив MaxValue (4/beam для ONE_PEAK, 8/beam для TWO_PEAKS).
   * @param beam_count    Число beams.
   * @param nFFT          Размер FFT.
   * @param search_range  Диапазон поиска (bin'ы) — фильтрует DC и mirror.
   * @param sample_rate   Sample rate (Hz) для пересчёта bin → frequency.
   * @param peak_mode     ONE_PEAK или TWO_PEAKS.
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
