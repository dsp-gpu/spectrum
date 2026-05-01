#pragma once

// ============================================================================
// ComputeMagnitudesOp — операция complex → |z| (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): преобразует массив complex<float>
//         в массив амплитуд |z| = sqrt(re² + im²). Запускает HIP-kernel
//         `compute_magnitudes` в double-load схеме (один поток обрабатывает
//         2 элемента) для лучшего использования memory bandwidth.
//
// ЗАЧЕМ:  Используется внутри SpectrumProcessorROCm для двух путей:
//           1. Median path (fft_output → magnitudes → медианный фильтр).
//           2. AllMaxima pipeline (нужны амплитуды для Detect-стадии).
//         Выделено в отдельный Op из старого ExecuteComputeMagnitudes() для
//         повторного использования и чистого Layer 5 (один Op = один kernel).
//
// ПОЧЕМУ: - Layer 5 Ref03 (Concrete Op): один файл = один класс = один kernel
//           launch (SRP). Initialize/IsReady/Release делегируются базовому
//           GpuKernelOp; здесь только Execute.
//         - Double-load grid (kBlockSize × 2 элемента/thread) — общий приём
//           HIP optimization: меньше threads, больше работы на thread, лучше
//           memory coalescing на RDNA4.
//         - kBlockSize=256 — стандарт RDNA4 (warp×4, gfx1201).
//         - Без BufferSet — Op stateless, буферы передаются явно как параметры.
//         - Throws std::runtime_error при ошибке kernel launch (через GpuKernelOp).
//
// Использование:
//   ComputeMagnitudesOp mag_op;
//   mag_op.Initialize(ctx);  // GpuContext с скомпилированным kernel
//   mag_op.Execute(fft_output, magnitudes_buf, beam_count * nFFT);
//   // magnitudes_buf теперь содержит |fft_output[i]| для i = 0..total-1
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
 * @class ComputeMagnitudesOp
 * @brief Layer 5 Ref03 Op: complex → |z| magnitude через kernel compute_magnitudes (double-load).
 *
 * @note Stateless (наследует общий ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see fft_processor::MagnitudeOp — аналогичный Op в namespace fft_processor (с inv_n + squared)
 */
class ComputeMagnitudesOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "ComputeMagnitudes"; }

  /**
   * @brief Преобразовать complex → |z| (sqrt(re² + im²)) на GPU.
   * @param fft_output     Device pointer на complex FFT данные [total × complex<float>].
   * @param magnitudes     Device pointer на float-выход [total × float].
   * @param total_elements beam_count × nFFT — число элементов на обработку.
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
