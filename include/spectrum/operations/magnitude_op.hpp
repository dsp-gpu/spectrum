#pragma once

// ============================================================================
// MagnitudeOp — операция complex → |X| с нормировкой (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): преобразует complex<float> в
//         амплитуду с нормировкой inv_n. Два режима:
//           - squared=false → |X| * inv_n              (kernel complex_to_magnitude)
//           - squared=true  → (re² + im²) * inv_n     (kernel complex_to_magnitude_squared)
//         В отличие от ComputeMagnitudesOp (namespace antenna_fft) — этот в
//         namespace fft_processor и поддерживает inv_n + squared.
//
// ЗАЧЕМ:  Нужен внутри ComplexToMagPhaseROCm (фасад без FFT) и FFTProcessorROCm
//         (магнитуда после FFT с нормировкой 1/n_point). SNR-estimator (CFAR
//         на power spectrum) использует squared=true: нет sqrt в hot-path,
//         ratio |X_peak|² / mean(|X_ref|²) одинаково корректен.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один логический kernel (SRP), выбор
//           варианта (squared) — host-side по имени kernel'а; zero runtime
//           overhead, нет if-веток в device-коде.
//         - inv_n как параметр — позволяет: norm=1.0 (no norm), 1/n_point
//           (FFT scale), или custom (по запросу пользователя).
//         - squared=true даёт ~7× ускорение для SNR-задач: убран sqrt
//           (slow на RDNA, занимает ~25 cycles).
//         - Без BufferSet — Op stateless, как все Layer 5 Op'ы.
//
// Использование:
//   fft_processor::MagnitudeOp mag_op;
//   mag_op.Initialize(ctx);
//   // Обычная магнитуда с нормировкой 1/n:
//   mag_op.Execute(fft_out, mag_buf, total, 1.0f/n);
//   // Квадрат магнитуды для SNR (быстрее в ~7×):
//   mag_op.Execute(fft_out, pwr_buf, total, 1.0f/n, /*squared=*/true);
//
// История:
//   - Создан:  2026-03-22 (Ref03 Layer 5, выделено из ComplexToMagPhaseROCm)
//   - Изменён: 2026-04-09 (SNR_02: добавлен squared-параметр без sqrt)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

/**
 * @class MagnitudeOp
 * @brief Layer 5 Ref03 Op: complex → |X| (или |X|²) с нормировкой inv_n.
 *
 * @note Stateless (наследует ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @note squared=true (|X|²) — для SNR-estimator (CFAR), ~7× быстрее (без sqrt).
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see antenna_fft::ComputeMagnitudesOp — упрощённая версия без inv_n/squared
 */
class MagnitudeOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "Magnitude"; }

  /**
   * @brief Преобразовать complex → |X| с нормировкой (или |X|² при squared=true).
   * @param input          Device pointer на complex-данные [total × complex<float>].
   * @param output         Device pointer на float-выход [total × float].
   * @param total_elements beam_count × n_point — число элементов.
   * @param inv_n          Множитель нормировки (1.0 = без нормировки, 1/n_point — FFT scale, или custom).
   * @param squared        false (default) = |X|; true = |X|² (без sqrt, ~7× быстрее).
   *
   * Выбор kernel host-side по имени (zero runtime overhead):
   *   - squared=false → "complex_to_magnitude"
   *   - squared=true  → "complex_to_magnitude_squared"
   *
   * Для SNR-estimator (CFAR на power spectrum) — squared=true: ratio
   * |X_peak|² / mean(|X_ref|²) корректен, sqrt не нужен.
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
