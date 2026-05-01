#pragma once

// ============================================================================
// MagPhaseOp — операция complex → {magnitude, phase} (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): преобразует complex<float> в
//         interleaved {mag, phase} float2 формат — magnitude = sqrt(re²+im²),
//         phase = atan2(im, re). Запускает kernel `complex_to_mag_phase`.
//
// ЗАЧЕМ:  Используется внутри FFTProcessorROCm (когда нужны и амплитуда и
//         фаза после FFT) и ComplexToMagPhaseROCm (без FFT, прямая конверсия
//         IQ → mag+phase). atan2 на GPU дороже чем sqrt, поэтому отдельный
//         Op нужен только когда фаза реально нужна (interferometry, MIMO
//         AoA estimation).
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один kernel = одна Execute() (SRP).
//         - Interleaved {mag, phase} как float2 — оптимально для последующего
//           D2H copy: один hipMemcpy вместо двух separate buffers.
//         - block_size = 256 — стандарт RDNA4 (gfx1201).
//         - Stateless (без BufferSet) — Op переиспользуется для разных
//           буферов; вызывающий передаёт device pointers явно.
//         - Throws std::runtime_error при ошибке kernel launch.
//
// Использование:
//   MagPhaseOp mp_op;
//   mp_op.Initialize(ctx);
//   mp_op.Execute(fft_output, mag_phase_buf, beam_count * nFFT);
//   // mag_phase_buf теперь содержит float2{mag, phase} интерливед
//
// История:
//   - Создан:  2026-03-14 (Ref03 Layer 5, выделено из FFTProcessorROCm)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace fft_processor {

/**
 * @class MagPhaseOp
 * @brief Layer 5 Ref03 Op: complex → {magnitude, phase} interleaved float2.
 *
 * @note Stateless (наследует ctx_/kernel из GpuKernelOp).
 * @note Требует #if ENABLE_ROCM.
 * @note atan2 на GPU дорогой — использовать только когда фаза реально нужна.
 * @see drv_gpu_lib::GpuKernelOp — базовый Layer 3 Ref03
 * @see fft_processor::MagnitudeOp — только |X| без phase
 */
class MagPhaseOp : public drv_gpu_lib::GpuKernelOp {
public:
  const char* Name() const override { return "MagPhase"; }

  /**
   * @brief Преобразовать complex → {mag, phase} (interleaved float2) на GPU.
   * @param fft_output     Device pointer на FFT-результат [total × complex<float>].
   * @param mag_phase_buf  Device pointer на выход [total × float2 {mag, phase}].
   * @param total_elements beam_count × nFFT — число элементов.
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
