#pragma once

// ============================================================================
// FirFilterROCm — GPU FIR-фильтр свёрткой во временной области (ROCm/HIP)
//
// ЧТО:    Применяет FIR-фильтр (Finite Impulse Response) к комплексному
//         multi-channel IQ сигналу [channels × points × complex<float>].
//         Коэффициенты загружаются один раз через SetCoefficients(), persistent
//         GPU-буфер переиспользуется между вызовами Process().
//
// ЗАЧЕМ:  В ЦОС-пайплайне фильтрация — обязательная стадия (anti-aliasing,
//         decimation, формирование полосы). FIR гарантирует линейную фазу
//         (важно для радара), в отличие от IIR. На GPU — параллелизм по
//         каналам и samples, на больших batch'ах быстрее CPU в 10-50×.
//
// ПОЧЕМУ: - hiprtc + GpuContext (Layer 1 Ref03) — kernel компилируется один
//           раз, lazy через EnsureCompiled(), кешируется на диск.
//         - Persistent coeff_buf_ — коэффициенты загружаются однократно при
//           SetCoefficients(); Process() использует уже-выделенный буфер,
//           не делает hipMalloc/hipFree в hot-path.
//         - kBlockSize=256 — стандартный warp×4 для RDNA4 (gfx1201): меньше
//           idle threads, больше register не spill'ит.
//         - Move-only — coeff_buf_ владеет hipMalloc-памятью, копирование
//           привело бы к double-free в деструкторе.
//         - prof_events optional — null = production (zero overhead);
//           non-null = benchmark, заполняется парами (stage_name, ROCmProfilingData)
//           через MakeROCmDataFromEvents (см. utils/rocm_profiling_helpers.hpp).
//
// Использование:
//   filters::FirFilterROCm fir(rocm_backend);
//   fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});
//   auto result = fir.Process(gpu_input, channels, points);
//   // result.data — void* (HIP device pointer), caller обязан hipFree
//
// История:
//   - Создан:  2026-02-23 (ROCm-порт OpenCL FirFilter, namespace filters)
//   - Изменён: 2026-04-15 (миграция в DSP-GPU/spectrum, GpuContext Ref03)
// ============================================================================

#if ENABLE_ROCM

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/types/filter_params.hpp>
#include <core/services/profiling_types.hpp>

#include <hip/hip_runtime.h>

#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

/// Список событий профилирования ROCm (имя стадии + ROCmProfilingData)
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class FirFilterROCm
 * @brief GPU FIR-фильтр через свёртку во временной области (multi-channel IQ).
 *
 * @note Move-only. Требует #if ENABLE_ROCM (CPU-only сборки получают stub).
 * @note Coeff-buffer persistent — повторные Process() не делают hipMalloc.
 * @note Не thread-safe (один экземпляр = один владелец GPU-ресурсов).
 * @see filters::IirFilterROCm — IIR-альтернатива (рекурсивный фильтр)
 * @see filters::MovingAverageFilterROCm — упрощённый частный случай
 * @ingroup grp_filters
 */
class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~FirFilterROCm();

  // No copy
  FirFilterROCm(const FirFilterROCm&) = delete;
  FirFilterROCm& operator=(const FirFilterROCm&) = delete;

  // Move
  FirFilterROCm(FirFilterROCm&& other) noexcept;
  FirFilterROCm& operator=(FirFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void LoadConfig(const std::string& json_path);
  void SetCoefficients(const std::vector<float>& coeffs);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Применить FIR-фильтр на GPU (ROCm).
   * @param input_ptr   HIP device pointer на [channels × points × complex<float>].
   * @param channels    Число параллельных каналов.
   * @param points      Samples на канал.
   * @param prof_events Опциональный сборщик ROCm-профайл-событий (null = production).
   * @return InputData<void*> с отфильтрованным сигналом (caller обязан hipFree result.data).
   * @note input_ptr НЕ освобождается этим методом.
   */
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief Применить FIR-фильтр к CPU-данным (upload H2D + Process; результат остаётся на GPU).
   */
  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief CPU-эталон (для валидации GPU-результатов в тестах).
   */
  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels, uint32_t points);

  // ════════════════════════════════════════════════════════════════════════
  // Getters
  // ════════════════════════════════════════════════════════════════════════

  uint32_t GetNumTaps() const { return static_cast<uint32_t>(coefficients_.size()); }
  const std::vector<float>& GetCoefficients() const { return coefficients_; }
  bool IsReady() const { return compiled_ && !coefficients_.empty(); }

private:
  void EnsureCompiled();
  void UploadCoefficients();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  std::vector<float> coefficients_;
  bool compiled_ = false;

  // GPU buffer for coefficients (persistent)
  void* coeff_buf_ = nullptr;

  static constexpr unsigned int kBlockSize = 256;
};

}  // namespace filters

#else  // !ENABLE_ROCM

// ═══════════════════════════════════════════════════════════════════════════
// Stub for non-ROCm builds (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <spectrum/types/filter_params.hpp>

#include <stdexcept>
#include <vector>
#include <complex>
#include <string>
#include <cstdint>

namespace filters {

class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend*) {}
  ~FirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  void SetCoefficients(const std::vector<float>&) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("FirFilterROCm: ROCm not enabled");
  }

  uint32_t GetNumTaps() const { return 0; }
  const std::vector<float>& GetCoefficients() const {
    static std::vector<float> empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
