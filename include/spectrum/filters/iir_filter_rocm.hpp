#pragma once

// ============================================================================
// IirFilterROCm — GPU IIR-фильтр каскадом biquad-секций (ROCm/HIP)
//
// ЧТО:    Применяет каскад biquad-секций (Second-Order Sections, SOS) к
//         комплексному multi-channel IQ сигналу. Каждая секция — рекурсивный
//         фильтр 2-го порядка с 5 коэффициентами {b0, b1, b2, a1, a2}.
//         SOS-матрица грузится один раз через SetBiquadSections() в persistent
//         GPU-буфер sos_buf_.
//
// ЗАЧЕМ:  Высокопорядковые IIR-фильтры (Butterworth, Chebyshev, Elliptic)
//         численно нестабильны как один полином — стандартная практика
//         разбивать на каскад biquad-ов. ROCm-порт нужен для интеграции в
//         hot-path радар-пайплайна, где CPU IIR на 1024 каналах = bottleneck.
//
// ПОЧЕМУ: - GPU IIR эффективен ТОЛЬКО на большом числе каналов (>= 8) —
//           внутри канала filter рекурсивен (sample[n] зависит от [n-1]),
//           параллелизм лежит ТОЛЬКО по каналам. На 1 канале CPU быстрее.
//         - hiprtc + GpuContext (Layer 1 Ref03) — kernel компилируется
//           один раз, lazy через EnsureCompiled().
//         - Persistent sos_buf_ — SOS-матрица заливается через
//           SetBiquadSections(), Process() не аллоцирует.
//         - kBlockSize=256 — стандарт RDNA4 (gfx1201).
//         - Move-only — sos_buf_ владеет hipMalloc-памятью.
//
// Использование:
//   filters::IirFilterROCm iir(rocm_backend);
//   iir.SetBiquadSections({{0.02f, 0.04f, 0.02f, -1.56f, 0.64f}});
//   auto result = iir.Process(gpu_input, channels, points);
//   // result.data — void* HIP device pointer, caller обязан hipFree
//
// История:
//   - Создан:  2026-02-23 (ROCm-порт OpenCL IirFilter)
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
 * @class IirFilterROCm
 * @brief GPU IIR-фильтр каскадом biquad-секций (multi-channel IQ).
 *
 * @note Move-only. Требует #if ENABLE_ROCM (CPU-only сборки получают stub).
 * @note Эффективен ТОЛЬКО при channels >= 8 — рекурсия внутри канала, параллелизм по каналам.
 * @note SOS-буфер persistent — повторные Process() не делают hipMalloc.
 * @note Не thread-safe (один экземпляр = один владелец GPU-ресурсов).
 * @see filters::FirFilterROCm — линейная фаза, неэффективен для high-Q
 * @ingroup grp_filters
 */
class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~IirFilterROCm();

  // No copy
  IirFilterROCm(const IirFilterROCm&) = delete;
  IirFilterROCm& operator=(const IirFilterROCm&) = delete;

  // Move
  IirFilterROCm(IirFilterROCm&& other) noexcept;
  IirFilterROCm& operator=(IirFilterROCm&& other) noexcept;

  // ════════════════════════════════════════════════════════════════════════
  // Configuration
  // ════════════════════════════════════════════════════════════════════════

  void LoadConfig(const std::string& json_path);
  void SetBiquadSections(const std::vector<BiquadSection>& sections);

  // ════════════════════════════════════════════════════════════════════════
  // Processing
  // ════════════════════════════════════════════════════════════════════════

  /**
   * @brief Применить IIR biquad-каскад на GPU (ROCm).
   * @param input_ptr   HIP device pointer на [channels × points × complex<float>].
   * @param channels    Число параллельных каналов (рекомендуется >= 8 для GPU-эффективности).
   * @param points      Samples на канал.
   * @param prof_events Опциональный сборщик ROCm-профайл-событий (null = production).
   * @return InputData<void*> с отфильтрованным сигналом (caller обязан hipFree result.data).
   * @note input_ptr НЕ освобождается этим методом.
   */
  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr, uint32_t channels, uint32_t points,
      ROCmProfEvents* prof_events = nullptr);

  /**
   * @brief Применить IIR-фильтр к CPU-данным (upload H2D + Process; результат остаётся на GPU).
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

  uint32_t GetNumSections() const { return static_cast<uint32_t>(sections_.size()); }
  const std::vector<BiquadSection>& GetSections() const { return sections_; }
  bool IsReady() const { return compiled_ && !sections_.empty(); }

private:
  void EnsureCompiled();
  void UploadSosMatrix();
  void ReleaseGpuResources();

  drv_gpu_lib::GpuContext ctx_;
  std::vector<BiquadSection> sections_;
  bool compiled_ = false;

  // GPU buffer for SOS matrix (persistent)
  void* sos_buf_ = nullptr;

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

class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend*) {}
  ~IirFilterROCm() = default;

  void LoadConfig(const std::string&) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  void SetBiquadSections(const std::vector<BiquadSection>&) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> Process(void*, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t, void* = nullptr) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>&, uint32_t, uint32_t) {
    throw std::runtime_error("IirFilterROCm: ROCm not enabled");
  }

  uint32_t GetNumSections() const { return 0; }
  const std::vector<BiquadSection>& GetSections() const {
    static std::vector<BiquadSection> empty;
    return empty;
  }
  bool IsReady() const { return false; }
};

}  // namespace filters

#endif  // ENABLE_ROCM
