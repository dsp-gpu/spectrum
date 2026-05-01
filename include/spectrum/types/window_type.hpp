#pragma once

/**
 * @file window_type.hpp
 * @brief Типы оконных функций для FFT pre-processing (Hann, Hamming, Blackman).
 *
 * @note Тип B (technical header): только enum.
 *       Применяется в PadDataOp ДО zero-padding для подавления sinc sidelobes.
 *       Default None = rectangular (legacy path, без изменений поведения).
 * @note Калибровка SNR-estimator (PyPanelAntennas/SNR/REPORT.md, 2026-04-09):
 *         - None     → −27 dB bias (sinc sidelobes −13 dB) ❌
 *         - Hann     → sidelobes −32 dB
 *         - Hamming  → sidelobes −43 dB (выше processing loss)
 *         - Blackman → sidelobes −58 dB (макс. loss)
 *
 * История:
 *   - Создан:  2026-04-09
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <cstdint>

namespace fft_processor {

/// Window function type (для FFTProcessorROCm::ProcessMagnitudesToGPU, PadDataOp).
enum class WindowType : uint32_t {
  None     = 0,  ///< rectangular (default, без обработки, legacy path)
  Hann     = 1,  ///< Hann: w[n] = 0.5*(1 - cos(2π·n/(N-1))), sidelobe -32 dB
  Hamming  = 2,  ///< Hamming: w[n] = 0.54 - 0.46*cos(2π·n/(N-1)), sidelobe -43 dB
  Blackman = 3,  ///< Blackman: three-cos, sidelobe -58 dB
};

}  // namespace fft_processor
