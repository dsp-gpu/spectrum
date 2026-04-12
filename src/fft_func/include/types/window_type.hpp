#pragma once

/**
 * @file window_type.hpp
 * @brief Window function types для FFT pre-processing (Hann, Hamming, Blackman).
 *
 * Применяется в PadDataOp перед zero-padding для подавления sinc sidelobes
 * спектра конечного сигнала. Default WindowType::None = rectangular
 * (существующее поведение).
 *
 * Калибровано Python моделью SNR-estimator (2026-04-09):
 *   - rect (None)  даёт катастрофический −27 dB bias (sinc sidelobes −13 dB)
 *   - Hann         решает проблему (sidelobes −32 dB)
 *   - Hamming      сильнее (−43 dB), но больше processing loss
 *   - Blackman     самый мягкий (−58 dB), максимальный loss
 *
 * Источник: PyPanelAntennas/SNR/REPORT.md
 *
 * @author Kodo (AI Assistant)
 * @date 2026-04-09
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
