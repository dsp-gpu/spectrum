#pragma once

/**
 * @file fft_modes.hpp
 * @brief Режимы вывода FFT
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

namespace fft_processor {

enum class FFTOutputMode {
    COMPLEX,             ///< Возвращает complex<float>[nFFT] — сырой FFT-спектр (re + im).
                         ///< Нормализация: результат совместим с np.fft.fft() (без деления на N)
    MAGNITUDE_PHASE,     ///< Возвращает magnitude[nFFT] + phase[nFFT] в радианах [-π, π].
                         ///< GPU-kernel: __fsqrt_rn(re²+im²) и atan2f(im, re)
    MAGNITUDE_PHASE_FREQ ///< То же + frequency[nFFT] в Гц: freq[k] = k × sample_rate / nFFT.
                         ///< frequency[] вычисляется на CPU после download (без дополнительного kernel)
};

}  // namespace fft_processor
