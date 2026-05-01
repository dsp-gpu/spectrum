#pragma once

/**
 * @file fft_modes.hpp
 * @brief Режимы вывода FFTProcessor: COMPLEX / MAGNITUDE_PHASE / MAGNITUDE_PHASE_FREQ.
 *
 * @note Тип B (technical header): только enum, без логики.
 *       Семантика каждого режима — в комментариях значений.
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace fft_processor {

/**
 * @enum FFTOutputMode
 * @brief Формат возвращаемых FFTProcessor данных.
 *
 * @note От режима зависит набор kernel'ов на post-stage и тип результата
 *       (FFTComplexResult vs FFTMagPhaseResult).
 */
enum class FFTOutputMode {
    COMPLEX,             ///< Возвращает complex<float>[nFFT] — сырой FFT-спектр (re + im).
                         ///< Нормализация: результат совместим с np.fft.fft() (без деления на N)
    MAGNITUDE_PHASE,     ///< Возвращает magnitude[nFFT] + phase[nFFT] в радианах [-π, π].
                         ///< GPU-kernel: __fsqrt_rn(re²+im²) и atan2f(im, re)
    MAGNITUDE_PHASE_FREQ ///< То же + frequency[nFFT] в Гц: freq[k] = k × sample_rate / nFFT.
                         ///< frequency[] вычисляется на CPU после download (без дополнительного kernel)
};

}  // namespace fft_processor
