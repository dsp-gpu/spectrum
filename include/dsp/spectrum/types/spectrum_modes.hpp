#pragma once

/**
 * @file spectrum_modes.hpp
 * @brief Режимы поиска пиков (fft_func spectrum)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

namespace antenna_fft {

/**
 * @enum PeakSearchMode
 * @brief Режим поиска пиков в спектре
 */
enum class PeakSearchMode {
    ONE_PEAK,    ///< Один пик (левый или правый) → 4 MaxValue
    TWO_PEAKS,   ///< Два пика (левый и правый) → 8 MaxValue
    ALL_MAXIMA   ///< Все локальные максимумы
};

}  // namespace antenna_fft
