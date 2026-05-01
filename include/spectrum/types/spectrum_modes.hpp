#pragma once

/**
 * @file spectrum_modes.hpp
 * @brief Режимы поиска пиков спектра: ONE_PEAK / TWO_PEAKS / ALL_MAXIMA.
 *
 * @note Тип B (technical header): только enum, без логики.
 *       Используется SpectrumMaximaFinder и pipeline'ами AllMaximaPipelineROCm.
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
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
