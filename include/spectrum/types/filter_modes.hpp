#pragma once

/**
 * @file filter_modes.hpp
 * @brief Режимы и точность вычисления фильтров (FilterPrecision, FirAlgorithm).
 *
 * @note Тип B (technical header): только enum'ы, без логики.
 *       FirAlgorithm::Tiled / OLS_FFT — заглушки на будущее, сейчас работает Direct.
 *
 * История:
 *   - Создан:  2026-02-18
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace filters {

/**
 * @enum FilterPrecision
 * @brief Precision of filter computation
 */
enum class FilterPrecision {
  Float32,  ///< Single precision (default, fast)
  Float64   ///< Double precision (requires GPU support)
};

/**
 * @enum FirAlgorithm
 * @brief FIR implementation strategy (auto-selected based on num_taps)
 *
 * Decision tree:
 *   num_taps <= 128   -> Direct (each work-item computes one output)
 *   num_taps <= 512   -> Tiled  (local memory optimization) [future]
 *   num_taps > 512    -> FFT    (Overlap-Save via FFTProcessor) [future]
 */
enum class FirAlgorithm {
  Direct,   ///< Direct-form convolution (__constant coeffs)
  Tiled,    ///< Local memory tiled approach [future]
  OLS_FFT   ///< Overlap-Save via FFT [future]
};

} // namespace filters
