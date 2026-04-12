#pragma once

/**
 * @file filter_modes.hpp
 * @brief Filter processing modes and precision settings
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
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
