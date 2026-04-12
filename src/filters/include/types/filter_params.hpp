#pragma once

/**
 * @file filter_params.hpp
 * @brief Parameter types for FIR and IIR filters
 *
 * BiquadSection, FirParams, IirParams, FilterConfig (JSON loading).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace filters {

// ════════════════════════════════════════════════════════════════════════════
// BiquadSection: one second-order IIR section
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct BiquadSection
 * @brief Coefficients for one biquad (second-order section)
 *
 * Transfer function:
 *   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
 *
 * Difference equation (Direct Form II Transposed):
 *   y[n] = b0*x[n] + w1
 *   w1   = b1*x[n] - a1*y[n] + w2
 *   w2   = b2*x[n] - a2*y[n]
 *
 * @note a0 is always 1.0 (normalized). SciPy: butter(N, Wn, output='sos')
 */
struct BiquadSection {
  float b0 = 1.0f;
  float b1 = 0.0f;
  float b2 = 0.0f;
  float a1 = 0.0f;  ///< Feedback coeff (a0=1, normalized)
  float a2 = 0.0f;  ///< Feedback coeff
};

// ════════════════════════════════════════════════════════════════════════════
// FirParams / IirParams
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FirParams
 * @brief Parameters for FIR filter operation
 */
struct FirParams {
  std::vector<float> coefficients;  ///< h[k], length = num_taps
  uint32_t channels = 1;           ///< Number of parallel channels
  uint32_t points   = 1024;        ///< Samples per channel
};

/**
 * @struct IirParams
 * @brief Parameters for IIR filter operation (cascade of biquads)
 */
struct IirParams {
  std::vector<BiquadSection> sections;  ///< Cascade of biquad sections
  uint32_t channels = 1;
  uint32_t points   = 1024;
};

// ════════════════════════════════════════════════════════════════════════════
// FilterConfig: JSON loading
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FilterConfig
 * @brief Universal filter configuration loaded from JSON
 *
 * JSON formats:
 * @code
 * // FIR:
 * { "type": "fir", "coefficients": [0.1, 0.2, ...] }
 *
 * // IIR (SOS):
 * { "type": "iir", "sections": [
 *     {"b0":0.5, "b1":1.0, "b2":0.5, "a1":-0.3, "a2":0.1},
 *     ...
 * ]}
 * @endcode
 */
struct FilterConfig {
  std::string type;                    ///< "fir" or "iir"
  std::vector<float> coefficients;     ///< FIR coefficients
  std::vector<BiquadSection> sections; ///< IIR biquad sections

  /**
   * @brief Load filter config from JSON file (minimal parser, no nlohmann)
   * @param path Path to JSON file
   * @return FilterConfig populated from file
   * @throws std::runtime_error on parse error
   */
  static FilterConfig LoadJson(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error(
          "FilterConfig::LoadJson: cannot open " + path);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    FilterConfig cfg;

    // Detect type
    if (content.find("\"fir\"") != std::string::npos) {
      cfg.type = "fir";
    } else if (content.find("\"iir\"") != std::string::npos) {
      cfg.type = "iir";
    } else {
      throw std::runtime_error(
          "FilterConfig::LoadJson: no \"fir\" or \"iir\" type found");
    }

    if (cfg.type == "fir") {
      // Parse "coefficients": [...]
      auto pos = content.find("\"coefficients\"");
      if (pos == std::string::npos)
        throw std::runtime_error("FilterConfig: no 'coefficients' key");

      auto arr_start = content.find('[', pos);
      auto arr_end   = content.find(']', arr_start);
      if (arr_start == std::string::npos || arr_end == std::string::npos)
        throw std::runtime_error("FilterConfig: malformed coefficients array");

      cfg.coefficients = ParseFloatArray(
          content.substr(arr_start + 1, arr_end - arr_start - 1));

    } else {
      // IIR: parse "sections": [{...}, ...]
      auto pos = content.find("\"sections\"");
      if (pos == std::string::npos)
        throw std::runtime_error("FilterConfig: no 'sections' key");

      // Find each {...} block
      size_t search_from = pos;
      while (true) {
        auto brace_start = content.find('{', search_from);
        if (brace_start == std::string::npos) break;
        // Skip the outer { of the whole JSON
        if (brace_start < pos) { search_from = brace_start + 1; continue; }

        auto brace_end = content.find('}', brace_start);
        if (brace_end == std::string::npos) break;

        std::string section_str =
            content.substr(brace_start, brace_end - brace_start + 1);

        BiquadSection sec;
        sec.b0 = ParseJsonFloat(section_str, "\"b0\"");
        sec.b1 = ParseJsonFloat(section_str, "\"b1\"");
        sec.b2 = ParseJsonFloat(section_str, "\"b2\"");
        sec.a1 = ParseJsonFloat(section_str, "\"a1\"");
        sec.a2 = ParseJsonFloat(section_str, "\"a2\"");
        cfg.sections.push_back(sec);

        search_from = brace_end + 1;
      }

      if (cfg.sections.empty())
        throw std::runtime_error("FilterConfig: no biquad sections parsed");
    }

    return cfg;
  }

private:
  /// Parse comma-separated floats from a string
  static std::vector<float> ParseFloatArray(const std::string& s) {
    std::vector<float> result;
    std::string num;
    for (char c : s) {
      if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
          c == 'e' || c == 'E' || c == '+') {
        num += c;
      } else if (!num.empty()) {
        result.push_back(std::stof(num));
        num.clear();
      }
    }
    if (!num.empty()) result.push_back(std::stof(num));
    return result;
  }

  /// Parse a float value after a JSON key
  static float ParseJsonFloat(const std::string& s, const std::string& key) {
    auto pos = s.find(key);
    if (pos == std::string::npos) return 0.0f;
    pos = s.find(':', pos);
    if (pos == std::string::npos) return 0.0f;
    pos++;
    // Skip whitespace
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
    std::string num;
    for (; pos < s.size(); pos++) {
      char c = s[pos];
      if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
          c == 'e' || c == 'E' || c == '+') {
        num += c;
      } else break;
    }
    return num.empty() ? 0.0f : std::stof(num);
  }
};

// ════════════════════════════════════════════════════════════════════════════
// Moving Average Filter (ROCm)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @enum MAType
 * @brief Type of moving average filter
 */
enum class MAType {
  SMA,   ///< Simple MA — ring buffer, equal weights 1/N
  EMA,   ///< Exponential MA — alpha = 2/(N+1)
  MMA,   ///< Modified MA (Wilder) — alpha = 1/N
  DEMA,  ///< Double EMA — 2*EMA1 - EMA2
  TEMA   ///< Triple EMA — 3*EMA1 - 3*EMA2 + EMA3
};

/**
 * @struct MovingAverageParams
 * @brief Parameters for moving average filter
 */
struct MovingAverageParams {
  MAType   type        = MAType::EMA;  ///< Filter type
  uint32_t window_size = 10;           ///< N — window size (for SMA: max 128)
};

// ════════════════════════════════════════════════════════════════════════════
// Kalman Filter (ROCm)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct KalmanParams
 * @brief Parameters for 1D scalar Kalman filter
 *
 * Applied to Re and Im parts independently.
 *
 * Parameter selection:
 *   R — measurement noise variance. Start with: R = (FFT_bin_size)^2 / 12
 *   Q — process noise variance. Start with: Q = R / 100
 *   Q/R << 1: strong smoothing, slow reaction
 *   Q/R >> 1: weak smoothing, fast reaction
 */
struct KalmanParams {
  float Q  = 0.1f;    ///< Process noise variance
  float R  = 25.0f;   ///< Measurement noise variance
  float x0 = 0.0f;    ///< Initial state estimate
  float P0 = 25.0f;   ///< Initial error covariance (usually = R)
};

// ════════════════════════════════════════════════════════════════════════════
// Kaufman Adaptive Moving Average — KAMA (ROCm)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct KaufmanParams
 * @brief Parameters for KAMA filter
 *
 * Algorithm:
 *   ER  = |x[n] - x[n-N]| / sum(|x[i] - x[i-1]|, i=n-N+1..n)
 *   SC  = (ER * (fast_sc - slow_sc) + slow_sc)^2
 *   KAMA[n] = KAMA[n-1] + SC * (x[n] - KAMA[n-1])
 *
 * Standard Kaufman: er_period=10, fast=2, slow=30
 */
struct KaufmanParams {
  uint32_t er_period   = 10;  ///< N — Efficiency Ratio period (max 128)
  uint32_t fast_period = 2;   ///< Fast EMA period (when ER≈1)
  uint32_t slow_period = 30;  ///< Slow EMA period (when ER≈0)
};

} // namespace filters
