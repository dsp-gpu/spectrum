#pragma once

/**
 * @file mag_phase_types.hpp
 * @brief Типы для ComplexToMagPhaseROCm — прямая конверсия complex → magnitude+phase.
 *
 * @note Тип B (technical header): POD struct'ы для standalone-конвертера (без FFT).
 *       Легче чем FFTMagPhaseResult: нет nFFT/sample_rate/frequency.
 * @note norm_coeff: 0=skip (×1), -1=÷n_point (×1/n), >0=multiply by value.
 *
 * История:
 *   - Создан:  2026-03-01
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <cstdint>
#include <vector>

namespace fft_processor {

/**
 * @struct MagPhaseParams
 * @brief Параметры ComplexToMagPhaseROCm-конверсии (без FFT).
 */
struct MagPhaseParams {
    uint32_t beam_count   = 1;     ///< Number of parallel beams/channels
    uint32_t n_point      = 0;     ///< Points per beam
    float    memory_limit = 0.80f; ///< Fraction of GPU memory to use (0.0 .. 1.0)
    float    norm_coeff   = 1.0f;  ///< Normalization: 0=skip (×1), -1=÷n_point (×1/n), >0=multiply by value
};

/**
 * @struct MagPhaseResult
 * @brief Результат complex→mag/phase конверсии (один на луч).
 */
struct MagPhaseResult {
    uint32_t beam_id = 0;         ///< Beam index
    uint32_t n_point = 0;         ///< Points in this beam
    std::vector<float> magnitude;  ///< |z| = sqrt(re^2 + im^2)
    std::vector<float> phase;      ///< arg(z) = atan2(im, re) [radians]
};

/**
 * @struct MagnitudeResult
 * @brief Результат magnitude-only конверсии (фаза не считается).
 */
struct MagnitudeResult {
    uint32_t beam_id = 0;         ///< Beam index
    uint32_t n_point = 0;         ///< Points in this beam
    std::vector<float> magnitude;  ///< |z| * inv_n (normalization applied)
};

}  // namespace fft_processor
