#pragma once

/**
 * @file fft_params.hpp
 * @brief Параметры FFTProcessor
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "types/fft_modes.hpp"

#include <cstdint>

namespace fft_processor {

struct FFTProcessorParams {
    uint32_t beam_count = 1;   ///< Количество параллельных лучей; каждый луч — один независимый FFT
    uint32_t n_point = 0;      ///< Число реальных точек на луч (до zero-padding).
                               ///< nFFT = nextPow2(n_point) × repeat_count.
                               ///< В Python-биндинге: 0 → автоопределение из формы массива
    float sample_rate = 1000.0f;  ///< Частота дискретизации, Гц.
                                  ///< Нужна для: freq[k] = k × sample_rate / nFFT (режим MAGNITUDE_PHASE_FREQ)
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;  ///< Формат возвращаемых данных
    uint32_t repeat_count = 1;  ///< Коэффициент zero-padding сверх nextPow2.
                                ///< repeat_count=2 → nFFT = nextPow2(n_point) × 2.
                                ///< Увеличивает интерполяцию в частотной области, НЕ добавляет информацию.
    float memory_limit = 0.80f; ///< Предел GPU-памяти для одного батча (0.0..1.0, доля от свободной).
                                ///< BatchManager делит лучи на батчи, чтобы не превысить этот лимит.
};

}  // namespace fft_processor
