#pragma once

/**
 * @file fft_results.hpp
 * @brief Типы результатов FFTProcessor: complex spectrum / mag+phase / локальное профилирование.
 *
 * @note Тип B (technical header): POD-struct'ы без логики.
 *       Для централизованного профилирования (JSON, MD) — `ProfilingFacade`.
 *       `FFTProfilingData` — локальный legacy-формат для `GetProfilingData()`
 *       (исторически заполнялся из GPUProfiler::GetStats(), сейчас — best-effort).
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <cstdint>
#include <complex>
#include <vector>

namespace fft_processor {

/**
 * @struct FFTBeamResult
 * @brief База результата FFT для одного луча (общие поля для Complex/MagPhase).
 */
struct FFTBeamResult {
    uint32_t beam_id = 0;      ///< Глобальный индекс луча (с учётом батчей); совпадает с позицией в исходном массиве
    uint32_t nFFT = 0;         ///< Размер FFT после zero-padding: nextPow2(n_point) × repeat_count.
                               ///< Важно: nFFT ≥ n_point! Размер spectrum/magnitude/phase = nFFT
    float sample_rate = 0.0f;  ///< Копия из FFTProcessorParams.sample_rate — нужна для интерпретации frequency[]
};

/**
 * @struct FFTComplexResult
 * @brief Результат FFT в комплексном формате (режим COMPLEX).
 */
struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum; ///< FFT-спектр [nFFT], ненормализован.
                                               ///< Для физической амплитуды: amplitude = |spectrum[k]| / n_point
};

/**
 * @struct FFTMagPhaseResult
 * @brief Результат FFT в формате магнитуда+фаза (MAGNITUDE_PHASE / MAGNITUDE_PHASE_FREQ).
 *
 * @note frequency[] заполняется только для MAGNITUDE_PHASE_FREQ; иначе пуст.
 */
struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude;  ///< |X[k]| = sqrt(re²+im²), ненормализован [nFFT]
    std::vector<float> phase;      ///< arg(X[k]) = atan2(im, re), радианы [-π, π] [nFFT]
    std::vector<float> frequency;  ///< freq[k] = k × sample_rate / nFFT, Гц [nFFT].
                                   ///< ПУСТОЙ при output_mode == MAGNITUDE_PHASE — заполняется только для MAGNITUDE_PHASE_FREQ
};

/**
 * @struct FFTProfilingData
 * @brief Локальные данные профилирования FFTProcessor::GetProfilingData() (legacy).
 *
 * @note Для детального профилирования по стадиям — использовать `ProfilingFacade`
 *       + ROCmProfEvents напрямую (см. core/services/profiling).
 */
struct FFTProfilingData {
    double upload_time_ms = 0.0;           ///< CPU→GPU transfer (H2D или D2D copy)
    double fft_time_ms = 0.0;             ///< Чистое FFT-вычисление (hipfftExecC2C)
    double post_processing_time_ms = 0.0; ///< Kernel complex→mag+phase (0 если режим COMPLEX)
    double download_time_ms = 0.0;        ///< GPU→CPU transfer (D2H)
    double total_time_ms = 0.0;           ///< Суммарное время из GPUProfiler (включая overhead)
};

}  // namespace fft_processor
