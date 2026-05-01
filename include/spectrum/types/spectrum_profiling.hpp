#pragma once

/**
 * @file spectrum_profiling.hpp
 * @brief Локальный legacy-формат данных профилирования модуля spectrum (ProfilingData).
 *
 * @note Тип B (technical header): один POD-struct для GetProfilingData().
 *       Для централизованного профилирования (JSON / MD / per-stage) — `ProfilingFacade`
 *       (см. core/services/profiling/profiling_facade.hpp).
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace antenna_fft {

/**
 * @struct ProfilingData
 * @brief Локальные данные профилирования (время в мс) для SpectrumProcessor::GetProfilingData().
 */
struct ProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_kernel_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};

}  // namespace antenna_fft
