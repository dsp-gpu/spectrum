#pragma once

/**
 * @file spectrum_types.hpp
 * @brief Aggregator: подключает все типы spectrum maxima (modes + params + results + profiling).
 *
 * @note Тип B (technical header): только #include'ы, без объявлений.
 *       Удобно подключать одной строкой потребителями SpectrumMaximaFinder.
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <spectrum/types/spectrum_modes.hpp>
#include <spectrum/types/spectrum_params.hpp>
#include <spectrum/types/spectrum_result_types.hpp>
#include <spectrum/types/spectrum_profiling.hpp>
