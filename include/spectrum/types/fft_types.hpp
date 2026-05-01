#pragma once

/**
 * @file fft_types.hpp
 * @brief Aggregator: подключает все типы fft_processor (modes + params + results).
 *
 * @note Тип B (technical header): только #include'ы, без объявлений.
 *       Удобно подключать одной строкой в кодах потребителей FFTProcessor.
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <spectrum/types/fft_modes.hpp>
#include <spectrum/types/fft_params.hpp>
#include <spectrum/types/fft_results.hpp>
