#pragma once

/**
 * @file spectrum_params.hpp
 * @brief Параметры поиска максимума спектра
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <spectrum/types/spectrum_modes.hpp>

#include <cstddef>
#include <cstdint>

namespace antenna_fft {

/**
 * @struct SpectrumParams
 * @brief Параметры для поиска максимума спектра (заполняется из InputData)
 *
 * В новом API заполняется через PrepareParams() из InputData<T>.
 * nFFT и base_fft вычисляются автоматически, не задаются вручную.
 */
struct SpectrumParams {
    uint32_t antenna_count = 5;           ///< Количество антенн/лучей в batch
    uint32_t n_point = 1000;              ///< Точек на луч (длина сигнала); для AllMaxima: n_point = nFFT
    uint32_t repeat_count = 2;            ///< Коэффициент zero-padding: nFFT = nextPow2(n_point) × repeat_count
                                          ///< repeat_count=1 — минимальный FFT, repeat_count=4 — высокое частотное разрешение
    float sample_rate = 1000.0f;          ///< Частота дискретизации в Гц (определяет fs/nFFT → Δf на бин)
    uint32_t search_range = 0;            ///< Диапазон поиска пика [0, search_range); 0 = авто = nFFT/4
                                          ///< Ограничение: ищем только первую четверть спектра (положительные частоты)
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK;  ///< ONE_PEAK / TWO_PEAKS / ALL_MAXIMA
    float memory_limit = 0.80f;           ///< Доля GPU памяти для batch (0.0-1.0); 0.8 = 80% доступной VRAM
    size_t max_maxima_per_beam = 1000;    ///< Лимит максимумов на луч для AllMaxima (ограничивает размер out_maxima)
    uint32_t nFFT = 0;                    ///< ВЫЧИСЛЯЕМОЕ: nextPow2(n_point) × repeat_count (≥ n_point, степень двойки)
    uint32_t base_fft = 0;               ///< ВЫЧИСЛЯЕМОЕ: nextPow2(n_point) — ближайшая степень двойки к n_point
};

}  // namespace antenna_fft
