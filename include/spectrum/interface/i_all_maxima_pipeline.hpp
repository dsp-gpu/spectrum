#pragma once

// ============================================================================
// IAllMaximaPipeline — интерфейс пайплайна поиска ВСЕХ максимумов спектра
//
// ЧТО:    Pure-virtual интерфейс трёхстадийного pipeline'а на готовых
//         амплитудах: Detect (выделить локальные максимумы) → Scan
//         (prefix-sum для упаковки) → Compact (собрать в плотный массив).
//         Реализация — AllMaximaPipelineROCm (hipFFT + rocPRIM).
//
// ЗАЧЕМ:  В отличие от ISpectrumProcessor::ProcessFromCPU (находит TOP-1 или
//         TOP-2 пика за beam), AllMaxima возвращает ВСЕ локальные максимумы
//         выше threshold'а — нужно для radar-сценария когда целей несколько
//         (multi-target detection). Отделено от ISpectrumProcessor для ISP
//         (Interface Segregation): не все клиенты нужны AllMaxima.
//
// ПОЧЕМУ: - Узкий интерфейс из 1 метода Execute() — ISP, не плодим обязательства.
//         - Вход — уже-вычисленные magnitudes (а не raw FFT) — позволяет
//           переиспользовать magnitudes из ProcessFromCPU без двойного счёта.
//         - max_maxima_per_beam = 1000 default — типовой radar-сценарий
//           (256 целей реалистичный максимум, 1000 с запасом).
//         - search_start / search_end — фильтрация частотного диапазона
//           (не интересны DC и зеркало для real-input).
//         - Bridge: ISpectrumProcessor::FindAllMaxima* делегирует сюда,
//           каждый backend держит свой IAllMaximaPipeline.
//
// Использование:
//   // Внутри SpectrumProcessorROCm:
//   AllMaximaResult result = pipeline_->Execute(
//       magnitudes_buffer_, fft_output_,
//       beam_count, nFFT, sample_rate,
//       OutputDestination::CPU, 1, nFFT/2, 1000);
//
// История:
//   - Создан:  2026-02-15 (Phase 3 refactoring — выделение из ISpectrumProcessor)
//   - Изменён: 2026-04-15 (миграция в DSP-GPU/spectrum)
// ============================================================================

#include <spectrum/interface/spectrum_maxima_types.h>

#include <cstdint>

namespace antenna_fft {

/**
 * @class IAllMaximaPipeline
 * @brief Pure-virtual интерфейс пайплайна Detect → Scan → Compact на спектральных амплитудах.
 *
 * @note Pure interface — единственный метод Execute().
 * @note Вход: GPU-буфер с |FFT[i]| (beam_count × nFFT).
 * @note Выход: AllMaximaResult — позиции / амплитуды / частоты максимумов.
 * @see AllMaximaPipelineROCm — ROCm-реализация (hipFFT + rocPRIM)
 * @see ISpectrumProcessor::FindAllMaxima — фасадная точка входа
 */
class IAllMaximaPipeline {
public:
    virtual ~IAllMaximaPipeline() = default;

    /**
     * @brief Выполнить pipeline на pre-computed magnitudes.
     * @param magnitudes_gpu       GPU-буфер с |FFT[i]| (float*, beam_count × nFFT элементов).
     * @param fft_data_gpu         GPU-буфер с complex FFT (для пересчёта позиции в частоту).
     * @param beam_count           Число beams.
     * @param nFFT                 Размер FFT (степень двойки).
     * @param sample_rate          Sample rate (Hz) для пересчёта в частоту.
     * @param dest                 Куда вернуть результат: CPU (D2H) или GPU (остаётся в device-памяти).
     * @param search_start         Начало поиска (bin) — обычно 1, чтобы пропустить DC.
     * @param search_end           Конец поиска (bin) — 0 = до nFFT/2 (для real-input).
     * @param max_maxima_per_beam  Максимум максимумов на один beam (default 1000).
     */
    virtual AllMaximaResult Execute(
        void* magnitudes_gpu,
        void* fft_data_gpu,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 1,
        uint32_t search_end = 0,
        size_t max_maxima_per_beam = 1000) = 0;
};

} // namespace antenna_fft
