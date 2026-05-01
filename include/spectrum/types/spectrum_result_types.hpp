#pragma once

/**
 * @file spectrum_result_types.hpp
 * @brief Типы результатов поиска максимумов спектра (MaxValue, SpectrumResult, AllMaximaResult).
 *
 * @note Тип B (technical header): POD-struct'ы под GPU-side layout (32 байта с pad).
 * @note ⚠️ OWNERSHIP в AllMaximaResult: при destination=GPU/ALL caller ОБЯЗАН
 *       освободить gpu_maxima/gpu_counts (hipFree для ROCm, clReleaseMemObject для OpenCL).
 *       При destination=CPU указатели == nullptr (освобождены внутри pipeline).
 *
 * История:
 *   - Создан:  2026-02-15
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <core/interface/output_destination.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace antenna_fft {

using drv_gpu_lib::OutputDestination;

/**
 * @struct MaxValue
 * @brief Один максимум спектра — GPU-side layout (32 байта с pad).
 *
 * @note Заполняется post_kernel (ONE/TWO_PEAKS) или compact_kernel (ALL_MAXIMA).
 *       phase в ГРАДУСАХ (не радианах!), refined_frequency = (index + δ) × fs/nFFT.
 */
struct MaxValue {
    uint32_t index;               ///< Бин FFT — позиция в спектре [0, nFFT)
    float real, imag;             ///< Re/Im FFT[index] — для фазового анализа
    float magnitude;              ///< |FFT[index]| = sqrt(real²+imag²)
    float phase;                  ///< arg(FFT[index]) в ГРАДУСАХ = atan2(imag,real)×180/π
    float freq_offset;            ///< Дробная часть параболической интерполяции δ ∈ [-0.5, +0.5]
    float refined_frequency;      ///< Уточнённая частота в Гц = (index + δ) × sample_rate / nFFT
                                  ///< В compact_maxima δ=0 (без интерполяции): index×fs/nFFT
    uint32_t pad;                 ///< Выравнивание до 32 байт
};

/**
 * @struct SpectrumResult
 * @brief Результат одной антенны/луча (режим ONE_PEAK или TWO_PEAKS).
 *
 * @note interpolated — параболически уточнённый пик;
 *       left/center/right — соседние бины (без интерполяции).
 */
struct SpectrumResult {
    uint32_t antenna_id;          ///< Индекс антенны в batch
    MaxValue interpolated;        ///< Главный пик (peak[0]) с параболической интерполяцией
    MaxValue left_point;          ///< FFT[peak_bin - 1] — левый сосед
    MaxValue center_point;        ///< FFT[peak_bin]     — сам пик
    MaxValue right_point;         ///< FFT[peak_bin + 1] — правый сосед
};

/**
 * @struct CPUSpectrumResult
 * @brief Устаревшая обёртка над двумя SpectrumResult — для совместимости со старым API.
 * @deprecated Использовать SpectrumResult[] напрямую.
 */
struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left, SpectrMax_right;
};

/**
 * @struct AllMaximaBeamResult
 * @brief Все максимумы одного луча (AllMaxima pipeline).
 *
 * @note num_maxima ≤ max_maxima_per_beam; maxima пуст при destination=GPU.
 */
struct AllMaximaBeamResult {
    uint32_t antenna_id;            ///< Индекс луча
    uint32_t num_maxima;            ///< Реальное число найденных максимумов (0..max_maxima_per_beam)
    std::vector<MaxValue> maxima;   ///< Результаты (CPU copy); пусто при Dest=GPU
};

/**
 * @struct AllMaximaResult
 * @brief Выходной контейнер FindAllMaxima/AllMaxima pipeline.
 *
 * @note ⚠️ OWNERSHIP: при destination=GPU/ALL caller ОБЯЗАН освободить gpu_maxima и gpu_counts!
 *       OpenCL: clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima))
 *       ROCm:   hipFree(result.gpu_maxima)
 *       При destination=CPU указатели == nullptr (освобождены внутри pipeline).
 */
struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams;  ///< Результаты по лучам (заполнено при Dest=CPU/ALL)
    OutputDestination destination;
    void* gpu_maxima = nullptr;   ///< GPU буфер MaxValue[beam_count×max_per_beam] — caller owns при GPU/ALL!
    void* gpu_counts = nullptr;   ///< GPU буфер uint32[beam_count] — caller owns при GPU/ALL!
    size_t total_maxima = 0;      ///< Суммарное число найденных максимумов
    size_t gpu_bytes = 0;         ///< Размер gpu_maxima в байтах
    size_t TotalMaxima() const { return total_maxima; }
};

}  // namespace antenna_fft
