#pragma once

/**
 * @file spectrum_result_types.hpp
 * @brief Типы результатов поиска максимумов
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <core/interface/output_destination.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace antenna_fft {

using drv_gpu_lib::OutputDestination;

// Один максимум спектра — GPU-side layout compact_maxima/post_kernel, 32 байта с pad.
// Заполняется post_kernel (ONE/TWO_PEAKS) или compact_kernel (ALL_MAXIMA).
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

// Результат одной антенны/луча (ONE_PEAK или TWO_PEAKS).
// interpolated — параболически уточнённый пик; left/center/right — соседние бины.
struct SpectrumResult {
    uint32_t antenna_id;          ///< Индекс антенны в batch
    MaxValue interpolated;        ///< Главный пик (peak[0]) с параболической интерполяцией
    MaxValue left_point;          ///< FFT[peak_bin - 1] — левый сосед
    MaxValue center_point;        ///< FFT[peak_bin]     — сам пик
    MaxValue right_point;         ///< FFT[peak_bin + 1] — правый сосед
};

// Устаревший тип — обёртка над двумя SpectrumResult для совместимости со старым API
struct CPUSpectrumResult {
    SpectrumResult SpectrMax_left, SpectrMax_right;
};

// Все максимумы одного луча (AllMaxima pipeline).
// num_maxima ≤ max_maxima_per_beam; maxima заполнено compact_kernel'ом.
struct AllMaximaBeamResult {
    uint32_t antenna_id;            ///< Индекс луча
    uint32_t num_maxima;            ///< Реальное число найденных максимумов (0..max_maxima_per_beam)
    std::vector<MaxValue> maxima;   ///< Результаты (CPU copy); пусто при Dest=GPU
};

// Выходной контейнер FindAllMaxima/AllMaxima.
// ⚠️ OWNERSHIP: при destination=GPU или ALL — caller ОБЯЗАН освободить gpu_maxima и gpu_counts!
//    OpenCL: clReleaseMemObject(static_cast<cl_mem>(result.gpu_maxima))
//    ROCm:   hipFree(result.gpu_maxima)
// При destination=CPU — gpu_maxima/gpu_counts == nullptr (освобождены внутри).
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
