#pragma once

// ============================================================================
// ISpectrumProcessor — интерфейс GPU-обработки спектра (Strategy + Bridge)
//
// ЧТО:    Pure-virtual интерфейс главного процессора модуля spectrum: FFT
//         (через hipFFT), поиск пиков (ONE_PEAK / TWO_PEAKS), полный
//         AllMaxima pipeline (Detect → Scan → Compact). Реализация —
//         SpectrumProcessorROCm (hipFFT + hiprtc). Старая ветка nvidia
//         содержит SpectrumProcessorOpenCL (заморожена).
//
// ЗАЧЕМ:  Без интерфейса фасад SpectrumMaximaFinder зависел бы от конкретной
//         реализации (нарушение DIP) → нельзя подменить backend для тестов
//         или будущих платформ. Через ISpectrumProcessor* фасад работает с
//         любым backend, а Factory выбирает по BackendType.
//
// ПОЧЕМУ: - Strategy GoF — фасад (`SpectrumMaximaFinder`) хранит
//           unique_ptr<ISpectrumProcessor>; реализация выбирается через
//           SpectrumProcessorFactory::Create(BackendType, IBackend*).
//         - Bridge GoF — отделяет «семантику spectrum» (что считаем) от
//           «реализации backend» (как считаем). Позволяет добавить новый
//           backend без правки фасада.
//         - Богатый интерфейс (~14 методов) — намеренно: фасад делегирует
//           всё, не только Process(). ProcessBatch / FindAllMaxima* /
//           ReallocateBuffersForBatch — поведение, специфичное для
//           backend (hipFFT plan vs OpenCL plan, batch sizing).
//         - Initialize отдельно от ctor — нужны hipFFT plan + kernel compile,
//           они тяжёлые (~50-200ms) и должны быть отделены от создания
//           объекта (можно создать → Initialize при первом use).
//         - GetProfilingData() возвращает агрегированные данные модуля —
//           backend знает свои стадии лучше фасада.
//
// Использование:
//   auto proc = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);
//   proc->Initialize(params);
//   auto results = proc->ProcessFromCPU(iq_data);  // ONE_PEAK/TWO_PEAKS
//   auto all_max = proc->FindAllMaximaFromCPU(iq_data, OutputDestination::CPU, 0, 0);
//
// История:
//   - Создан:  2026-02-15 (Strategy + Bridge для multi-backend)
//   - Изменён: 2026-04-15 (миграция в DSP-GPU/spectrum, ROCm — main backend)
// ============================================================================

#include <spectrum/interface/spectrum_maxima_types.h>
#include <spectrum/interface/spectrum_input_data.hpp>

#include <vector>
#include <complex>
#include <memory>
#include <cstdint>

namespace antenna_fft {

/**
 * @class ISpectrumProcessor
 * @brief Pure-virtual интерфейс GPU-обработки спектра (FFT + peak search + AllMaxima).
 *
 * @note Pure interface — все методы обязательны. Реализации:
 *       SpectrumProcessorROCm (hipFFT, ROCm — main backend),
 *       SpectrumProcessorOpenCL (legacy, ветка nvidia).
 * @note Initialize отделён от ctor (heavy: hipFFT plan + kernel compile).
 * @see SpectrumProcessorFactory — Factory Method для выбора backend
 * @see SpectrumProcessorROCm — основная ROCm-реализация
 */
class ISpectrumProcessor {
public:
    virtual ~ISpectrumProcessor() = default;

    /**
     * @brief Инициализировать GPU-ресурсы под параметры (FFT plan, buffers, kernels).
     * @param params Параметры спектра (antenna_count, n_point, nFFT, sample_rate и т.д.).
     */
    virtual void Initialize(const SpectrumParams& params) = 0;

    /**
     * @brief Готов ли процессор к обработке (после успешного Initialize).
     */
    virtual bool IsInitialized() const = 0;

    /**
     * @brief Обработать CPU-данные (ONE_PEAK или TWO_PEAKS режим).
     *
     * Pipeline: Upload H2D → FFT → PostKernel → ReadResults D2H.
     */
    virtual std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) = 0;

    /**
     * @brief Обработать GPU-данные (ONE_PEAK или TWO_PEAKS режим).
     *
     * Pipeline: GPU copy D2D → FFT → PostKernel → ReadResults D2H.
     * @param gpu_data         OpenCL: cl_mem, ROCm: hipDeviceptr_t (передаётся как void*).
     * @param gpu_memory_bytes Фактический размер GPU-буфера в байтах.
     */
    virtual std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) = 0;

    /**
     * @brief Обработать одну batch-порцию из CPU-данных (вызывается фасадом при batch-оркестрации).
     */
    virtual std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) = 0;

    /**
     * @brief Обработать одну batch-порцию из GPU-буфера.
     */
    virtual std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) = 0;

    /**
     * @brief Полный pipeline AllMaxima из CPU-данных: Upload → FFT → Detect→Scan→Compact.
     */
    virtual AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief Полный pipeline AllMaxima из GPU-буфера: D2D → FFT → Detect→Scan→Compact.
     */
    virtual AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief AllMaxima из готовых CPU FFT-данных (без FFT — данные уже преобразованы).
     */
    virtual AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) = 0;

    /**
     * @brief Найти все максимумы в готовых GPU FFT-данных (Detect → Scan → Compact).
     * @param fft_data GPU-буфер с complex FFT-результатом.
     */
    virtual AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) = 0;

    /**
     * @brief Получить тип backend'а реализации (ROCm / OPENCL).
     */
    virtual DriverType GetDriverType() const = 0;

    /**
     * @brief Получить агрегированные данные профилирования модуля.
     */
    virtual ProfilingData GetProfilingData() const = 0;

    /**
     * @brief Переаллоцировать буферы под новый размер batch'а.
     */
    virtual void ReallocateBuffersForBatch(size_t batch_antenna_count) = 0;

    /**
     * @brief Сколько байт нужно на одну антенну (для BatchManager — расчёт max batch).
     */
    virtual size_t CalculateBytesPerAntenna() const = 0;

    /**
     * @brief Скомпилировать post-kernel (lazy init, может вызываться явно).
     */
    virtual void CompilePostKernel() = 0;
};

} // namespace antenna_fft
