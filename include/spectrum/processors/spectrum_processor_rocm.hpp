#pragma once

// ============================================================================
// SpectrumProcessorROCm — главный ROCm-фасад модуля spectrum (Layer 6 Ref03)
//
// ЧТО:    Реализация ISpectrumProcessor: координирует весь pipeline спектральной
//         обработки на ROCm — pad → hipFFT → post-kernel → readback. Содержит
//         3 Layer-5 Op'а (SpectrumPadOp, ComputeMagnitudesOp, SpectrumPostOp)
//         и AllMaximaPipelineROCm для полного FindAllMaxima сценария.
//         Поддерживает и Process (top-1/top-2 пиков), и AllMaxima (все пики).
//
// ЗАЧЕМ:  Это публичный API модуля spectrum — Python-биндинги и фасад
//         SpectrumMaximaFinder работают через интерфейс ISpectrumProcessor*,
//         выбирая SpectrumProcessorROCm через Factory. Один класс координирует
//         все стадии: фасад потребителю (Python/тесты), оркестратор внутри
//         (Ref03 Layer 6, держит Op'ы и pipeline).
//
// ПОЧЕМУ: - Layer 6 Ref03 (Facade) — НЕ делает kernel-launch'и сам, делегирует
//           Op'ам через unique_ptr/value-членов (pad_op_, mag_op_, post_op_).
//           Pipeline для AllMaxima — отдельный класс через unique_ptr.
//         - 2 hipFFT plan'а (plan_ + allmax_plan_) — Process использует pre-
//           callback path для скорости, AllMaxima требует «raw» FFT с
//           отдельным pad-kernel'ом. Разные plan'ы избегают рекомпиляции.
//         - Buffer-pool (input_buffer_, fft_input_, fft_output_, ...) с lazy
//           аллокацией и переиспользованием. ReallocateBuffersForBatch
//           меняет размер только когда batch_count изменился (избегает
//           hipFree+hipMalloc на каждый Process).
//         - Move запрещён (=delete) — owns hipFFT plans, kernel modules,
//           несколько device buffer'ов; копирование = chaos с lifetime.
//         - GpuContext ctx_ (Ref03 Layer 1) — единая точка для kernel
//           compile/cache; все Op'ы используют один ctx через Initialize().
//         - Магнитуды (magnitudes_buffer_) — лениво аллоцируется через
//           EnsureMagnitudesBuffer, для AllMaxima пути или для median-стратегии.
//         - Прямые void* буферы (а не BufferSet) — historical, миграция на
//           BufferSet — TODO. Не блокер, фасад работает корректно.
//         - GetProfilingData — агрегирует метрики ProfilingFacade per-stage.
//         - prof_events перегрузки методов (с/без аргумента) — production
//           путь без overhead'а сборки событий, benchmark-путь — с явным
//           prof_events*.
//
// Использование:
//   auto proc = std::make_unique<SpectrumProcessorROCm>(rocm_backend);
//   SpectrumParams params{.antenna_count=1024, .n_point=1100,
//                          .sample_rate=10e6f, .peak_mode=PeakSearchMode::ONE_PEAK};
//   proc->Initialize(params);
//   auto results = proc->ProcessFromCPU(iq_data);
//   // или:
//   auto all_max = proc->FindAllMaximaFromCPU(iq_data, OutputDestination::CPU, 1, 0);
//
// История:
//   - Создан:  2026-02-23 (ROCm-реализация ISpectrumProcessor, Strategy)
//   - Изменён: 2026-04-15 (миграция в DSP-GPU/spectrum, Ref03 Op'ы как Layer 5)
// ============================================================================

#if ENABLE_ROCM

#include <spectrum/interface/i_spectrum_processor.hpp>
#include <spectrum/interface/spectrum_maxima_types.h>
#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>
#include <spectrum/pipelines/all_maxima_pipeline_rocm.hpp>
#include <core/services/profiling_types.hpp>

// Op classes (Layer 5)
#include <spectrum/operations/spectrum_pad_op.hpp>
#include <spectrum/operations/compute_magnitudes_op.hpp>
#include <spectrum/operations/spectrum_post_op.hpp>

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

#include <complex>
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>

namespace antenna_fft {

/// ROCm profiling events: (name, ROCmProfilingData) pairs collected during processing
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

/**
 * @class SpectrumProcessorROCm
 * @brief Layer 6 Ref03 фасад: hipFFT + post-kernel + AllMaxima pipeline на ROCm.
 *
 * @note Move/copy запрещены — owns hipFFT plans + kernel modules + GPU buffers.
 * @note Требует #if ENABLE_ROCM. На non-ROCm сборках — stub с runtime_error.
 * @note Lifecycle: ctor(backend) → Initialize(params) → Process*/FindAllMaxima* → dtor.
 * @note Не thread-safe. Один экземпляр = один владелец GPU-ресурсов.
 * @see ISpectrumProcessor — интерфейс (Strategy + Bridge)
 * @see SpectrumProcessorFactory — создание по BackendType
 * @see AllMaximaPipelineROCm — отдельный pipeline для FindAllMaxima
 * @ingroup grp_fft_func
 */
class SpectrumProcessorROCm : public ISpectrumProcessor {
public:
    explicit SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend);
    ~SpectrumProcessorROCm() override;

    SpectrumProcessorROCm(const SpectrumProcessorROCm&) = delete;
    SpectrumProcessorROCm& operator=(const SpectrumProcessorROCm&) = delete;

    // =========================================================================
    // ISpectrumProcessor interface
    // =========================================================================

    void Initialize(const SpectrumParams& params) override;
    bool IsInitialized() const override { return initialized_; }

    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) override {
        return ProcessFromCPU(data, nullptr);
    }
    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data,
        ROCmProfEvents* prof_events);

    std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) override;

    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) override {
        return ProcessBatch(batch_data, start_antenna, batch_antenna_count, nullptr);
    }
    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count,
        ROCmProfEvents* prof_events);

    std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) override;

    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override {
        return FindAllMaximaFromCPU(data, dest, search_start, search_end, nullptr);
    }
    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end,
        ROCmProfEvents* prof_events);

    AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) override;

    DriverType GetDriverType() const override { return DriverType::ROCm; }
    ProfilingData GetProfilingData() const override;
    void ReallocateBuffersForBatch(size_t batch_antenna_count) override;
    size_t CalculateBytesPerAntenna() const override;
    void CompilePostKernel() override;

    const SpectrumParams& GetParams() const { return params_; }

private:
    // =========================================================================
    // Internal methods
    // =========================================================================

    void CalculateFFTSize();
    static uint32_t NextPowerOf2(uint32_t n);

    /// Allocate GPU buffers for current_batch_size_ beams
    void AllocateBuffers();

    /// Create hipFFT plan for batch processing
    void CreateFFTPlan(size_t batch_count);

    /// Compile HIP kernels via hiprtc (pad + compute_magnitudes + post_kernel)
    void CompileKernels();

    /// Upload CPU data → input_buffer_ (H2D)
    void UploadData(const std::complex<float>* data, size_t element_count);

    /// Copy GPU data → input_buffer_ (D2D)
    void CopyGpuData(void* src, size_t src_offset_bytes, size_t element_count);

    /// Execute pad kernel: input_buffer_ → fft_input_ (zero-padding n_point → nFFT)
    void ExecutePadKernel(size_t beam_count, size_t beam_offset = 0);

    /// Execute hipFFT: fft_input_ → fft_output_
    void ExecuteFFT();

    /// Execute post-processing kernel (ONE_PEAK or TWO_PEAKS) → maxima_output_
    void ExecutePostKernel(size_t beam_count, size_t beam_offset = 0);

    /// Execute compute_magnitudes kernel: fft_output_ → magnitudes_buffer_
    void ExecuteComputeMagnitudes(size_t total_elements);

    /// Read results from GPU → SpectrumResult vector
    std::vector<SpectrumResult> ReadResults(size_t beam_count, size_t beam_offset = 0);

    /// Release all GPU resources
    void ReleaseResources();

    /// Ensure magnitudes buffer is large enough
    void EnsureMagnitudesBuffer(size_t total_elements);

    /// Create hipFFT plan for AllMaxima (no pre-callback, separate pad)
    void CreateAllMaximaFFTPlan(size_t batch_count);

    /// Release AllMaxima-specific resources
    void ReleaseAllMaximaResources();

    // =========================================================================
    // State
    // =========================================================================

    SpectrumParams params_;
    bool initialized_ = false;
    drv_gpu_lib::IBackend* backend_ = nullptr;

    // Ref03: GpuContext for kernel compilation (replaces manual hiprtc + cache)
    drv_gpu_lib::GpuContext ctx_;
    bool compiled_ = false;

    // Op instances (Layer 5) — initialized in CompileKernels()
    SpectrumPadOp         pad_op_;
    ComputeMagnitudesOp   mag_op_;
    SpectrumPostOp        post_op_;

    // hipFFT (Process mode)
    hipfftHandle plan_ = 0;
    bool plan_created_ = false;
    size_t plan_batch_size_ = 0;

    // hipFFT for AllMaxima (separate plan)
    hipfftHandle allmax_plan_ = 0;
    bool allmax_plan_created_ = false;
    size_t allmax_plan_batch_size_ = 0;

    // GPU buffers (raw — migrate to BufferSet in future)
    void* input_buffer_ = nullptr;
    void* fft_input_ = nullptr;
    void* fft_output_ = nullptr;
    void* maxima_output_ = nullptr;
    void* magnitudes_buffer_ = nullptr;

    // =========================================================================
    // Sizes and batch tracking
    // =========================================================================

    size_t current_batch_size_ = 0;      ///< Currently allocated buffer size (beams)
    size_t actual_batch_size_ = 0;       ///< Actual beams in current batch
    size_t magnitudes_buffer_size_ = 0;  ///< Current magnitudes buffer size (elements)

    static constexpr size_t LOCAL_SIZE = 256;

    // =========================================================================
    // AllMaxima pipeline
    // =========================================================================

    std::unique_ptr<AllMaximaPipelineROCm> pipeline_;
};

}  // namespace antenna_fft

#else  // !ENABLE_ROCM

// ============================================================================
// Stub for non-ROCm builds (Windows)
// ============================================================================

#include <spectrum/interface/i_spectrum_processor.hpp>
#include <spectrum/interface/spectrum_maxima_types.h>
#include <core/interface/i_backend.hpp>

namespace antenna_fft {

class SpectrumProcessorROCm : public ISpectrumProcessor {
public:
    explicit SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend);
    ~SpectrumProcessorROCm() override = default;

    void Initialize(const SpectrumParams& params) override;
    bool IsInitialized() const override { return false; }

    std::vector<SpectrumResult> ProcessFromCPU(
        const std::vector<std::complex<float>>& data) override;

    std::vector<SpectrumResult> ProcessFromGPU(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes = 0) override;

    std::vector<SpectrumResult> ProcessBatch(
        const std::vector<std::complex<float>>& batch_data,
        size_t start_antenna,
        size_t batch_antenna_count) override;

    std::vector<SpectrumResult> ProcessBatchFromGPU(
        void* gpu_data, size_t src_offset_bytes,
        size_t start_antenna, size_t batch_antenna_count) override;

    AllMaximaResult FindAllMaximaFromCPU(
        const std::vector<std::complex<float>>& data,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaximaFromGPUPipeline(
        void* gpu_data, size_t antenna_count, size_t n_point,
        size_t gpu_memory_bytes,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult AllMaximaFromCPU(
        const std::vector<std::complex<float>>& fft_data,
        uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest, uint32_t search_start, uint32_t search_end) override;

    AllMaximaResult FindAllMaxima(
        void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 0, uint32_t search_end = 0) override;

    DriverType GetDriverType() const override { return DriverType::ROCm; }
    ProfilingData GetProfilingData() const override;
    void ReallocateBuffersForBatch(size_t batch_antenna_count) override;
    size_t CalculateBytesPerAntenna() const override;
    void CompilePostKernel() override;

private:
    drv_gpu_lib::IBackend* backend_ = nullptr;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
