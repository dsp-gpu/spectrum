/**
 * @file spectrum_processor_rocm.cpp
 * @brief ROCm/HIP implementation of ISpectrumProcessor (hipFFT + hiprtc kernels)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CONTENTS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PART 1: Constructor, Destructor
 * PART 2: Initialize
 * PART 3: Process (ProcessFromCPU, ProcessFromGPU, ProcessBatch, ProcessBatchFromGPU)
 * PART 4: FindAllMaxima pipeline (FindAllMaximaFromCPU, FromGPUPipeline, AllMaximaFromCPU, FindAllMaxima)
 * PART 5: GPU Resources (AllocateBuffers, CreateFFTPlan, CompileKernels, CompilePostKernel)
 * PART 6: GPU Operations (UploadData, CopyGpuData, ExecutePadKernel, ExecuteFFT, ExecutePostKernel, etc.)
 * PART 7: Utilities (CalculateFFTSize, ReallocateBuffersForBatch, ReleaseResources, etc.)
 *
 * Key differences from OpenCL version:
 * - No pre-callback → separate pad_data kernel before hipFFT
 * - No post-callback → separate compute_magnitudes kernel after hipFFT
 * - Stream-ordered execution (no explicit events)
 * - void* device pointers instead of cl_mem
 * - hiprtc for runtime kernel compilation
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <spectrum/processors/spectrum_processor_rocm.hpp>
#include "kernels/fft_kernel_sources_rocm.hpp"
#include "kernels/all_maxima_kernel_sources_rocm.hpp"
#include <spectrum/utils/rocm_profiling_helpers.hpp>
#include <core/services/console_output.hpp>
#include <core/services/gpu_profiler.hpp>
#include <core/services/batch_manager.hpp>
#include <core/interface/i_backend.hpp>
#include <core/services/profiling_types.hpp>

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

using fft_func_utils::MakeROCmDataFromEvents;
using fft_func_utils::MakeROCmDataFromClock;

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// PART 1: Constructor, Destructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Конструктор — получить HIP stream из backend и создать pipeline
 *
 * ctx_.stream() берётся из backend_->GetNativeQueue() — backend владеет очередью,
 * мы только используем её (не закрываем в деструкторе).
 *
 * pipeline_ = AllMaximaPipelineROCm(ctx_.stream(), backend_) создаётся сразу —
 * lazy CompileKernels внутри pipeline произойдёт при первом вызове FindAllMaxima.
 *
 * Ключевое отличие от OpenCL: нет cl_context/cl_queue, только hipStream_t.
 *
 * @param backend Инициализированный IBackend с ROCm backend (не nullptr, не uninit)
 * @throws std::invalid_argument если backend == nullptr
 * @throws std::runtime_error если backend не инициализирован или нет HIP stream
 */
static const std::vector<std::string> kSpectrumKernelNames = {
    "pad_data",
    "compute_magnitudes",
    "post_kernel_one_peak",
    "post_kernel_two_peaks"
};

SpectrumProcessorROCm::SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend)
    , ctx_(backend, "SpectrumMaxima", "modules/fft_func/kernels") {

    hipStream_t stream = ctx_.stream();
    pipeline_ = std::make_unique<AllMaximaPipelineROCm>(stream, backend_);
}

/**
 * @brief Деструктор — освободить все GPU ресурсы
 *
 * Порядок освобождения в ReleaseResources() критичен:
 * pipeline_.reset() → allmax_plan → plan_ → буферы → hiprtc module.
 * ctx_.stream() НЕ закрывается — он принадлежит backend_.
 */
SpectrumProcessorROCm::~SpectrumProcessorROCm() {
    ReleaseResources();
}

// ════════════════════════════════════════════════════════════════════════════
// PART 2: Initialize
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Инициализировать процессор для заданных параметров
 *
 * Вычисляет nFFT, определяет нужен ли batch режим, выделяет GPU буферы
 * и компилирует kernels (hiprtc JIT).
 *
 * Два пути:
 *  - Batch mode: ReallocateBuffersForBatch(max_batch_size) — буферы на max batch
 *  - Normal mode: AllocateBuffers() → CreateFFTPlan() → CompileKernels()
 *
 * Повторный вызов (initialized_==true) → только обновляет params_, пропускает выделение.
 *
 * @param params Параметры обработки (antenna_count, n_point, repeat_count, etc.)
 */
void SpectrumProcessorROCm::Initialize(const SpectrumParams& params) {
    params_ = params;
    if (initialized_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima[ROCm]", "Initializing...");

    CalculateFFTSize();

    con.Print(0, "SpectrumMaxima[ROCm]",
        "antenna_count=" + std::to_string(params_.antenna_count) +
        " n_point=" + std::to_string(params_.n_point) +
        " repeat_count=" + std::to_string(params_.repeat_count) +
        " nFFT=" + std::to_string(params_.nFFT) +
        " search_range=" + std::to_string(params_.search_range) +
        " sample_rate=" + std::to_string(static_cast<int>(params_.sample_rate)) + " Hz");

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    bool need_batch = !drv_gpu_lib::BatchManager::AllItemsFit(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    if (need_batch) {
        size_t max_batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
            backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

        con.Print(0, "SpectrumMaxima[ROCm]", "[Batch mode] max batch=" + std::to_string(max_batch_size));
        ReallocateBuffersForBatch(max_batch_size);
        CompilePostKernel();
    } else {
        AllocateBuffers();
        CreateFFTPlan(params_.antenna_count);
        CompileKernels();
        CompilePostKernel();

        current_batch_size_ = params_.antenna_count;
        actual_batch_size_ = params_.antenna_count;
    }

    initialized_ = true;
    con.Print(0, "SpectrumMaxima[ROCm]", "Initialization complete");
}

// ════════════════════════════════════════════════════════════════════════════
// PART 3: Process (ONE_PEAK / TWO_PEAKS mode)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Обработать один batch CPU-данных: Upload → Pad → FFT → Post → Sync → ReadResults
 *
 * ROCm версия использует stream-ordered execution без явных cl_event:
 * - Порядок операций гарантирован одним hipStream_t (ctx_.stream())
 * - Синхронизация: hipStreamSynchronize(ctx_.stream()) перед ReadResults
 *
 * Профилирование через hipEvent (если prof_events != nullptr):
 *   5 операций: Upload(H2D) | PadKernel | FFT | PostKernel | Download(D2H wall-clock)
 *   Download измеряется wall-clock (MakeROCmDataFromClock) — hipMemcpyDtoH синхронный.
 *
 * @param input_data          Весь массив входных данных [total_antenna_count × n_point]
 * @param start_antenna       Индекс первой антенны в input_data для этого batch
 * @param batch_antenna_count Количество антенн в batch
 * @param prof_events         Список ROCmProfilingData для профилирования (nullptr = не собираем)
 * @return vector<SpectrumResult>[batch_antenna_count] с корректными antenna_id (start_antenna + i)
 */
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(
    const std::vector<std::complex<float>>& input_data,
    size_t start_antenna,
    size_t batch_antenna_count,
    ROCmProfEvents* prof_events)
{
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    size_t offset = start_antenna * params_.n_point;
    size_t count  = batch_antenna_count * params_.n_point;

    // ── Upload (H2D) ─────────────────────────────────────────────────────────
    hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_up_s); hipEventCreate(&ev_up_e);
        hipEventRecord(ev_up_s, ctx_.stream());
    }
    UploadData(input_data.data() + offset, count);
    if (prof_events) hipEventRecord(ev_up_e, ctx_.stream());

    // ── PadKernel ────────────────────────────────────────────────────────────
    hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
        hipEventRecord(ev_pad_s, ctx_.stream());
    }
    ExecutePadKernel(batch_antenna_count);
    if (prof_events) hipEventRecord(ev_pad_e, ctx_.stream());

    // ── FFT ──────────────────────────────────────────────────────────────────
    hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
        hipEventRecord(ev_fft_s, ctx_.stream());
    }
    ExecuteFFT();
    if (prof_events) hipEventRecord(ev_fft_e, ctx_.stream());

    // ── PostKernel ───────────────────────────────────────────────────────────
    hipEvent_t ev_post_s = nullptr, ev_post_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_post_s); hipEventCreate(&ev_post_e);
        hipEventRecord(ev_post_s, ctx_.stream());
    }
    ExecutePostKernel(batch_antenna_count);
    if (prof_events) hipEventRecord(ev_post_e, ctx_.stream());

    hipStreamSynchronize(ctx_.stream());

    // ── Download (D2H sync — wall-clock) ─────────────────────────────────────
    auto t_dl_s = std::chrono::high_resolution_clock::now();
    auto results = ReadResults(batch_antenna_count);
    auto t_dl_e = std::chrono::high_resolution_clock::now();

    // ── Собрать события ──────────────────────────────────────────────────────
    if (prof_events) {
        prof_events->push_back({"Upload",     MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
        prof_events->push_back({"PadKernel",  MakeROCmDataFromEvents(ev_pad_s,  ev_pad_e,  0, "pad_kernel")});
        prof_events->push_back({"FFT",        MakeROCmDataFromEvents(ev_fft_s,  ev_fft_e,  0, "hipfftExecC2C")});
        prof_events->push_back({"PostKernel", MakeROCmDataFromEvents(ev_post_s, ev_post_e, 0, "post_kernel")});
        prof_events->push_back({"Download",   MakeROCmDataFromClock(t_dl_s,     t_dl_e,    1, "D2H")});
    }

    for (auto& result : results) {
        result.antenna_id += static_cast<uint32_t>(start_antenna);
    }

    return results;
}

/**
 * @brief Обработать все антенны с CPU-данных: разбить на batch'и если нужно
 *
 * Если все данные помещаются в VRAM (BatchManager::AllItemsFit) → один ProcessBatch.
 * Иначе → loop по batch'ам через CreateBatches (с overlap=3, logProgress=true).
 *
 * @param data        Плоский массив [antenna_count × n_point] complex<float>
 * @param prof_events Список ROCmProfilingData (nullptr = не профилируем)
 * @return vector<SpectrumResult>[antenna_count]
 * @throws std::runtime_error если не инициализирован или размер данных не совпадает
 */
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    ROCmProfEvents* prof_events)
{
    if (!initialized_) {
        throw std::runtime_error("SpectrumProcessorROCm::ProcessFromCPU: not initialized");
    }

    size_t expected_size = params_.antenna_count * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "SpectrumProcessorROCm::ProcessFromCPU: input size mismatch. "
            "Expected " + std::to_string(expected_size) +
            ", got " + std::to_string(data.size()));
    }

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                bytes_per_antenna, params_.memory_limit)) {
        return ProcessBatch(data, 0, params_.antenna_count, prof_events);
    }

    // Batch processing
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    size_t batch_size = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, params_.antenna_count, bytes_per_antenna, params_.memory_limit);

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(
        params_.antenna_count, batch_size, 3, true);

    con.Print(0, "SpectrumMaxima[ROCm]", "Batch Processing: " +
        std::to_string(batches.size()) + " batches");

    std::vector<SpectrumResult> all_results;
    all_results.reserve(params_.antenna_count);

    for (const auto& batch : batches) {
        auto batch_results = ProcessBatch(data, batch.start, batch.count, prof_events);
        all_results.insert(all_results.end(), batch_results.begin(), batch_results.end());
    }

    return all_results;
}

/**
 * @brief Обработать GPU-данные (D2D): CopyGpuData → Pad → FFT → Post → Sync → ReadResults
 *
 * Данные уже на GPU (void* device pointer) → D2D copy через hipMemcpyDtoDAsync.
 * external_memory учитывается в BatchManager::CalculateOptimalBatchSize как уже занятая VRAM.
 *
 * @param gpu_data         void* device pointer на входные данные [antenna_count × n_point]
 * @param antenna_count    Количество антенн
 * @param n_point          Точек на антенну в исходных данных (до padding)
 * @param gpu_memory_bytes Реальный размер gpu_data в байтах (0 = вычисляем сами)
 * @return vector<SpectrumResult>[antenna_count]
 * @throws std::invalid_argument если gpu_data == nullptr
 */
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromGPU(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes)
{
    if (!gpu_data) {
        throw std::invalid_argument("SpectrumProcessorROCm::ProcessFromGPU: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "SpectrumMaxima[ROCm]", "ProcessFromGPU: " +
        std::to_string(antenna_count) + " antennas, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) {
        CalculateFFTSize();
    }
    if (!compiled_) {
        CompileKernels();
    }
    CompilePostKernel();

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    bool need_batch = (optimal_batch < antenna_count);

    if (!need_batch) {
        ReallocateBuffersForBatch(antenna_count);
        actual_batch_size_ = antenna_count;

        // D2D copy → Pad → FFT → PostKernel → Read
        CopyGpuData(gpu_data, 0, antenna_count * n_point);
        ExecutePadKernel(antenna_count);
        ExecuteFFT();
        ExecutePostKernel(antenna_count);
        hipStreamSynchronize(ctx_.stream());

        return ReadResults(antenna_count);
    }

    auto batches = drv_gpu_lib::BatchManager::CreateBatches(antenna_count, optimal_batch);
    ReallocateBuffersForBatch(optimal_batch);

    std::vector<SpectrumResult> all_results;
    all_results.reserve(antenna_count);

    for (const auto& batch : batches) {
        auto batch_results = ProcessBatchFromGPU(gpu_data,
            batch.start * n_point * sizeof(std::complex<float>),
            batch.start, batch.count);
        for (auto& r : batch_results) {
            all_results.push_back(std::move(r));
        }
    }

    return all_results;
}

/**
 * @brief Обработать один batch из внешнего GPU буфера (вызывается из ProcessFromGPU)
 *
 * CopyGpuData с src_offset → Pad → FFT → PostKernel → Sync → ReadResults.
 * antenna_id в результатах сдвигается на start_antenna.
 *
 * @param gpu_data             Внешний device pointer с данными
 * @param src_offset_bytes     Смещение в gpu_data для этого batch (в байтах)
 * @param start_antenna        Абсолютный индекс первой антенны batch (для antenna_id)
 * @param batch_antenna_count  Количество антенн в batch
 * @return vector<SpectrumResult>[batch_antenna_count] с корректными antenna_id
 */
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatchFromGPU(
    void* gpu_data, size_t src_offset_bytes,
    size_t start_antenna, size_t batch_antenna_count)
{
    if (batch_antenna_count > current_batch_size_ || !plan_created_) {
        ReallocateBuffersForBatch(batch_antenna_count);
    }
    actual_batch_size_ = batch_antenna_count;

    CopyGpuData(gpu_data, src_offset_bytes, batch_antenna_count * params_.n_point);
    ExecutePadKernel(batch_antenna_count);
    ExecuteFFT();
    ExecutePostKernel(batch_antenna_count);
    hipStreamSynchronize(ctx_.stream());

    auto results = ReadResults(batch_antenna_count);

    for (size_t i = 0; i < results.size(); ++i) {
        results[i].antenna_id = static_cast<uint32_t>(start_antenna + i);
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 4: FindAllMaxima pipeline
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Полный AllMaxima pipeline из CPU-данных (raw signal → все пики)
 *
 * Отличие от ProcessFromCPU: вместо PostKernel (ONE/TWO_PEAKS) запускает
 * полный AllMaxima pipeline: ComputeMagnitudes → pipeline_->Execute() (Detect+Scan+Compact).
 *
 * ⚠️ Batch НЕ поддерживается — если данные не помещаются → throw runtime_error.
 *    Причина: stream compaction с prefix sum требует полного спектра всех лучей.
 *
 * Профилирование (5 операций): Upload | PadKernel | FFT | ComputeMagnitudes | Pipeline
 *
 * @param data        Плоский массив [antenna_count × n_point] complex<float>
 * @param dest        Куда писать результаты (CPU/GPU/ALL)
 * @param search_start Первый индекс FFT для поиска (0 → авто=1, пропуск DC)
 * @param search_end   Последний индекс FFT (0 → авто=nFFT/2)
 * @param prof_events  ROCmProfEvents для профилирования (nullptr = не собираем)
 * @return AllMaximaResult; при dest=GPU/ALL — caller обязан освободить gpu_maxima/gpu_counts!
 * @throws std::runtime_error если не инициализирован или данные не помещаются
 */
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(
    const std::vector<std::complex<float>>& data,
    OutputDestination dest, uint32_t search_start, uint32_t search_end,
    ROCmProfEvents* prof_events)
{
    if (!initialized_) {
        throw std::runtime_error("SpectrumProcessorROCm::FindAllMaximaFromCPU: not initialized");
    }

    size_t expected_size = static_cast<size_t>(params_.antenna_count) * params_.n_point;
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "FindAllMaximaFromCPU: input size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaximaFromCPU: " +
        std::to_string(params_.antenna_count) + " beams, nFFT=" +
        std::to_string(params_.nFFT));

    size_t bytes_per_antenna = CalculateBytesPerAntenna();

    if (!drv_gpu_lib::BatchManager::AllItemsFit(backend_, params_.antenna_count,
                                                  bytes_per_antenna, params_.memory_limit)) {
        throw std::runtime_error(
            "FindAllMaximaFromCPU: data doesn't fit in GPU memory");
    }

    ReallocateBuffersForBatch(params_.antenna_count);
    actual_batch_size_ = params_.antenna_count;
    CreateAllMaximaFFTPlan(params_.antenna_count);

    if (!compiled_) CompileKernels();

    size_t total_elements = static_cast<size_t>(params_.antenna_count) * params_.nFFT;

    // ── Upload (H2D) ─────────────────────────────────────────────────────────
    hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_up_s); hipEventCreate(&ev_up_e);
        hipEventRecord(ev_up_s, ctx_.stream());
    }
    UploadData(data.data(), data.size());
    if (prof_events) hipEventRecord(ev_up_e, ctx_.stream());

    // ── PadKernel ────────────────────────────────────────────────────────────
    hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
        hipEventRecord(ev_pad_s, ctx_.stream());
    }
    ExecutePadKernel(params_.antenna_count);
    if (prof_events) hipEventRecord(ev_pad_e, ctx_.stream());

    // ── FFT ──────────────────────────────────────────────────────────────────
    hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
        hipEventRecord(ev_fft_s, ctx_.stream());
    }
    ExecuteFFT();
    if (prof_events) hipEventRecord(ev_fft_e, ctx_.stream());

    // ── ComputeMagnitudes ────────────────────────────────────────────────────
    hipEvent_t ev_mag_s = nullptr, ev_mag_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_mag_s); hipEventCreate(&ev_mag_e);
        hipEventRecord(ev_mag_s, ctx_.stream());
    }
    ExecuteComputeMagnitudes(total_elements);
    if (prof_events) hipEventRecord(ev_mag_e, ctx_.stream());

    // ── Pipeline (Detect+Scan+Compact) ───────────────────────────────────────
    hipEvent_t ev_pipe_s = nullptr, ev_pipe_e = nullptr;
    if (prof_events) {
        hipEventCreate(&ev_pipe_s); hipEventCreate(&ev_pipe_e);
        hipEventRecord(ev_pipe_s, ctx_.stream());
    }
    hipStreamSynchronize(ctx_.stream());

    AllMaximaResult result = pipeline_->Execute(
        magnitudes_buffer_, fft_output_,
        params_.antenna_count, params_.nFFT, params_.sample_rate,
        dest, search_start, search_end);

    if (prof_events) hipEventRecord(ev_pipe_e, ctx_.stream());

    // ── Собрать события ──────────────────────────────────────────────────────
    if (prof_events) {
        hipStreamSynchronize(ctx_.stream());
        prof_events->push_back({"Upload",            MakeROCmDataFromEvents(ev_up_s,   ev_up_e,   1, "H2D")});
        prof_events->push_back({"PadKernel",         MakeROCmDataFromEvents(ev_pad_s,  ev_pad_e,  0, "pad_kernel")});
        prof_events->push_back({"FFT",               MakeROCmDataFromEvents(ev_fft_s,  ev_fft_e,  0, "hipfftExecC2C")});
        prof_events->push_back({"ComputeMagnitudes", MakeROCmDataFromEvents(ev_mag_s,  ev_mag_e,  0, "compute_mag")});
        prof_events->push_back({"Pipeline",          MakeROCmDataFromEvents(ev_pipe_s, ev_pipe_e, 0, "detect+scan+compact")});
    }

    return result;
}

/**
 * @brief AllMaxima pipeline из GPU-данных (D2D): CopyGpu → Pad → FFT → ComputeMag → pipeline
 *
 * Аналог FindAllMaximaFromCPU, но входные данные уже на GPU (void* device pointer).
 * D2D copy через CopyGpuData(src_offset=0) → далее всё то же что и в FromCPU.
 *
 * ⚠️ Batch НЕ поддерживается — если данные не помещаются → throw runtime_error.
 *
 * @param gpu_data         void* device pointer [antenna_count × n_point]
 * @param antenna_count    Число антенн
 * @param n_point          Точек на антенну до padding
 * @param gpu_memory_bytes Размер gpu_data (0 = вычисляем)
 * @param dest             OutputDestination (CPU/GPU/ALL)
 * @param search_start     Первый FFT-бин (0 → авто=1)
 * @param search_end       Последний FFT-бин (0 → авто=nFFT/2)
 * @return AllMaximaResult; при dest=GPU/ALL — caller обязан освободить gpu_maxima/gpu_counts!
 */
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromGPUPipeline(
    void* gpu_data, size_t antenna_count, size_t n_point,
    size_t gpu_memory_bytes,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!gpu_data) {
        throw std::invalid_argument("FindAllMaximaFromGPUPipeline: gpu_data cannot be null");
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaximaFromGPUPipeline: " +
        std::to_string(antenna_count) + " beams, n_point=" + std::to_string(n_point));

    if (params_.nFFT == 0) CalculateFFTSize();
    if (!compiled_) CompileKernels();

    size_t bytes_per_antenna = CalculateBytesPerAntenna();
    size_t external_memory = (gpu_memory_bytes > 0)
        ? gpu_memory_bytes
        : antenna_count * n_point * sizeof(std::complex<float>);

    size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
        backend_, antenna_count, bytes_per_antenna,
        params_.memory_limit, external_memory);

    if (optimal_batch < antenna_count) {
        throw std::runtime_error("FindAllMaximaFromGPUPipeline: data doesn't fit in GPU memory");
    }

    ReallocateBuffersForBatch(antenna_count);
    actual_batch_size_ = antenna_count;
    CreateAllMaximaFFTPlan(antenna_count);

    // D2D copy → Pad → FFT → ComputeMagnitudes → pipeline
    CopyGpuData(gpu_data, 0, antenna_count * n_point);
    ExecutePadKernel(antenna_count);
    ExecuteFFT();

    size_t total_elements = antenna_count * params_.nFFT;
    ExecuteComputeMagnitudes(total_elements);
    hipStreamSynchronize(ctx_.stream());

    return pipeline_->Execute(
        magnitudes_buffer_, fft_output_,
        static_cast<uint32_t>(antenna_count), params_.nFFT, params_.sample_rate,
        dest, search_start, search_end);
}

/**
 * @brief AllMaxima из готовых FFT данных с CPU (без FFT шага)
 *
 * Вход — уже готовые FFT спектры (complex<float>), не сырые данные.
 * Создаёт временный GPU буфер через hipMalloc → hipMemcpyHtoDAsync → FindAllMaxima → hipFree.
 *
 * Ownership временного gpu_fft: только внутри функции, освобождается до return.
 * Ownership AllMaximaResult.gpu_maxima/gpu_counts при dest=GPU/ALL: caller обязан освободить!
 *
 * @param fft_data    Плоский массив [beam_count × nFFT] complex<float> (FFT спектры)
 * @param beam_count  Количество лучей/антенн
 * @param nFFT        Длина FFT на луч
 * @param sample_rate Частота дискретизации (для freq в MaxValue)
 * @param dest        OutputDestination (CPU/GPU/ALL)
 * @param search_start Первый FFT-бин (0 → авто=1)
 * @param search_end   Последний FFT-бин (0 → авто=nFFT/2)
 * @return AllMaximaResult
 * @throws std::invalid_argument если размер не совпадает
 * @throws std::runtime_error если hipMalloc/hipMemcpy неудача
 */
AllMaximaResult SpectrumProcessorROCm::AllMaximaFromCPU(
    const std::vector<std::complex<float>>& fft_data,
    uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    size_t expected_size = static_cast<size_t>(beam_count) * nFFT;
    if (fft_data.size() != expected_size) {
        throw std::invalid_argument(
            "AllMaximaFromCPU: size mismatch. Expected " +
            std::to_string(expected_size) + ", got " + std::to_string(fft_data.size()));
    }

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "AllMaximaFromCPU: " +
        std::to_string(beam_count) + " beams, nFFT=" + std::to_string(nFFT));

    if (!compiled_) CompileKernels();

    // Upload FFT data to GPU
    size_t data_bytes = expected_size * sizeof(std::complex<float>);
    void* gpu_fft = nullptr;

    hipError_t err = hipMalloc(&gpu_fft, data_bytes);
    if (err != hipSuccess)
        throw std::runtime_error("AllMaximaFromCPU: hipMalloc failed");

    err = hipMemcpyHtoDAsync(gpu_fft, const_cast<std::complex<float>*>(fft_data.data()),
                              data_bytes, ctx_.stream());
    if (err != hipSuccess) {
        (void)hipFree(gpu_fft);
        throw std::runtime_error("AllMaximaFromCPU: hipMemcpyHtoDAsync failed");
    }

    hipStreamSynchronize(ctx_.stream());

    AllMaximaResult result = FindAllMaxima(gpu_fft, beam_count, nFFT,
                                            sample_rate, dest, search_start, search_end);

    (void)hipFree(gpu_fft);
    return result;
}

/**
 * @brief Core AllMaxima: ComputeMagnitudes → pipeline_->Execute() (Strategy pattern!)
 *
 * Принимает уже готовые FFT данные на GPU (void* device pointer).
 * Вычисляет magnitudes через явный hipModuleLaunchKernel с fft_data (не fft_output_!).
 *
 * ⚠️ Внимание на двойной вызов: ExecuteComputeMagnitudes(total_elements) вверху функции
 * обрабатывает fft_output_, но потом перезаписывается явным kernel launch с fft_data.
 * Это нужно потому что fft_data может быть не тем же указателем что fft_output_.
 *
 * Нормализация search_start/search_end:
 *   search_start=0 → 1 (пропуск DC-компоненты)
 *   search_end=0   → nFFT/2 (только положительные частоты)
 *   search_end>=nFFT → nFFT-1
 *
 * @param fft_data    void* device pointer на FFT спектры [beam_count × nFFT] complex<float>
 * @param beam_count  Количество лучей
 * @param nFFT        Длина FFT на луч
 * @param sample_rate Частота дискретизации (Hz)
 * @param dest        OutputDestination (CPU/GPU/ALL)
 * @param search_start Первый FFT-бин (0 → авто=1)
 * @param search_end   Последний FFT-бин (0 → авто=nFFT/2)
 * @return AllMaximaResult; при dest=GPU/ALL — caller обязан освободить gpu_maxima/gpu_counts!
 */
AllMaximaResult SpectrumProcessorROCm::FindAllMaxima(
    void* fft_data, uint32_t beam_count, uint32_t nFFT, float sample_rate,
    OutputDestination dest, uint32_t search_start, uint32_t search_end)
{
    if (!fft_data)
        throw std::invalid_argument("FindAllMaxima: fft_data cannot be null");

    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    con.Print(0, "AllMaxima[ROCm]", "FindAllMaxima: beam_count=" + std::to_string(beam_count)
        + " nFFT=" + std::to_string(nFFT)
        + " range=[" + std::to_string(search_start) + "," + std::to_string(search_end) + ")"
        + " Fs=" + std::to_string(static_cast<int>(sample_rate)) + "Hz");

    if (!compiled_) CompileKernels();

    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;
    EnsureMagnitudesBuffer(total_elements);

    // Compute magnitudes: fft_data → magnitudes_buffer_
    ExecuteComputeMagnitudes(total_elements);
    hipStreamSynchronize(ctx_.stream());

    // We need to pass fft_data to compute_magnitudes, but the function uses fft_output_
    // Actually, we need a local magnitude compute since fft_data might not be fft_output_.
    // Let me fix: compute magnitudes from arbitrary fft_data.
    // We'll launch the kernel with fft_data pointer explicitly.
    {
        unsigned int total_size = static_cast<unsigned int>(total_elements);

        void* args[] = { &fft_data, &magnitudes_buffer_, &total_size };

        unsigned int grid_x = static_cast<unsigned int>((total_elements + 255) / 256);
        unsigned int block_x = 256;

        hipError_t hip_err = hipModuleLaunchKernel(
            ctx_.GetKernel("compute_magnitudes"),
            grid_x, 1, 1,
            block_x, 1, 1,
            0, ctx_.stream(),
            args, nullptr);
        if (hip_err != hipSuccess) {
            throw std::runtime_error("FindAllMaxima: compute_magnitudes launch failed: " +
                                      std::string(hipGetErrorString(hip_err)));
        }
        hipStreamSynchronize(ctx_.stream());
    }

    return pipeline_->Execute(
        magnitudes_buffer_, fft_data,
        beam_count, nFFT, sample_rate,
        dest, search_start, search_end);
}

// ════════════════════════════════════════════════════════════════════════════
// PART 5: GPU Resources
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Выделить 4 основных GPU буфера для params_.antenna_count антенн
 *
 * Буферы:
 *  1. input_buffer_  [antenna_count × n_point × sizeof(complex)] — сырые данные до padding
 *  2. fft_input_     [antenna_count × nFFT × sizeof(complex)]    — после pad_data kernel
 *  3. fft_output_    [antenna_count × nFFT × sizeof(complex)]    — результат hipfftExecC2C
 *  4. maxima_output_ [antenna_count × (4 или 8) × sizeof(MaxValue)] — результат post_kernel
 *     ONE_PEAK → 4 MaxValue/beam, TWO_PEAKS → 8 MaxValue/beam
 *
 * Вызывается из Initialize() когда все антенны помещаются в VRAM.
 * При batch режиме вместо этого вызывается ReallocateBuffersForBatch().
 *
 * @throws std::runtime_error если hipMalloc неудача для любого буфера
 */
void SpectrumProcessorROCm::AllocateBuffers() {
    hipError_t err;

    // 1. Input buffer (raw data, not padded)
    size_t input_size = params_.antenna_count * params_.n_point * sizeof(std::complex<float>);
    err = hipMalloc(&input_buffer_, input_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: input_buffer failed: " +
                                  std::string(hipGetErrorString(err)));

    // 2. FFT buffers (padded)
    size_t fft_size = params_.antenna_count * params_.nFFT * sizeof(std::complex<float>);

    err = hipMalloc(&fft_input_, fft_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: fft_input failed");

    err = hipMalloc(&fft_output_, fft_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: fft_output failed");

    // 3. Maxima output (ONE_PEAK=4, TWO_PEAKS=8 per beam)
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_size = params_.antenna_count * max_values_per_beam * sizeof(MaxValue);

    err = hipMalloc(&maxima_output_, maxima_size);
    if (err != hipSuccess)
        throw std::runtime_error("AllocateBuffers: maxima_output failed");
}

/**
 * @brief Создать hipFFT план для ProcessBatch (Process режим)
 *
 * hipfftPlan1d(nFFT, C2C, batch_count) → привязка к ctx_.stream() через hipfftSetStream.
 * Guard: повторный вызов с тем же batch_count — ничего не делает.
 * При изменении batch_count — старый план уничтожается.
 *
 * Отличие от CreateAllMaximaFFTPlan: это план для Process (ONE/TWO_PEAKS),
 * план для AllMaxima — отдельный allmax_plan_.
 *
 * @param batch_count Количество антенн для этого плана
 * @throws std::runtime_error если hipfftPlan1d или hipfftSetStream неудача
 */
void SpectrumProcessorROCm::CreateFFTPlan(size_t batch_count) {
    if (plan_created_ && plan_batch_size_ == batch_count) return;

    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
    }

    hipfftResult result = hipfftPlan1d(&plan_,
        static_cast<int>(params_.nFFT),
        HIPFFT_C2C,
        static_cast<int>(batch_count));
    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("CreateFFTPlan: hipfftPlan1d failed: " +
                                  std::to_string(static_cast<int>(result)));
    }

    result = hipfftSetStream(plan_, ctx_.stream());
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(plan_);
        throw std::runtime_error("CreateFFTPlan: hipfftSetStream failed");
    }

    plan_batch_size_ = batch_count;
    plan_created_ = true;
}

/**
 * @brief Создать hipFFT план для FindAllMaxima pipeline (отдельный от plan_)
 *
 * allmax_plan_ может использоваться параллельно с plan_ — разные режимы работы.
 * ExecuteFFT() выбирает активный план: allmax_plan_created_ ? allmax_plan_ : plan_.
 *
 * Также вызывает EnsureMagnitudesBuffer(batch_count × nFFT) — чтобы magnitudes_buffer_
 * был достаточного размера до запуска ComputeMagnitudes.
 *
 * @param batch_count Количество антенн для AllMaxima FFT плана
 * @throws std::runtime_error если hipfftPlan1d или hipfftSetStream неудача
 */
void SpectrumProcessorROCm::CreateAllMaximaFFTPlan(size_t batch_count) {
    if (allmax_plan_created_ && allmax_plan_batch_size_ == batch_count) return;

    if (allmax_plan_created_) {
        hipfftDestroy(allmax_plan_);
        allmax_plan_ = 0;
        allmax_plan_created_ = false;
    }

    size_t total_fft_elements = batch_count * params_.nFFT;
    EnsureMagnitudesBuffer(total_fft_elements);

    hipfftResult result = hipfftPlan1d(&allmax_plan_,
        static_cast<int>(params_.nFFT),
        HIPFFT_C2C,
        static_cast<int>(batch_count));
    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("CreateAllMaximaFFTPlan: hipfftPlan1d failed");
    }

    result = hipfftSetStream(allmax_plan_, ctx_.stream());
    if (result != HIPFFT_SUCCESS) {
        hipfftDestroy(allmax_plan_);
        throw std::runtime_error("CreateAllMaximaFFTPlan: hipfftSetStream failed");
    }

    allmax_plan_batch_size_ = batch_count;
    allmax_plan_created_ = true;
}

/**
 * @brief JIT-компилировать HIP kernels через hiprtc
 *
 * Компилирует GetSpectrumHIPKernelSource() с -O3 → hipModule → извлекает функции:
 *  - pad_data: zero-padding [n_point → nFFT] для всех лучей
 *  - compute_magnitudes: |z|² = re² + im² (complex→float)
 *
 * All 4 kernels compiled in one GpuContext::CompileModule() call.
 * post_kernel selection (ONE_PEAK/TWO_PEAKS) happens at Execute time via ctx_.GetKernel().
 *
 * При ошибке компиляции → print лог через ConsoleOutput (не PrintError) + throw.
 *
 * @throws std::runtime_error если hiprtcCreateProgram / Compile / ModuleLoadData неудача
 */
void SpectrumProcessorROCm::CompileKernels() {
    if (compiled_) return;
    ctx_.CompileModule(kernels::GetSpectrumHIPKernelSource(), kSpectrumKernelNames);
    pad_op_.Initialize(ctx_);
    mag_op_.Initialize(ctx_);
    post_op_.Initialize(ctx_);
    compiled_ = true;
}

// Legacy CompileKernels body removed (2026-03-22): was ~140 lines of manual hiprtc.

void SpectrumProcessorROCm::CompilePostKernel() {
    if (!compiled_) CompileKernels();
    // post_kernel извлекается через ctx_.GetKernel() при каждом Execute
    // (выбор ONE_PEAK / TWO_PEAKS по params_.peak_mode делается в ExecutePostKernel)
}

// ════════════════════════════════════════════════════════════════════════════
// PART 6: GPU Operations
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Асинхронный upload CPU→GPU (H2D) в input_buffer_
 *
 * hipMemcpyHtoDAsync в ctx_.stream() — не блокирует CPU до hipStreamSynchronize.
 * Отличие от OpenCL версии: нет Pinned Memory (нет CL_MEM_ALLOC_HOST_PTR),
 * hipMemcpyHtoDAsync сам оптимизирует transfer через DMA если данные page-locked.
 *
 * @param data          Указатель на CPU данные [element_count × sizeof(complex<float>)]
 * @param element_count Количество complex<float> элементов для копирования
 * @throws std::runtime_error если hipMemcpyHtoDAsync неудача
 */
void SpectrumProcessorROCm::UploadData(const std::complex<float>* data, size_t element_count) {
    size_t data_size = element_count * sizeof(std::complex<float>);

    hipError_t err = hipMemcpyHtoDAsync(input_buffer_,
                                         const_cast<std::complex<float>*>(data),
                                         data_size, ctx_.stream());
    if (err != hipSuccess) {
        throw std::runtime_error("UploadData: hipMemcpyHtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

/**
 * @brief Асинхронный D2D copy из внешнего GPU буфера в input_buffer_
 *
 * hipMemcpyDtoDAsync(dst=input_buffer_, src=src+offset, count, ctx_.stream()).
 * Используется в ProcessFromGPU и FindAllMaximaFromGPUPipeline для batch обработки
 * где src_offset_bytes указывает на начало текущего batch в исходном буфере.
 *
 * @param src              Внешний device pointer
 * @param src_offset_bytes Смещение в src в байтах (byte pointer arithmetic)
 * @param element_count    Количество complex<float> элементов для копирования
 * @throws std::runtime_error если hipMemcpyDtoDAsync неудача
 */
void SpectrumProcessorROCm::CopyGpuData(void* src, size_t src_offset_bytes, size_t element_count) {
    size_t data_size = element_count * sizeof(std::complex<float>);
    char* src_ptr = static_cast<char*>(src) + src_offset_bytes;

    hipError_t err = hipMemcpyDtoDAsync(input_buffer_, src_ptr, data_size, ctx_.stream());
    if (err != hipSuccess) {
        throw std::runtime_error("CopyGpuData: hipMemcpyDtoDAsync failed: " +
                                  std::string(hipGetErrorString(err)));
    }
}

/**
 * @brief Запустить pad_data kernel: zero-padding [n_point → nFFT] для всех лучей
 *
 * Аналог pre-callback в OpenCL версии, но здесь — отдельный kernel.
 * Читает из input_buffer_ [beam × n_point], пишет в fft_input_ [beam × nFFT].
 * Паддинг нулями: элементы [n_point..nFFT-1] заполняются комплексным нулём.
 *
 * NDRange: grid=((beam_count × nFFT + 255) / 256, 1, 1), block=(256, 1, 1)
 * Args: [input_buffer_, fft_input_, bc, n_point, nFFT, beam_offset]
 *
 * beam_offset используется при batch обработке для индексации исходного буфера.
 *
 * @param beam_count  Количество лучей в текущем batch
 * @param beam_offset Смещение начального луча (для частичных batch, по умолчанию=0)
 * @throws std::runtime_error если hipModuleLaunchKernel неудача
 */
void SpectrumProcessorROCm::ExecutePadKernel(size_t beam_count, size_t beam_offset) {
    pad_op_.Execute(input_buffer_, fft_input_,
                    beam_count, params_.n_point, params_.nFFT,
                    static_cast<uint32_t>(beam_offset));
}

/**
 * @brief Запустить hipFFT (C2C Forward) над fft_input_ → fft_output_
 *
 * Выбирает активный план: allmax_plan_created_ ? allmax_plan_ : plan_.
 * Это позволяет использовать один метод как для Process, так и для AllMaxima pipeline,
 * избегая дублирования кода.
 *
 * hipfftSetStream(plan_, ctx_.stream()) при создании плана — FFT выполняется в ctx_.stream().
 * Результат: fft_output_[beam × nFFT] — комплексный спектр каждого луча.
 *
 * @throws std::runtime_error если hipfftExecC2C неудача
 */
void SpectrumProcessorROCm::ExecuteFFT() {
    hipfftHandle active_plan = allmax_plan_created_ ? allmax_plan_ : plan_;

    hipfftResult result = hipfftExecC2C(
        active_plan,
        static_cast<hipfftComplex*>(fft_input_),
        static_cast<hipfftComplex*>(fft_output_),
        HIPFFT_FORWARD);

    if (result != HIPFFT_SUCCESS) {
        throw std::runtime_error("ExecuteFFT: hipfftExecC2C failed: " +
                                  std::to_string(static_cast<int>(result)));
    }
}

/**
 * @brief Запустить post_kernel: найти пик(и) в спектре каждого луча
 *
 * Аналог post-callback в OpenCL версии, но здесь — отдельный kernel.
 * Читает fft_output_ → пишет maxima_output_ [beam × (4 или 8) × MaxValue].
 *
 * NDRange: ONE workgroup per beam — grid=(beam_count,1,1), block=(LOCAL_SIZE,1,1)
 * Каждый workgroup редуцирует search_range элементов до 1 (ONE_PEAK) или 2 (TWO_PEAKS) пиков.
 *
 * Args: [fft_output_, maxima_output_, bc, nFFT, search_range, sample_rate]
 *
 * @param beam_count  Количество лучей в текущем batch
 * @param beam_offset Смещение (для частичных batch — пока не используется в kernel, будущее)
 * @throws std::runtime_error если hipModuleLaunchKernel неудача
 */
void SpectrumProcessorROCm::ExecutePostKernel(size_t beam_count, size_t beam_offset) {
    post_op_.Execute(fft_output_, maxima_output_,
                     static_cast<uint32_t>(beam_count), params_.nFFT,
                     params_.search_range, params_.sample_rate,
                     params_.peak_mode);
}

/**
 * @brief Запустить compute_magnitudes: fft_output_ → magnitudes_buffer_ (float)
 *
 * |z|² = re² + im² для всех total_elements элементов.
 * Читает fft_output_, пишет в magnitudes_buffer_[total_elements] float.
 *
 * Lazy compile: если kernels_compiled_=false → вызывает CompileKernels() первым.
 * NDRange: grid=((total_elements+255)/256, 1, 1), block=(256, 1, 1)
 * Args: [fft_output_, magnitudes_buffer_, total_size]
 *
 * @param total_elements beam_count × nFFT — общее количество float элементов в magnitudes
 * @throws std::runtime_error если hipModuleLaunchKernel неудача
 */
void SpectrumProcessorROCm::ExecuteComputeMagnitudes(size_t total_elements) {
    if (!compiled_) CompileKernels();
    mag_op_.Execute(fft_output_, magnitudes_buffer_, total_elements);
}

/**
 * @brief Синхронный D2H download результатов из maxima_output_ → SpectrumResult[]
 *
 * hipMemcpyDtoH (блокирующий!) — поэтому вызывается ПОСЛЕ hipStreamSynchronize(ctx_.stream())
 * в ProcessBatch/ProcessFromGPU/ProcessBatchFromGPU.
 *
 * Формат maxima_output_:
 *  - ONE_PEAK:  [beam × 4×MaxValue] → 1 SpectrumResult/beam  (interpolated/left/center/right)
 *  - TWO_PEAKS: [beam × 8×MaxValue] → 2 SpectrumResult/beam (два пика: left+right)
 *
 * antenna_id заполняется как (beam_offset + i).
 *
 * @param beam_count  Количество лучей для чтения
 * @param beam_offset Смещение для antenna_id (при batch — абсолютный индекс первой антенны)
 * @return vector<SpectrumResult>[beam_count] или [2×beam_count] (TWO_PEAKS)
 * @throws std::runtime_error если hipMemcpyDtoH неудача
 */
std::vector<SpectrumResult> SpectrumProcessorROCm::ReadResults(size_t beam_count, size_t beam_offset) {
    size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t num_results = beam_count * max_values_per_beam;
    std::vector<MaxValue> raw_results(num_results);

    hipError_t err = hipMemcpyDtoH(raw_results.data(), maxima_output_,
                                     num_results * sizeof(MaxValue));
    if (err != hipSuccess) {
        throw std::runtime_error("ReadResults: hipMemcpyDtoH failed: " +
                                  std::string(hipGetErrorString(err)));
    }

    std::vector<SpectrumResult> results;

    if (params_.peak_mode == PeakSearchMode::ONE_PEAK) {
        results.reserve(beam_count);
        for (uint32_t i = 0; i < beam_count; ++i) {
            size_t base = i * 4;
            SpectrumResult result{};
            result.antenna_id = static_cast<uint32_t>(beam_offset + i);
            result.interpolated = raw_results[base + 0];
            result.left_point   = raw_results[base + 1];
            result.center_point = raw_results[base + 2];
            result.right_point  = raw_results[base + 3];
            results.push_back(result);
        }
    } else {
        results.reserve(beam_count * 2);
        for (uint32_t i = 0; i < beam_count; ++i) {
            size_t base = i * 8;

            SpectrumResult left{};
            left.antenna_id = static_cast<uint32_t>(beam_offset + i);
            left.interpolated = raw_results[base + 0];
            left.left_point   = raw_results[base + 1];
            left.center_point = raw_results[base + 2];
            left.right_point  = raw_results[base + 3];
            results.push_back(left);

            SpectrumResult right{};
            right.antenna_id = static_cast<uint32_t>(beam_offset + i);
            right.interpolated = raw_results[base + 4];
            right.left_point   = raw_results[base + 5];
            right.center_point = raw_results[base + 6];
            right.right_point  = raw_results[base + 7];
            results.push_back(right);
        }
    }

    return results;
}

// ════════════════════════════════════════════════════════════════════════════
// PART 7: Utilities
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Вычислить nFFT и search_range из params_.n_point и repeat_count
 *
 * base_fft = nextPow2(n_point) — hipFFT требует степень двойки.
 * nFFT = base_fft × repeat_count — zero-padding для частотного разрешения.
 * search_range = 0 → авто = nFFT/4 (первая четверть: положительные частоты).
 *
 * Идентично OpenCL версии — алгоритм backend-независим.
 */
void SpectrumProcessorROCm::CalculateFFTSize() {
    params_.base_fft = NextPowerOf2(params_.n_point);
    params_.nFFT = params_.base_fft * params_.repeat_count;
    if (params_.search_range == 0) {
        params_.search_range = params_.nFFT / 4;
    }
}

/**
 * @brief Округлить вверх до следующей степени двойки (bitwise алгоритм)
 * Например: 1000 → 1024, 1024 → 1024, 0 → 1.
 */
uint32_t SpectrumProcessorROCm::NextPowerOf2(uint32_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

/**
 * @brief Рассчитать потребление памяти на одну антенну для BatchManager
 *
 * Включает все 5 буферов:
 *  - input_bytes:  n_point × sizeof(complex) — сырые данные
 *  - fft_bytes:    2 × nFFT × sizeof(complex) — fft_input + fft_output
 *  - mag_bytes:    nFFT × sizeof(float) — magnitudes (для AllMaxima)
 *  - maxima_bytes: (4 или 8) × sizeof(MaxValue) — результат post_kernel
 *
 * ⚠️ Включает mag_bytes в отличие от OpenCL версии — ROCm всегда создаёт mag буфер.
 *
 * @return Байт на одну антенну (используется BatchManager для расчёта batch_size)
 */
size_t SpectrumProcessorROCm::CalculateBytesPerAntenna() const {
    size_t input_bytes = params_.n_point * sizeof(std::complex<float>);
    size_t fft_bytes = 2 * params_.nFFT * sizeof(std::complex<float>);  // input + output
    size_t mag_bytes = params_.nFFT * sizeof(float);  // magnitudes

    size_t maxima_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
    size_t maxima_bytes = maxima_per_beam * sizeof(MaxValue);

    return input_bytes + fft_bytes + mag_bytes + maxima_bytes;
}

/**
 * @brief Lazy выделение/переаллокация magnitudes_buffer_ [total_elements × float]
 *
 * Guard: если существующий буфер >= total_elements — ничего не делает (переиспользуем).
 * Иначе → освободить старый (hipFree) и выделить новый (hipMalloc).
 *
 * Вызывается из CreateAllMaximaFFTPlan и FindAllMaxima перед ExecuteComputeMagnitudes.
 * magnitudes_buffer_size_ хранит текущий размер в элементах (не байтах).
 *
 * @param total_elements beam_count × nFFT — общее количество float в буфере
 * @throws std::runtime_error если hipMalloc неудача
 */
void SpectrumProcessorROCm::EnsureMagnitudesBuffer(size_t total_elements) {
    if (magnitudes_buffer_ && magnitudes_buffer_size_ >= total_elements) return;

    if (magnitudes_buffer_) {
        (void)hipFree(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
    }

    hipError_t err = hipMalloc(&magnitudes_buffer_, total_elements * sizeof(float));
    if (err != hipSuccess) {
        throw std::runtime_error("EnsureMagnitudesBuffer: hipMalloc failed: " +
                                  std::string(hipGetErrorString(err)));
    }
    magnitudes_buffer_size_ = total_elements;
}

/**
 * @brief Переаллоцировать GPU буферы и/или FFT план под batch_antenna_count антенн
 *
 * ROCm версия имеет 2 пути (без FAST PATH — нет эквивалента WritePreCallbackHeader):
 *
 * FULL PATH (need_new_buffers=true):
 *   - hipFree все буферы → hipMalloc заново под batch_antenna_count
 *   - current_batch_size_ обновляется
 *
 * PLAN ONLY PATH (need_new_buffers=false, need_new_plan=true):
 *   - Буферы достаточны, только пересоздать FFT план через CreateFFTPlan
 *
 * Если оба false — return немедленно (буферы и план актуальны).
 *
 * CompileKernels() вызывается в конце если ещё не скомпилированы.
 *
 * @param batch_antenna_count Целевое количество антенн для нового batch
 * @throws std::runtime_error если hipMalloc или CreateFFTPlan неудача
 */
void SpectrumProcessorROCm::ReallocateBuffersForBatch(size_t batch_antenna_count) {
    bool need_new_plan = (plan_batch_size_ != batch_antenna_count) || !plan_created_;
    bool need_new_buffers = (current_batch_size_ < batch_antenna_count) || !input_buffer_;

    if (!need_new_buffers && !need_new_plan) return;

    if (need_new_buffers) {
        // Free old buffers
        if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
        if (fft_input_)       { (void)hipFree(fft_input_);       fft_input_ = nullptr; }
        if (fft_output_)      { (void)hipFree(fft_output_);      fft_output_ = nullptr; }
        if (maxima_output_)   { (void)hipFree(maxima_output_);   maxima_output_ = nullptr; }

        hipError_t err;

        // Input buffer
        size_t input_size = batch_antenna_count * params_.n_point * sizeof(std::complex<float>);
        err = hipMalloc(&input_buffer_, input_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: input_buffer failed");

        // FFT buffers
        size_t fft_size = batch_antenna_count * params_.nFFT * sizeof(std::complex<float>);
        err = hipMalloc(&fft_input_, fft_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: fft_input failed");

        err = hipMalloc(&fft_output_, fft_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: fft_output failed");

        // Maxima output
        size_t max_values_per_beam = (params_.peak_mode == PeakSearchMode::ONE_PEAK) ? 4 : 8;
        size_t maxima_size = batch_antenna_count * max_values_per_beam * sizeof(MaxValue);
        err = hipMalloc(&maxima_output_, maxima_size);
        if (err != hipSuccess)
            throw std::runtime_error("ReallocateBuffersForBatch: maxima_output failed");

        current_batch_size_ = batch_antenna_count;
    }

    if (need_new_plan) {
        CreateFFTPlan(batch_antenna_count);
    }

    if (!compiled_) CompileKernels();
}

/**
 * @brief Освободить ресурсы AllMaxima pipeline (pipeline + allmax_plan + magnitudes_buffer_)
 *
 * Вызывается из ReleaseResources() и может вызываться явно для сброса AllMaxima режима.
 * Порядок: pipeline_.reset() (Detect+Scan+Compact kernels) → allmax_plan_ → magnitudes_buffer_.
 *
 * Не трогает основные буферы (input/fft_in/fft_out/maxima) и plan_ (Process режим).
 */
void SpectrumProcessorROCm::ReleaseAllMaximaResources() {
    pipeline_.reset();

    if (allmax_plan_created_) {
        hipfftDestroy(allmax_plan_);
        allmax_plan_ = 0;
        allmax_plan_created_ = false;
        allmax_plan_batch_size_ = 0;
    }

    if (magnitudes_buffer_) {
        (void)hipFree(magnitudes_buffer_);
        magnitudes_buffer_ = nullptr;
        magnitudes_buffer_size_ = 0;
    }
}

/**
 * @brief Полное освобождение всех GPU ресурсов (вызывается из деструктора)
 *
 * Порядок освобождения критичен:
 *  1. ReleaseAllMaximaResources() — pipeline_ → allmax_plan_ → magnitudes_buffer_
 *  2. hipfftDestroy(plan_) — Process режим FFT план
 *  3. hipFree для 4 основных буферов (input/fft_in/fft_out/maxima)
 *  4. hipModuleUnload(module_) — hiprtc модуль (все kernels) последним
 *     ⚠️ Если освободить module_ до kernels→UB при следующем запуске
 *
 * ctx_.stream() НЕ закрывается — принадлежит backend_.
 * Флаги сбрасываются: initialized_=false, kernels_compiled_=false, etc.
 */
void SpectrumProcessorROCm::ReleaseResources() {
    ReleaseAllMaximaResources();

    // hipFFT plan (Process mode)
    if (plan_created_) {
        hipfftDestroy(plan_);
        plan_ = 0;
        plan_created_ = false;
        plan_batch_size_ = 0;
    }

    // GPU buffers
    if (input_buffer_)    { (void)hipFree(input_buffer_);    input_buffer_ = nullptr; }
    if (fft_input_)       { (void)hipFree(fft_input_);       fft_input_ = nullptr; }
    if (fft_output_)      { (void)hipFree(fft_output_);      fft_output_ = nullptr; }
    if (maxima_output_)   { (void)hipFree(maxima_output_);   maxima_output_ = nullptr; }

    // GpuContext manages kernel module — no manual hipModuleUnload

    current_batch_size_ = 0;
    actual_batch_size_ = 0;
    initialized_ = false;
}

/**
 * @brief Получить накопленные данные профилирования (TODO: не реализовано)
 *
 * ROCm версия пока возвращает пустой ProfilingData.
 * Профилирование передаётся через ROCmProfEvents* в ProcessBatch/FindAllMaximaFromCPU.
 * Агрегация в GPUProfiler происходит на уровне benchmark тестов, не здесь.
 *
 * @return Пустой ProfilingData (временная заглушка)
 */
ProfilingData SpectrumProcessorROCm::GetProfilingData() const {
    ProfilingData out{};
    // TODO: Add HIP profiling via hipEvent timing when running on AMD GPU
    return out;
}

}  // namespace antenna_fft

#else  // !ENABLE_ROCM

// ════════════════════════════════════════════════════════════════════════════
// Stub implementation for non-ROCm builds (Windows)
// ════════════════════════════════════════════════════════════════════════════

#include <spectrum/processors/spectrum_processor_rocm.hpp>
#include <stdexcept>

namespace antenna_fft {

SpectrumProcessorROCm::SpectrumProcessorROCm(drv_gpu_lib::IBackend* backend)
    : backend_(backend) {}

void SpectrumProcessorROCm::Initialize(const SpectrumParams&) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled (compile with ENABLE_ROCM=1)");
}

std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromCPU(const std::vector<std::complex<float>>&) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessFromGPU(void*, size_t, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatch(const std::vector<std::complex<float>>&, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
std::vector<SpectrumResult> SpectrumProcessorROCm::ProcessBatchFromGPU(void*, size_t, size_t, size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromCPU(const std::vector<std::complex<float>>&, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaximaFromGPUPipeline(void*, size_t, size_t, size_t, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::AllMaximaFromCPU(const std::vector<std::complex<float>>&, uint32_t, uint32_t, float, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
AllMaximaResult SpectrumProcessorROCm::FindAllMaxima(void*, uint32_t, uint32_t, float, OutputDestination, uint32_t, uint32_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
ProfilingData SpectrumProcessorROCm::GetProfilingData() const { return {}; }
void SpectrumProcessorROCm::ReallocateBuffersForBatch(size_t) {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
size_t SpectrumProcessorROCm::CalculateBytesPerAntenna() const {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}
void SpectrumProcessorROCm::CompilePostKernel() {
    throw std::runtime_error("SpectrumProcessorROCm: ROCm not enabled");
}

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
