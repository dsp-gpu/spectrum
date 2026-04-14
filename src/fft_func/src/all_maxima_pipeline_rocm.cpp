/**
 * @file all_maxima_pipeline_rocm.cpp
 * @brief ROCm/HIP pipeline: Detect -> Scan -> Compact
 *
 * Port of all_maxima_pipeline_opencl.cpp with HIP equivalents.
 * All operations stream-ordered via hipStream (no explicit events).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <spectrum/pipelines/all_maxima_pipeline_rocm.hpp>
#include "kernels/all_maxima_kernel_sources_rocm.hpp"
#include <core/services/console_output.hpp>
#include <core/services/gpu_profiler.hpp>
#include <core/services/kernel_cache_service.hpp>
#include <core/backends/rocm/rocm_backend.hpp>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

namespace antenna_fft {

// ════════════════════════════════════════════════════════════════════════════
// Constructor, destructor
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Создать ROCm pipeline; kernel'ы компилируются лениво при первом Execute()
 *
 * Все операции выполняются на указанном hipStream (stream-ordered, без явных event).
 * Не владеет stream_ и backend_ — их lifetime управляется IBackend.
 *
 * @param stream  HIP stream для всех kernel launch и memcpy
 * @param backend IBackend для доступа к ресурсам ROCm backend
 * @throws std::invalid_argument если stream == nullptr
 */
AllMaximaPipelineROCm::AllMaximaPipelineROCm(hipStream_t stream,
                                               drv_gpu_lib::IBackend* backend)
    : stream_(stream), backend_(backend)
{
    if (!stream_) {
        throw std::invalid_argument("AllMaximaPipelineROCm: hipStream required");
    }
}

/**
 * @brief Деструктор — выгружает hipModule (освобождает все kernel'ы одним вызовом)
 * hipModuleUnload освобождает module + все связанные hipFunction_t автоматически.
 */
AllMaximaPipelineROCm::~AllMaximaPipelineROCm() {
    if (module_) {
        (void)hipModuleUnload(module_);
        module_ = nullptr;
    }
    detect_kernel_ = nullptr;
    block_scan_kernel_ = nullptr;
    block_add_kernel_ = nullptr;
    compact_kernel_ = nullptr;
    kernels_compiled_ = false;
}

// ════════════════════════════════════════════════════════════════════════════
// CompileKernels — hiprtc compilation
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Скомпилировать HIP kernel'ы через hiprtc (idempotent)
 *
 * Один hipModule содержит все 4 функции: detect_all_maxima, block_scan, block_add, compact_maxima.
 * Компилируется с -O3. Source из GetAllMaximaHIPKernelSource() (inline строка в all_maxima_kernel_sources_rocm.hpp).
 *
 * Почему hiprtc (не hipModuleLoad из файла):
 * - Нет зависимости от пути к .co файлам в runtime
 * - Компиляция под текущую GPU архитектуру (gfx1201 и др.)
 *
 * @throws std::runtime_error при ошибке компиляции (с полным hiprtc build log)
 */
void AllMaximaPipelineROCm::CompileKernels() {
    if (kernels_compiled_) return;

    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
    constexpr const char* kCacheName = "all_maxima_kernels";
    const char* src = kernels::GetAllMaximaHIPKernelSource();

    // ── Try loading from disk cache (HSACO) ──
    drv_gpu_lib::KernelCacheService cache(
        "modules/fft_func/kernels", drv_gpu_lib::BackendType::ROCm);
    {
        auto entry = cache.Load(kCacheName);
        if (entry && entry->has_binary()) {
            hipError_t hip_err = hipModuleLoadData(&module_, entry->binary.data());
            if (hip_err == hipSuccess) {
                hip_err = hipModuleGetFunction(&detect_kernel_, module_, "detect_all_maxima");
                if (hip_err == hipSuccess)
                    hip_err = hipModuleGetFunction(&block_scan_kernel_, module_, "block_scan");
                if (hip_err == hipSuccess)
                    hip_err = hipModuleGetFunction(&block_add_kernel_, module_, "block_add");
                if (hip_err == hipSuccess)
                    hip_err = hipModuleGetFunction(&compact_kernel_, module_, "compact_maxima");
                if (hip_err == hipSuccess) {
                    kernels_compiled_ = true;
                    con.Print(0, "AllMaximaPipelineROCm",
                        "Pipeline kernels loaded from cache (HSACO)");
                    return;
                }
            }
            if (module_) { hipModuleUnload(module_); module_ = nullptr; }
        }
    }

    // ── Compile from source (hiprtc) ──
    hiprtcProgram prog;
    hiprtcResult rtc_err = hiprtcCreateProgram(&prog, src, "all_maxima_kernels.hip",
                                                0, nullptr, nullptr);
    if (rtc_err != HIPRTC_SUCCESS) {
        throw std::runtime_error("AllMaximaPipelineROCm: hiprtcCreateProgram failed: " +
                                  std::string(hiprtcGetErrorString(rtc_err)));
    }

    // Build compile options: -O3, --offload-arch, -DWARP_SIZE
    std::string arch_name;
    int warp_size = 32;
    try {
        auto* rocm_backend = static_cast<drv_gpu_lib::ROCmBackend*>(backend_);
        arch_name = rocm_backend->GetCore().GetArchName();
        warp_size = rocm_backend->GetCore().GetWarpSize();
    } catch (...) {
        arch_name = "";
    }
    std::string warp_define = "-DWARP_SIZE=" + std::to_string(warp_size);
    std::string arch_flag = arch_name.empty() ? "" : ("--offload-arch=" + arch_name);
    std::vector<const char*> opts = {"-O3", warp_define.c_str()};
    if (!arch_flag.empty())
        opts.push_back(arch_flag.c_str());

    rtc_err = hiprtcCompileProgram(prog,
        static_cast<int>(opts.size()), opts.data());
    if (rtc_err != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, log.data());

        con.PrintError(0, "AllMaximaPipelineROCm", "Kernel compile log:\n" + log);

        (void)hiprtcDestroyProgram(&prog);
        throw std::runtime_error("AllMaximaPipelineROCm: hiprtcCompileProgram failed: " +
                                  std::string(hiprtcGetErrorString(rtc_err)));
    }

    size_t code_size = 0;
    hiprtcGetCodeSize(prog, &code_size);
    std::vector<char> code(code_size);
    hiprtcGetCode(prog, code.data());
    (void)hiprtcDestroyProgram(&prog);

    hipError_t hip_err = hipModuleLoadData(&module_, code.data());
    if (hip_err != hipSuccess) {
        throw std::runtime_error("AllMaximaPipelineROCm: hipModuleLoadData failed: " +
                                  std::string(hipGetErrorString(hip_err)));
    }

    hip_err = hipModuleGetFunction(&detect_kernel_, module_, "detect_all_maxima");
    if (hip_err != hipSuccess)
        throw std::runtime_error("AllMaximaPipelineROCm: get detect_all_maxima failed");

    hip_err = hipModuleGetFunction(&block_scan_kernel_, module_, "block_scan");
    if (hip_err != hipSuccess)
        throw std::runtime_error("AllMaximaPipelineROCm: get block_scan failed");

    hip_err = hipModuleGetFunction(&block_add_kernel_, module_, "block_add");
    if (hip_err != hipSuccess)
        throw std::runtime_error("AllMaximaPipelineROCm: get block_add failed");

    hip_err = hipModuleGetFunction(&compact_kernel_, module_, "compact_maxima");
    if (hip_err != hipSuccess)
        throw std::runtime_error("AllMaximaPipelineROCm: get compact_maxima failed");

    kernels_compiled_ = true;

    // ── Save to cache for next run ──
    try {
        std::vector<uint8_t> binary(code.begin(), code.end());
        cache.Save(kCacheName, std::string(src), binary,
                   arch_name, "all_maxima: detect + scan + add + compact");
    } catch (...) {
        // Non-critical
    }

    con.Print(0, "AllMaximaPipelineROCm",
        "Pipeline kernels compiled (detect + scan + compact)" +
        (arch_name.empty() ? "" : " [" + arch_name + "]"));
}

// ════════════════════════════════════════════════════════════════════════════
// ExecutePrefixSum (beam-aware, recursive Blelloch)
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Beam-aware Blelloch prefix sum (ROCm/HIP stream-ordered)
 *
 * Аналог OpenCL версии но без явных cl_event — все kernel'ы выполняются
 * stream-ordered на stream_, sync происходит через hipStreamSynchronize перед hipFree.
 *
 * При blocks_per_beam == 1 → один block_scan, рекурсия не нужна.
 * При blocks_per_beam > 1 → block_scan + рекурсивный ExecutePrefixSum(block_sums) + block_add.
 *
 * temp буферы (block_sums, block_sums_scanned) выделяются hipMalloc и освобождаются
 * после hipStreamSynchronize — safe т.к. kernel'ы уже завершены.
 *
 * @param input      Входной буфер флагов (void* = hipDeviceptr_t, uint32[beam×nFFT])
 * @param output     Выходной буфер scan (uint32[beam×nFFT])
 * @param n_per_beam Элементов на луч (= nFFT)
 * @param beam_count Количество лучей
 */
void AllMaximaPipelineROCm::ExecutePrefixSum(
    void* input, void* output, size_t n_per_beam, size_t beam_count)
{
    size_t blocks_per_beam = (n_per_beam + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
    size_t total_blocks = beam_count * blocks_per_beam;

    unsigned int npb = static_cast<unsigned int>(n_per_beam);
    unsigned int bc = static_cast<unsigned int>(beam_count);
    unsigned int bpb = static_cast<unsigned int>(blocks_per_beam);

    // +1 padding eliminates LDS bank conflicts in Blelloch up/down sweep
    size_t shared_mem_size = (SCAN_BLOCK_SIZE + 1) * sizeof(unsigned int);

    if (blocks_per_beam == 1) {
        // Single block per beam — no block_add needed
        void* null_ptr = nullptr;

        void* args[] = {
            &input, &output, &null_ptr,
            &npb, &bc, &bpb
        };

        unsigned int grid_x = static_cast<unsigned int>(total_blocks);
        unsigned int block_x = static_cast<unsigned int>(SCAN_LOCAL_SIZE);

        hipError_t err = hipModuleLaunchKernel(
            block_scan_kernel_,
            grid_x, 1, 1,
            block_x, 1, 1,
            shared_mem_size, stream_,
            args, nullptr);
        if (err != hipSuccess)
            throw std::runtime_error("ExecutePrefixSum: block_scan launch failed: " +
                                      std::string(hipGetErrorString(err)));
        return;
    }

    // Multi-block: scan -> recursive scan of sums -> block_add
    void* block_sums = nullptr;
    void* block_sums_scanned = nullptr;

    hipError_t err = hipMalloc(&block_sums, total_blocks * sizeof(unsigned int));
    if (err != hipSuccess)
        throw std::runtime_error("ExecutePrefixSum: block_sums alloc failed");

    err = hipMalloc(&block_sums_scanned, total_blocks * sizeof(unsigned int));
    if (err != hipSuccess) {
        (void)hipFree(block_sums);
        throw std::runtime_error("ExecutePrefixSum: block_sums_scanned alloc failed");
    }

    // Step 1: Block-level scan (writes block_sums)
    {
        void* args[] = {
            &input, &output, &block_sums,
            &npb, &bc, &bpb
        };

        unsigned int grid_x = static_cast<unsigned int>(total_blocks);
        unsigned int block_x = static_cast<unsigned int>(SCAN_LOCAL_SIZE);

        err = hipModuleLaunchKernel(
            block_scan_kernel_,
            grid_x, 1, 1,
            block_x, 1, 1,
            shared_mem_size, stream_,
            args, nullptr);
        if (err != hipSuccess) {
            (void)hipFree(block_sums);
            (void)hipFree(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: L1 block_scan launch failed");
        }
    }

    // Step 2: Recursive scan of block sums
    ExecutePrefixSum(block_sums, block_sums_scanned, blocks_per_beam, beam_count);

    // Step 3: Add scanned block sums back (2D grid: X=positions, Y=beams)
    {
        unsigned int block_size_uint = static_cast<unsigned int>(SCAN_BLOCK_SIZE);

        void* args[] = {
            &output, &block_sums_scanned,
            &npb, &bc, &bpb, &block_size_uint
        };

        unsigned int grid_x = static_cast<unsigned int>((n_per_beam + 255) / 256);
        unsigned int grid_y = bc;
        unsigned int block_x = 256;

        err = hipModuleLaunchKernel(
            block_add_kernel_,
            grid_x, grid_y, 1,
            block_x, 1, 1,
            0, stream_,
            args, nullptr);
        if (err != hipSuccess) {
            (void)hipFree(block_sums);
            (void)hipFree(block_sums_scanned);
            throw std::runtime_error("ExecutePrefixSum: block_add launch failed");
        }
    }

    // Sync before freeing temp buffers (they're used by stream-ordered kernels)
    hipStreamSynchronize(stream_);

    (void)hipFree(block_sums);
    (void)hipFree(block_sums_scanned);
}

// ════════════════════════════════════════════════════════════════════════════
// Execute — main pipeline
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Запустить ROCm pipeline: Detect → Scan → Compact (все на hipStream)
 *
 * ROCm аналог AllMaximaPipelineOpenCL::Execute() — идентичная логика,
 * но без cl_event (stream-ordered), hipModuleLaunchKernel вместо clEnqueueNDRangeKernel.
 *
 * Temp буферы (flags, scan) выделяются hipMalloc внутри и освобождаются после sync.
 * out_maxima/out_counts: ownership зависит от dest (аналогично OpenCL версии — см. AllMaximaResult).
 *
 * hipStreamSynchronize вызывается ОДИН раз после Step 3 — wait перед CPU read beam_counts.
 * Это гарантирует что compact_kernel завершён, но не блокирует между шагами.
 *
 * @param magnitudes_gpu  HIP device pointer float[beam_count × nFFT]
 * @param fft_data_gpu    HIP device pointer complex float2[beam_count × nFFT]
 * @param beam_count      Количество лучей
 * @param nFFT            Размер FFT (power-of-2)
 * @param sample_rate     Для refined_frequency в MaxValue
 * @param dest            CPU / GPU / ALL
 * @param search_start    0 → auto = 1
 * @param search_end      0 → auto = nFFT/2
 * @param max_maxima_per_beam  Лимит MaxValue на луч
 */
AllMaximaResult AllMaximaPipelineROCm::Execute(
    void* magnitudes_gpu,
    void* fft_data_gpu,
    uint32_t beam_count,
    uint32_t nFFT,
    float sample_rate,
    OutputDestination dest,
    uint32_t search_start,
    uint32_t search_end,
    size_t max_maxima_per_beam)
{
    if (!magnitudes_gpu)
        throw std::invalid_argument("AllMaximaPipelineROCm::Execute: magnitudes_gpu cannot be null");
    if (!fft_data_gpu)
        throw std::invalid_argument("AllMaximaPipelineROCm::Execute: fft_data_gpu required");

    if (search_start == 0) search_start = 1;
    if (search_end == 0) search_end = nFFT / 2;
    if (search_end >= nFFT) search_end = nFFT - 1;

    CompileKernels();

    const size_t total_elements = static_cast<size_t>(beam_count) * nFFT;
    auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

    // ────────────────────────────────────────────────────────────────────
    // Allocate temp GPU buffers
    // ────────────────────────────────────────────────────────────────────

    void* flags_buf = nullptr;
    void* scan_buf = nullptr;
    void* out_maxima = nullptr;
    void* out_beam_counts = nullptr;

    hipError_t err;

    err = hipMalloc(&flags_buf, total_elements * sizeof(unsigned int));
    if (err != hipSuccess)
        throw std::runtime_error("AllMaximaPipelineROCm: flags_buf alloc failed");

    err = hipMalloc(&scan_buf, total_elements * sizeof(unsigned int));
    if (err != hipSuccess) {
        (void)hipFree(flags_buf);
        throw std::runtime_error("AllMaximaPipelineROCm: scan_buf alloc failed");
    }

    uint32_t max_output_per_beam = std::min(
        (search_end - search_start) / 2,
        static_cast<uint32_t>(max_maxima_per_beam)
    );

    err = hipMalloc(&out_maxima,
        static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue));
    if (err != hipSuccess) {
        (void)hipFree(flags_buf);
        (void)hipFree(scan_buf);
        throw std::runtime_error("AllMaximaPipelineROCm: out_maxima alloc failed");
    }

    err = hipMalloc(&out_beam_counts, beam_count * sizeof(unsigned int));
    if (err != hipSuccess) {
        (void)hipFree(flags_buf);
        (void)hipFree(scan_buf);
        (void)hipFree(out_maxima);
        throw std::runtime_error("AllMaximaPipelineROCm: out_beam_counts alloc failed");
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 1: Detect maxima (2D grid: X=positions, Y=beams)
    // ────────────────────────────────────────────────────────────────────

    {
        void* args[] = {
            &magnitudes_gpu, &flags_buf,
            &beam_count, &nFFT, &search_start, &search_end
        };

        unsigned int grid_x = static_cast<unsigned int>((nFFT + 255) / 256);
        unsigned int grid_y = beam_count;
        unsigned int block_x = 256;

        err = hipModuleLaunchKernel(
            detect_kernel_,
            grid_x, grid_y, 1,
            block_x, 1, 1,
            0, stream_,
            args, nullptr);
        if (err != hipSuccess) {
            (void)hipFree(flags_buf); (void)hipFree(scan_buf);
            (void)hipFree(out_maxima); (void)hipFree(out_beam_counts);
            throw std::runtime_error("AllMaximaPipelineROCm: detect launch failed");
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Step 2: Prefix sum (beam-aware Blelloch)
    // ────────────────────────────────────────────────────────────────────

    ExecutePrefixSum(flags_buf, scan_buf, nFFT, beam_count);

    // ────────────────────────────────────────────────────────────────────
    // Step 3: Stream compaction (2D grid: X=positions, Y=beams)
    // ────────────────────────────────────────────────────────────────────

    {
        unsigned int beam_offset = 0;

        void* args[] = {
            &fft_data_gpu, &magnitudes_gpu,
            &flags_buf, &scan_buf,
            &out_maxima, &out_beam_counts,
            &beam_count, &nFFT, &sample_rate,
            &max_output_per_beam, &beam_offset
        };

        unsigned int grid_x = static_cast<unsigned int>((nFFT + 255) / 256);
        unsigned int grid_y = beam_count;
        unsigned int block_x = 256;

        err = hipModuleLaunchKernel(
            compact_kernel_,
            grid_x, grid_y, 1,
            block_x, 1, 1,
            0, stream_,
            args, nullptr);
        if (err != hipSuccess) {
            (void)hipFree(flags_buf); (void)hipFree(scan_buf);
            (void)hipFree(out_maxima); (void)hipFree(out_beam_counts);
            throw std::runtime_error("AllMaximaPipelineROCm: compact launch failed");
        }
    }

    // Sync to ensure all GPU work is done
    hipStreamSynchronize(stream_);

    // ────────────────────────────────────────────────────────────────────
    // Read beam counts
    // ────────────────────────────────────────────────────────────────────

    std::vector<uint32_t> beam_counts(beam_count);
    err = hipMemcpyDtoH(beam_counts.data(), out_beam_counts,
                         beam_count * sizeof(unsigned int));
    if (err != hipSuccess) {
        (void)hipFree(flags_buf); (void)hipFree(scan_buf);
        (void)hipFree(out_maxima); (void)hipFree(out_beam_counts);
        throw std::runtime_error("AllMaximaPipelineROCm: read beam_counts failed");
    }

    // ────────────────────────────────────────────────────────────────────
    // Build result
    // ────────────────────────────────────────────────────────────────────

    AllMaximaResult result;
    result.destination = dest;
    result.total_maxima = 0;

    if (dest == OutputDestination::CPU || dest == OutputDestination::ALL) {
        result.beams.resize(beam_count);

        for (uint32_t b = 0; b < beam_count; ++b) {
            uint32_t count = beam_counts[b];
            if (count > max_output_per_beam) {
                con.Print(0, "AllMaximaPipeline",
                    "WARNING: Beam " + std::to_string(b) +
                    " reached max_maxima limit (" + std::to_string(count) +
                    "/" + std::to_string(max_output_per_beam) + "), results truncated");
                count = max_output_per_beam;
            }

            result.beams[b].antenna_id = b;
            result.beams[b].num_maxima = count;
            result.total_maxima += count;

            if (count > 0) {
                result.beams[b].maxima.resize(count);

                size_t out_offset = static_cast<size_t>(b) * max_output_per_beam;
                err = hipMemcpyDtoH(
                    result.beams[b].maxima.data(),
                    static_cast<char*>(out_maxima) + out_offset * sizeof(MaxValue),
                    count * sizeof(MaxValue));
                if (err != hipSuccess) {
                    con.PrintError(0, "AllMaximaPipelineROCm",
                        "Failed to read maxima for beam " + std::to_string(b));
                }
            }
        }
    } else {
        for (uint32_t b = 0; b < beam_count; ++b) {
            result.total_maxima += beam_counts[b];
        }
    }

    if (dest == OutputDestination::GPU || dest == OutputDestination::ALL) {
        result.gpu_maxima = out_maxima;
        result.gpu_counts = out_beam_counts;
        result.gpu_bytes = static_cast<size_t>(beam_count) * max_output_per_beam * sizeof(MaxValue);
        out_maxima = nullptr;
        out_beam_counts = nullptr;
    }

    con.Print(0, "AllMaximaPipelineROCm",
        "Found " + std::to_string(result.total_maxima) + " maxima");

    // ────────────────────────────────────────────────────────────────────
    // Cleanup temp buffers
    // ────────────────────────────────────────────────────────────────────

    (void)hipFree(flags_buf);
    (void)hipFree(scan_buf);
    if (out_maxima) (void)hipFree(out_maxima);
    if (out_beam_counts) (void)hipFree(out_beam_counts);

    return result;
}

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
