#pragma once

/**
 * @file all_maxima_pipeline_rocm.hpp
 * @brief ROCm/HIP implementation: Detect -> Scan -> Compact (via GpuContext)
 *
 * Phase B2 of kernel_cache_v2: manual hiprtc + own KernelCacheService removed,
 * compilation delegated to GpuContext::CompileModule (clean-slate v2 cache).
 *
 * Stream-ordered execution (no explicit events). All 4 kernels (detect,
 * block_scan, block_add, compact) live in one hipModule owned by GpuContext.
 *
 * Compiles ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23  (migrated 2026-04-22 to GpuContext)
 */

#if ENABLE_ROCM

#include <spectrum/interface/i_all_maxima_pipeline.hpp>
#include <spectrum/interface/spectrum_maxima_types.h>
#include <core/interface/i_backend.hpp>
#include <core/interface/gpu_context.hpp>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstddef>
#include <memory>

namespace antenna_fft {

/**
 * @class AllMaximaPipelineROCm
 * @brief HIP pipeline: detect_all_maxima -> prefix_sum -> compact_maxima
 *
 * All operations are stream-ordered on the hipStream provided by the backend.
 * Kernels compiled once via GpuContext::CompileModule, reused across Execute()
 * calls (GpuContext idempotent).
 */
class AllMaximaPipelineROCm : public IAllMaximaPipeline {
public:
    AllMaximaPipelineROCm(hipStream_t stream, drv_gpu_lib::IBackend* backend);
    ~AllMaximaPipelineROCm() override;

    AllMaximaPipelineROCm(const AllMaximaPipelineROCm&) = delete;
    AllMaximaPipelineROCm& operator=(const AllMaximaPipelineROCm&) = delete;

    AllMaximaResult Execute(
        void* magnitudes_gpu,
        void* fft_data_gpu,
        uint32_t beam_count,
        uint32_t nFFT,
        float sample_rate,
        OutputDestination dest = OutputDestination::CPU,
        uint32_t search_start = 1,
        uint32_t search_end = 0,
        size_t max_maxima_per_beam = 1000) override;

private:
    /// Lazy compile via GpuContext (idempotent, disk-cached).
    void CompileKernels();

    /// Recursive beam-aware Blelloch prefix sum
    void ExecutePrefixSum(void* input, void* output,
                          size_t n_per_beam, size_t beam_count);

    hipStream_t stream_ = nullptr;
    drv_gpu_lib::IBackend* backend_ = nullptr;

    /// Owns hipModule + 4 hipFunction_t via GpuContext + KernelCacheService v2.
    std::unique_ptr<drv_gpu_lib::GpuContext> ctx_;

    /// Cached kernel fn pointers after ctx_->CompileModule().
    hipFunction_t detect_kernel_     = nullptr;
    hipFunction_t block_scan_kernel_ = nullptr;
    hipFunction_t block_add_kernel_  = nullptr;
    hipFunction_t compact_kernel_    = nullptr;

    static constexpr size_t SCAN_LOCAL_SIZE = 256;
    static constexpr size_t SCAN_BLOCK_SIZE = SCAN_LOCAL_SIZE * 2;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
