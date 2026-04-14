#pragma once

/**
 * @file all_maxima_pipeline_rocm.hpp
 * @brief ROCm/HIP implementation: Detect -> Scan -> Compact
 *
 * Port of AllMaximaPipelineOpenCL with HIP equivalents:
 * - hiprtc-compiled kernels (detect, block_scan, block_add, compact)
 * - hipModuleLaunchKernel for execution
 * - hipMalloc/hipFree for temp buffers
 * - Stream-ordered execution (no explicit events)
 *
 * Compiles ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <spectrum/interface/i_all_maxima_pipeline.hpp>
#include <spectrum/interface/spectrum_maxima_types.h>
#include <core/interface/i_backend.hpp>

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <cstdint>
#include <cstddef>

namespace antenna_fft {

/**
 * @class AllMaximaPipelineROCm
 * @brief HIP pipeline: detect_all_maxima -> prefix_sum -> compact_maxima
 *
 * All operations are stream-ordered on the hipStream provided by the backend.
 * Kernels compiled once via hiprtc, reused across Execute() calls.
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
    /// Compile HIP kernels via hiprtc
    void CompileKernels();

    /// Recursive beam-aware Blelloch prefix sum
    void ExecutePrefixSum(void* input, void* output,
                          size_t n_per_beam, size_t beam_count);

    hipStream_t stream_ = nullptr;
    drv_gpu_lib::IBackend* backend_ = nullptr;

    hipModule_t module_ = nullptr;
    hipFunction_t detect_kernel_ = nullptr;
    hipFunction_t block_scan_kernel_ = nullptr;
    hipFunction_t block_add_kernel_ = nullptr;
    hipFunction_t compact_kernel_ = nullptr;
    bool kernels_compiled_ = false;

    static constexpr size_t SCAN_LOCAL_SIZE = 256;
    static constexpr size_t SCAN_BLOCK_SIZE = SCAN_LOCAL_SIZE * 2;
};

}  // namespace antenna_fft

#endif  // ENABLE_ROCM
