#pragma once

/**
 * @file all_maxima_kernel_sources_rocm.hpp
 * @brief HIP kernel sources for AllMaximaPipelineROCm
 *
 * Pipeline: Detection -> Prefix Sum (Blelloch Scan) -> Stream Compaction
 *
 * Contains:
 * - detect_all_maxima: mark local maxima (2D grid, no div/mod)
 * - block_scan: Blelloch work-efficient exclusive scan (beam-aware)
 * - block_add: add scanned block sums to each element (2D grid)
 * - compact_maxima: stream compaction -> MaxValue[] (2D grid)
 *
 * Ported from all_maxima_kernel_sources.hpp (OpenCL -> HIP).
 *
 * Optimizations ported from OpenCL:
 *   - __launch_bounds__(256) on all kernels
 *   - 2D grid for detect/block_add/compact (eliminates div/mod)
 *   - __restrict__ on all pointer params
 *   - INV_PI_DEG constant (multiply instead of divide)
 *   - +1 LDS padding in block_scan (via dispatch shared_mem_size)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

namespace antenna_fft {
namespace kernels {

inline const char* GetAllMaximaHIPKernelSource() {
    return R"HIP(

// ============================================================================
// Common structures
// ============================================================================

struct float2_t {
    float x;
    float y;
};

struct MaxValue_t {
    unsigned int index;
    float real;
    float imag;
    float magnitude;
    float phase;
    float freq_offset;
    float refined_frequency;
    unsigned int pad;
};

// ============================================================================
// 1. detect_all_maxima - Mark local maxima (2D grid, no div/mod)
// ============================================================================
//
// 2D grid: blockIdx.x = position blocks, blockIdx.y = beam index
// For each element: if mag[i] > mag[i-1] && mag[i] > mag[i+1] -> flags[i] = 1
//
extern "C" __launch_bounds__(256)
__global__ void detect_all_maxima(
    const float* __restrict__ magnitudes,
    unsigned int* __restrict__ flags,
    unsigned int beam_count,
    unsigned int nFFT,
    unsigned int search_start,
    unsigned int search_end)
{
    unsigned int pos      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int beam_idx = blockIdx.y;

    if (beam_idx >= beam_count) return;
    if (pos >= nFFT) return;

    unsigned int gid = beam_idx * nFFT + pos;

    // Outside search range -> not a maximum
    if (pos < search_start || pos >= search_end) {
        flags[gid] = 0;
        return;
    }

    // Read pre-computed magnitudes
    float mag_c = magnitudes[gid];
    float mag_l = magnitudes[gid - 1];
    float mag_r = magnitudes[gid + 1];

    // Strict inequality: local maximum
    flags[gid] = (mag_c > mag_l && mag_c > mag_r) ? 1 : 0;
}

// ============================================================================
// 2. block_scan - Blelloch work-efficient exclusive scan (beam-aware)
// ============================================================================
//
// One work-group processes BLOCK_SIZE = 2 * blockDim.x elements.
// All beams are scanned in parallel.
//
// Dynamic shared memory: extern __shared__ unsigned int temp[]
// Size must be (BLOCK_SIZE + 1) * sizeof(uint) — passed via launch parameter.
// +1 padding eliminates LDS bank conflicts in up/down sweep.
//
extern "C" __launch_bounds__(256)
__global__ void block_scan(
    const unsigned int* __restrict__ input,
    unsigned int* __restrict__ output,
    unsigned int* __restrict__ block_sums,
    unsigned int n_per_beam,
    unsigned int beam_count,
    unsigned int blocks_per_beam)
{
    extern __shared__ unsigned int temp[];

    unsigned int lid = threadIdx.x;
    unsigned int group_id = blockIdx.x;
    unsigned int local_size = blockDim.x;
    unsigned int BLOCK_SIZE = local_size * 2;

    // Beam-aware indexing
    unsigned int beam_idx = group_id / blocks_per_beam;
    unsigned int block_idx = group_id % blocks_per_beam;
    unsigned int block_offset = beam_idx * n_per_beam + block_idx * BLOCK_SIZE;
    unsigned int beam_end = beam_idx * n_per_beam + n_per_beam;

    unsigned int ai = lid;
    unsigned int bi = lid + local_size;

    // Load into shared memory (bounds check per beam)
    temp[ai] = (block_offset + ai < beam_end) ? input[block_offset + ai] : 0;
    temp[bi] = (block_offset + bi < beam_end) ? input[block_offset + bi] : 0;
    __syncthreads();

    // === Up-sweep (reduce) ===
    unsigned int offset = 1;
    for (unsigned int d = BLOCK_SIZE >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (lid < d) {
            unsigned int ai_idx = offset * (2 * lid + 1) - 1;
            unsigned int bi_idx = offset * (2 * lid + 2) - 1;
            temp[bi_idx] += temp[ai_idx];
        }
        offset <<= 1;
    }

    // Save block sum and zero last element
    if (lid == 0) {
        if (block_sums != 0) {
            block_sums[group_id] = temp[BLOCK_SIZE - 1];
        }
        temp[BLOCK_SIZE - 1] = 0;
    }

    // === Down-sweep ===
    for (unsigned int d = 1; d < BLOCK_SIZE; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (lid < d) {
            unsigned int ai_idx = offset * (2 * lid + 1) - 1;
            unsigned int bi_idx = offset * (2 * lid + 2) - 1;
            unsigned int t = temp[ai_idx];
            temp[ai_idx] = temp[bi_idx];
            temp[bi_idx] += t;
        }
    }
    __syncthreads();

    // Write result (bounds check per beam)
    if (block_offset + ai < beam_end) output[block_offset + ai] = temp[ai];
    if (block_offset + bi < beam_end) output[block_offset + bi] = temp[bi];
}

// ============================================================================
// 3. block_add - Add scanned block sums back (2D grid, no div/mod for beam)
// ============================================================================
//
// 2D grid: blockIdx.x = position blocks, blockIdx.y = beam index
//
extern "C" __launch_bounds__(256)
__global__ void block_add(
    unsigned int* __restrict__ data,
    const unsigned int* __restrict__ block_sums,
    unsigned int n_per_beam,
    unsigned int beam_count,
    unsigned int blocks_per_beam,
    unsigned int block_size)
{
    unsigned int pos_in_beam = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int beam_idx    = blockIdx.y;

    if (beam_idx >= beam_count) return;
    if (pos_in_beam >= n_per_beam) return;

    unsigned int block_idx = pos_in_beam / block_size;

    if (block_idx > 0) {
        unsigned int gid = beam_idx * n_per_beam + pos_in_beam;
        data[gid] += block_sums[beam_idx * blocks_per_beam + block_idx];
    }
}

// ============================================================================
// 4. compact_maxima - Stream compaction -> MaxValue[] (2D grid)
// ============================================================================
//
// 2D grid: blockIdx.x = position blocks, blockIdx.y = beam index
// For each flags[i] == 1: write MaxValue using scan_output[i] as write index.
//
extern "C" __launch_bounds__(256)
__global__ void compact_maxima(
    const float2_t* __restrict__ fft_output,
    const float* __restrict__ magnitudes,
    const unsigned int* __restrict__ flags,
    const unsigned int* __restrict__ scan_output,
    MaxValue_t* __restrict__ out_maxima,
    unsigned int* __restrict__ out_beam_counts,
    unsigned int beam_count,
    unsigned int nFFT,
    float sample_rate,
    unsigned int max_output_per_beam,
    unsigned int beam_offset)
{
    unsigned int pos      = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int beam_idx = blockIdx.y;

    if (beam_idx >= beam_count) return;
    if (pos >= nFFT) return;

    unsigned int gid = beam_idx * nFFT + pos;
    unsigned int global_beam_idx = beam_offset + beam_idx;

    // Last element in beam -> write count
    if (pos == nFFT - 1) {
        unsigned int base = beam_idx * nFFT;
        unsigned int count = scan_output[base + nFFT - 1] + flags[base + nFFT - 1];
        out_beam_counts[global_beam_idx] = count;
    }

    // If this is a maximum -> write MaxValue
    unsigned int base = beam_idx * nFFT;
    if (flags[base + pos] == 1) {
        unsigned int compact_idx = scan_output[base + pos];

        if (compact_idx < max_output_per_beam) {
            unsigned int out_base = global_beam_idx * max_output_per_beam;

            float2_t cv = fft_output[gid];
            float re = cv.x;
            float im = cv.y;
            float mag = magnitudes[gid];
            float phase_deg = atan2f(im, re) * 57.29577951f;
            float bin_width = sample_rate / (float)nFFT;
            float refined_freq = (float)pos * bin_width;

            MaxValue_t mv;
            mv.index = pos;
            mv.real = re;
            mv.imag = im;
            mv.magnitude = mag;
            mv.phase = phase_deg;
            mv.freq_offset = 0.0f;
            mv.refined_frequency = refined_freq;
            mv.pad = 0;

            out_maxima[out_base + compact_idx] = mv;
        }
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace antenna_fft

#endif  // ENABLE_ROCM
