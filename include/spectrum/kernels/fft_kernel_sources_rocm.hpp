#pragma once

/**
 * @file fft_kernel_sources_rocm.hpp
 * @brief HIP kernel sources for SpectrumProcessorROCm
 *
 * Contains:
 * - pad_data: zero-padding n_point -> nFFT for batch FFT (2D grid)
 * - compute_magnitudes: |FFT[i]| = sqrt(re^2 + im^2)
 * - post_kernel_one_peak: ONE_PEAK search with parabolic interpolation
 * - post_kernel_two_peaks: TWO_PEAKS search (left + right independent maxima)
 *
 * Kernels are compiled at runtime via hiprtc.
 * Uses custom float2_t struct to avoid hiprtc built-in type issues.
 *
 * Ported from fft_kernel_sources.hpp (OpenCL -> HIP).
 *
 * Optimizations ported from OpenCL:
 *   - __launch_bounds__(256) on all kernels
 *   - __fsqrt_rn instead of sqrtf (hardware intrinsic)
 *   - Tree-reduction O(log2(N)) instead of O(N) sequential loop
 *   - LDS +1 padding to avoid bank conflicts
 *   - 2D grid for pad_data (eliminates div/mod)
 *   - hipMemsetAsync + early return (no else-branch divergence)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

namespace antenna_fft {
namespace kernels {

// ============================================================================
// Combined HIP kernel source for SpectrumProcessorROCm
//
// Kernels:
//   1. pad_data           - zero-padding n_point -> nFFT (2D grid, beam_offset)
//   2. compute_magnitudes - |FFT[i]| computation
//   3. post_kernel_one_peak  - ONE_PEAK mode (4 MaxValue per beam)
//   4. post_kernel_two_peaks - TWO_PEAKS mode (8 MaxValue per beam)
// ============================================================================
inline const char* GetSpectrumHIPKernelSource() {
    return R"HIP(

// ============================================================================
// Common structures
// ============================================================================

struct float2_t {
    float x;
    float y;
};

// MaxValue: 32 bytes (GPU-aligned), must match C++ struct
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
// 1. pad_data - Zero-padding with 2D grid (no div/mod)
// ============================================================================
//
// 2D grid: blockIdx.x = position blocks, blockIdx.y = beam index
// Output must be pre-zeroed via hipMemsetAsync before kernel launch.
// Only copies valid data (pos < count_points), no else-branch.
//
extern "C" __launch_bounds__(256)
__global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int batch_beam_count,
    unsigned int count_points,
    unsigned int nFFT,
    unsigned int beam_offset)
{
    unsigned int pos_in_fft     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_beam_idx = blockIdx.y;

    if (local_beam_idx >= batch_beam_count) return;
    if (pos_in_fft >= count_points) return;

    unsigned int global_beam_idx = local_beam_idx + beam_offset;
    unsigned int src_idx = global_beam_idx * count_points + pos_in_fft;
    unsigned int dst_idx = local_beam_idx * nFFT + pos_in_fft;
    output[dst_idx] = input[src_idx];
}

// ============================================================================
// 2. compute_magnitudes - |FFT[i]| = sqrt(re^2 + im^2)
// ============================================================================
extern "C" __launch_bounds__(256)
__global__ void compute_magnitudes(
    const float2_t* __restrict__ fft_output,
    float* __restrict__ magnitudes,
    unsigned int total_size)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_size) return;

    float2_t val = fft_output[gid];
    magnitudes[gid] = __fsqrt_rn(val.x * val.x + val.y * val.y);
}

// ============================================================================
// 3. post_kernel_one_peak - ONE_PEAK mode
// ============================================================================
//
// Searches left [0, half_range] and right [nFFT-half_range, nFFT-1] ranges,
// selects the LARGEST magnitude overall.
// Outputs 4 MaxValue per beam:
//   [0] parabolic interpolation result
//   [1] left neighbor (index-1)
//   [2] center point (max)
//   [3] right neighbor (index+1)
//
// One block per beam, 256 threads per block for parallel reduction.
// Tree-reduction O(log2(256))=8 steps instead of O(256) sequential loop.
// LDS +1 padding to avoid bank conflicts during reduction.
//
extern "C" __launch_bounds__(256)
__global__ void post_kernel_one_peak(
    const float2_t* __restrict__ fft_output,
    MaxValue_t* __restrict__ maxima_output,
    unsigned int beam_count,
    unsigned int nFFT,
    unsigned int search_range,
    float sample_rate)
{
    unsigned int beam_idx = blockIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int local_size = blockDim.x;

    if (beam_idx >= beam_count) return;

    unsigned int half_range = search_range / 2;

    // +1 padding eliminates LDS bank conflicts during tree-reduction
    __shared__ float local_left_mag[257];
    __shared__ unsigned int local_left_idx[257];
    __shared__ float local_right_mag[257];
    __shared__ unsigned int local_right_idx[257];

    float my_left_mag = -1.0f;
    unsigned int my_left_idx = 0;
    unsigned int range2_start = nFFT - half_range;
    unsigned int my_right_idx = range2_start;

    // Search left range [0, half_range)
    for (unsigned int i = lid; i < half_range; i += local_size) {
        unsigned int fft_idx = beam_idx * nFFT + i;
        float2_t val = fft_output[fft_idx];
        float mag = __fsqrt_rn(val.x * val.x + val.y * val.y);
        if (mag > my_left_mag) {
            my_left_mag = mag;
            my_left_idx = i;
        }
    }

    // Search right range [nFFT - half_range, nFFT)
    float my_right_mag = -1.0f;
    for (unsigned int i = range2_start + lid; i < nFFT; i += local_size) {
        unsigned int fft_idx = beam_idx * nFFT + i;
        float2_t val = fft_output[fft_idx];
        float mag = __fsqrt_rn(val.x * val.x + val.y * val.y);
        if (mag > my_right_mag) {
            my_right_mag = mag;
            my_right_idx = i;
        }
    }

    local_left_mag[lid] = my_left_mag;
    local_left_idx[lid] = my_left_idx;
    local_right_mag[lid] = my_right_mag;
    local_right_idx[lid] = my_right_idx;
    __syncthreads();

    // Tree-reduction: O(log2(256))=8 parallel steps instead of O(256) sequential
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_left_mag[lid + stride] > local_left_mag[lid]) {
                local_left_mag[lid] = local_left_mag[lid + stride];
                local_left_idx[lid] = local_left_idx[lid + stride];
            }
            if (local_right_mag[lid + stride] > local_right_mag[lid]) {
                local_right_mag[lid] = local_right_mag[lid + stride];
                local_right_idx[lid] = local_right_idx[lid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0: read reduction result from [0] and write output
    if (lid == 0) {
        unsigned int global_left_idx  = local_left_idx[0];
        float global_left_mag         = local_left_mag[0];
        unsigned int global_right_idx = local_right_idx[0];
        float global_right_mag        = local_right_mag[0];

        // Select LARGER of left/right
        unsigned int center_idx;
        int is_right_side;
        if (global_right_mag > global_left_mag) {
            center_idx = global_right_idx;
            is_right_side = 1;
        } else {
            center_idx = global_left_idx;
            is_right_side = 0;
        }

        unsigned int base_fft_idx = beam_idx * nFFT;
        float bin_width = sample_rate / (float)nFFT;
        unsigned int out_base = beam_idx * 4;

        // Read center and neighbors
        float2_t cv = fft_output[base_fft_idx + center_idx];
        float y_c = __fsqrt_rn(cv.x * cv.x + cv.y * cv.y);

        float2_t lv; lv.x = 0.0f; lv.y = 0.0f;
        float2_t rv; rv.x = 0.0f; rv.y = 0.0f;
        float y_l = 0.0f, y_r = 0.0f;
        int hl = (center_idx > 0) ? 1 : 0;
        int hr = (center_idx < nFFT - 1) ? 1 : 0;
        if (hl) { lv = fft_output[base_fft_idx + center_idx - 1]; y_l = __fsqrt_rn(lv.x*lv.x + lv.y*lv.y); }
        if (hr) { rv = fft_output[base_fft_idx + center_idx + 1]; y_r = __fsqrt_rn(rv.x*rv.x + rv.y*rv.y); }

        // Parabolic interpolation
        float fo = 0.0f;
        float rf = (float)center_idx * bin_width;
        if (hl && hr) {
            float denom = y_l - 2.0f*y_c + y_r;
            if (fabsf(denom) > 1e-10f) {
                fo = 0.5f*(y_l - y_r)/denom;
                if (fo < -0.5f) fo = -0.5f;
                if (fo >  0.5f) fo =  0.5f;
                rf = ((float)center_idx + fo) * bin_width;
            }
        }
        if (is_right_side) { rf = sample_rate - rf; }

        // [0] Parabolic interpolation result
        maxima_output[out_base + 0].index = center_idx;
        maxima_output[out_base + 0].real = cv.x;
        maxima_output[out_base + 0].imag = cv.y;
        maxima_output[out_base + 0].magnitude = y_c;
        maxima_output[out_base + 0].phase = atan2f(cv.y, cv.x) * 57.29577951f;
        maxima_output[out_base + 0].freq_offset = fo;
        maxima_output[out_base + 0].refined_frequency = rf;
        maxima_output[out_base + 0].pad = 0;

        // [1] Left neighbor
        float rf_l = hl ? (float)(center_idx - 1) * bin_width : 0.0f;
        if (is_right_side && hl) rf_l = sample_rate - rf_l;
        maxima_output[out_base + 1].index = hl ? center_idx - 1 : 0;
        maxima_output[out_base + 1].real = lv.x;
        maxima_output[out_base + 1].imag = lv.y;
        maxima_output[out_base + 1].magnitude = y_l;
        maxima_output[out_base + 1].phase = hl ? atan2f(lv.y, lv.x) * 57.29577951f : 0.0f;
        maxima_output[out_base + 1].freq_offset = 0.0f;
        maxima_output[out_base + 1].refined_frequency = rf_l;
        maxima_output[out_base + 1].pad = 0;

        // [2] Center point
        float rf_c = (float)center_idx * bin_width;
        if (is_right_side) rf_c = sample_rate - rf_c;
        maxima_output[out_base + 2].index = center_idx;
        maxima_output[out_base + 2].real = cv.x;
        maxima_output[out_base + 2].imag = cv.y;
        maxima_output[out_base + 2].magnitude = y_c;
        maxima_output[out_base + 2].phase = atan2f(cv.y, cv.x) * 57.29577951f;
        maxima_output[out_base + 2].freq_offset = 0.0f;
        maxima_output[out_base + 2].refined_frequency = rf_c;
        maxima_output[out_base + 2].pad = 0;

        // [3] Right neighbor
        float rf_r = hr ? (float)(center_idx + 1) * bin_width : 0.0f;
        if (is_right_side && hr) rf_r = sample_rate - rf_r;
        maxima_output[out_base + 3].index = hr ? center_idx + 1 : 0;
        maxima_output[out_base + 3].real = rv.x;
        maxima_output[out_base + 3].imag = rv.y;
        maxima_output[out_base + 3].magnitude = y_r;
        maxima_output[out_base + 3].phase = hr ? atan2f(rv.y, rv.x) * 57.29577951f : 0.0f;
        maxima_output[out_base + 3].freq_offset = 0.0f;
        maxima_output[out_base + 3].refined_frequency = rf_r;
        maxima_output[out_base + 3].pad = 0;
    }
}

// ============================================================================
// 4. post_kernel_two_peaks - TWO_PEAKS mode
// ============================================================================
//
// Searches left and right ranges independently.
// Outputs 8 MaxValue per beam:
//   [0..3] left peak (interpolated, left, center, right)
//   [4..7] right peak (interpolated, left, center, right)
//   Right peak frequencies are mirrored: sample_rate - raw_freq
//
// Tree-reduction + LDS +1 padding (same as one_peak).
//
extern "C" __launch_bounds__(256)
__global__ void post_kernel_two_peaks(
    const float2_t* __restrict__ fft_output,
    MaxValue_t* __restrict__ maxima_output,
    unsigned int beam_count,
    unsigned int nFFT,
    unsigned int search_range,
    float sample_rate)
{
    unsigned int beam_idx = blockIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int local_size = blockDim.x;

    if (beam_idx >= beam_count) return;

    unsigned int half_range = search_range / 2;

    // +1 padding eliminates LDS bank conflicts
    __shared__ float local_left_mag[257];
    __shared__ unsigned int local_left_idx[257];
    __shared__ float local_right_mag[257];
    __shared__ unsigned int local_right_idx[257];

    float my_left_mag = -1.0f;
    unsigned int my_left_idx = 0;
    unsigned int range2_start = nFFT - half_range;
    unsigned int my_right_idx = range2_start;

    // Search left range [0, half_range)
    for (unsigned int i = lid; i < half_range; i += local_size) {
        unsigned int fft_idx = beam_idx * nFFT + i;
        float2_t val = fft_output[fft_idx];
        float mag = __fsqrt_rn(val.x * val.x + val.y * val.y);
        if (mag > my_left_mag) {
            my_left_mag = mag;
            my_left_idx = i;
        }
    }

    // Search right range [nFFT - half_range, nFFT)
    float my_right_mag = -1.0f;
    for (unsigned int i = range2_start + lid; i < nFFT; i += local_size) {
        unsigned int fft_idx = beam_idx * nFFT + i;
        float2_t val = fft_output[fft_idx];
        float mag = __fsqrt_rn(val.x * val.x + val.y * val.y);
        if (mag > my_right_mag) {
            my_right_mag = mag;
            my_right_idx = i;
        }
    }

    local_left_mag[lid] = my_left_mag;
    local_left_idx[lid] = my_left_idx;
    local_right_mag[lid] = my_right_mag;
    local_right_idx[lid] = my_right_idx;
    __syncthreads();

    // Tree-reduction: O(log2(256))=8 parallel steps
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_left_mag[lid + stride] > local_left_mag[lid]) {
                local_left_mag[lid] = local_left_mag[lid + stride];
                local_left_idx[lid] = local_left_idx[lid + stride];
            }
            if (local_right_mag[lid + stride] > local_right_mag[lid]) {
                local_right_mag[lid] = local_right_mag[lid + stride];
                local_right_idx[lid] = local_right_idx[lid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0: write 8 MaxValues
    if (lid == 0) {
        unsigned int global_left_idx  = local_left_idx[0];
        unsigned int global_right_idx = local_right_idx[0];

        unsigned int base_fft_idx = beam_idx * nFFT;
        float bin_width = sample_rate / (float)nFFT;
        unsigned int out_base = beam_idx * 8;

        // --- Helper: write 4 MaxValues for one peak ---
        // We write this inline for both left (mirror=0) and right (mirror=1)

        // LEFT PEAK [0..3]
        {
            unsigned int cidx = global_left_idx;
            float2_t cv = fft_output[base_fft_idx + cidx];
            float y_c = __fsqrt_rn(cv.x * cv.x + cv.y * cv.y);
            float2_t lv; lv.x = 0.0f; lv.y = 0.0f;
            float2_t rv; rv.x = 0.0f; rv.y = 0.0f;
            float y_l = 0.0f, y_r = 0.0f;
            int hl = (cidx > 0) ? 1 : 0;
            int hr = (cidx < nFFT - 1) ? 1 : 0;
            if (hl) { lv = fft_output[base_fft_idx + cidx - 1]; y_l = __fsqrt_rn(lv.x*lv.x + lv.y*lv.y); }
            if (hr) { rv = fft_output[base_fft_idx + cidx + 1]; y_r = __fsqrt_rn(rv.x*rv.x + rv.y*rv.y); }

            float fo = 0.0f;
            float rf = (float)cidx * bin_width;
            if (hl && hr) {
                float denom = y_l - 2.0f*y_c + y_r;
                if (fabsf(denom) > 1e-10f) {
                    fo = 0.5f*(y_l - y_r)/denom;
                    if (fo < -0.5f) fo = -0.5f;
                    if (fo >  0.5f) fo =  0.5f;
                    rf = ((float)cidx + fo) * bin_width;
                }
            }

            maxima_output[out_base + 0].index = cidx;
            maxima_output[out_base + 0].real = cv.x;
            maxima_output[out_base + 0].imag = cv.y;
            maxima_output[out_base + 0].magnitude = y_c;
            maxima_output[out_base + 0].phase = atan2f(cv.y, cv.x) * 57.29577951f;
            maxima_output[out_base + 0].freq_offset = fo;
            maxima_output[out_base + 0].refined_frequency = rf;
            maxima_output[out_base + 0].pad = 0;

            float rf_l2 = hl ? (float)(cidx - 1) * bin_width : 0.0f;
            maxima_output[out_base + 1].index = hl ? cidx - 1 : 0;
            maxima_output[out_base + 1].real = lv.x;
            maxima_output[out_base + 1].imag = lv.y;
            maxima_output[out_base + 1].magnitude = y_l;
            maxima_output[out_base + 1].phase = hl ? atan2f(lv.y, lv.x) * 57.29577951f : 0.0f;
            maxima_output[out_base + 1].freq_offset = 0.0f;
            maxima_output[out_base + 1].refined_frequency = rf_l2;
            maxima_output[out_base + 1].pad = 0;

            float rf_c2 = (float)cidx * bin_width;
            maxima_output[out_base + 2].index = cidx;
            maxima_output[out_base + 2].real = cv.x;
            maxima_output[out_base + 2].imag = cv.y;
            maxima_output[out_base + 2].magnitude = y_c;
            maxima_output[out_base + 2].phase = atan2f(cv.y, cv.x) * 57.29577951f;
            maxima_output[out_base + 2].freq_offset = 0.0f;
            maxima_output[out_base + 2].refined_frequency = rf_c2;
            maxima_output[out_base + 2].pad = 0;

            float rf_r2 = hr ? (float)(cidx + 1) * bin_width : 0.0f;
            maxima_output[out_base + 3].index = hr ? cidx + 1 : 0;
            maxima_output[out_base + 3].real = rv.x;
            maxima_output[out_base + 3].imag = rv.y;
            maxima_output[out_base + 3].magnitude = y_r;
            maxima_output[out_base + 3].phase = hr ? atan2f(rv.y, rv.x) * 57.29577951f : 0.0f;
            maxima_output[out_base + 3].freq_offset = 0.0f;
            maxima_output[out_base + 3].refined_frequency = rf_r2;
            maxima_output[out_base + 3].pad = 0;
        }

        // RIGHT PEAK [4..7] — mirror frequencies: sample_rate - raw
        {
            unsigned int cidx = global_right_idx;
            float2_t cv = fft_output[base_fft_idx + cidx];
            float y_c = __fsqrt_rn(cv.x * cv.x + cv.y * cv.y);
            float2_t lv; lv.x = 0.0f; lv.y = 0.0f;
            float2_t rv; rv.x = 0.0f; rv.y = 0.0f;
            float y_l = 0.0f, y_r = 0.0f;
            int hl = (cidx > 0) ? 1 : 0;
            int hr = (cidx < nFFT - 1) ? 1 : 0;
            if (hl) { lv = fft_output[base_fft_idx + cidx - 1]; y_l = __fsqrt_rn(lv.x*lv.x + lv.y*lv.y); }
            if (hr) { rv = fft_output[base_fft_idx + cidx + 1]; y_r = __fsqrt_rn(rv.x*rv.x + rv.y*rv.y); }

            float fo = 0.0f;
            float rf = (float)cidx * bin_width;
            if (hl && hr) {
                float denom = y_l - 2.0f*y_c + y_r;
                if (fabsf(denom) > 1e-10f) {
                    fo = 0.5f*(y_l - y_r)/denom;
                    if (fo < -0.5f) fo = -0.5f;
                    if (fo >  0.5f) fo =  0.5f;
                    rf = ((float)cidx + fo) * bin_width;
                }
            }
            rf = sample_rate - rf;  // Mirror!

            maxima_output[out_base + 4].index = cidx;
            maxima_output[out_base + 4].real = cv.x;
            maxima_output[out_base + 4].imag = cv.y;
            maxima_output[out_base + 4].magnitude = y_c;
            maxima_output[out_base + 4].phase = atan2f(cv.y, cv.x) * 57.29577951f;
            maxima_output[out_base + 4].freq_offset = fo;
            maxima_output[out_base + 4].refined_frequency = rf;
            maxima_output[out_base + 4].pad = 0;

            float rf_l2 = hl ? (float)(cidx - 1) * bin_width : 0.0f;
            if (hl) rf_l2 = sample_rate - rf_l2;
            maxima_output[out_base + 5].index = hl ? cidx - 1 : 0;
            maxima_output[out_base + 5].real = lv.x;
            maxima_output[out_base + 5].imag = lv.y;
            maxima_output[out_base + 5].magnitude = y_l;
            maxima_output[out_base + 5].phase = hl ? atan2f(lv.y, lv.x) * 57.29577951f : 0.0f;
            maxima_output[out_base + 5].freq_offset = 0.0f;
            maxima_output[out_base + 5].refined_frequency = rf_l2;
            maxima_output[out_base + 5].pad = 0;

            float rf_c2 = sample_rate - (float)cidx * bin_width;
            maxima_output[out_base + 6].index = cidx;
            maxima_output[out_base + 6].real = cv.x;
            maxima_output[out_base + 6].imag = cv.y;
            maxima_output[out_base + 6].magnitude = y_c;
            maxima_output[out_base + 6].phase = atan2f(cv.y, cv.x) * 57.29577951f;
            maxima_output[out_base + 6].freq_offset = 0.0f;
            maxima_output[out_base + 6].refined_frequency = rf_c2;
            maxima_output[out_base + 6].pad = 0;

            float rf_r2 = hr ? (float)(cidx + 1) * bin_width : 0.0f;
            if (hr) rf_r2 = sample_rate - rf_r2;
            maxima_output[out_base + 7].index = hr ? cidx + 1 : 0;
            maxima_output[out_base + 7].real = rv.x;
            maxima_output[out_base + 7].imag = rv.y;
            maxima_output[out_base + 7].magnitude = y_r;
            maxima_output[out_base + 7].phase = hr ? atan2f(rv.y, rv.x) * 57.29577951f : 0.0f;
            maxima_output[out_base + 7].freq_offset = 0.0f;
            maxima_output[out_base + 7].refined_frequency = rf_r2;
            maxima_output[out_base + 7].pad = 0;
        }
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace antenna_fft

#endif  // ENABLE_ROCM
