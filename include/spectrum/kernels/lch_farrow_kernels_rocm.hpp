#pragma once

/**
 * @file lch_farrow_kernels_rocm.hpp
 * @brief HIP kernel source for LchFarrowROCm (fractional delay, Lagrange 48x5)
 *
 * Contains:
 * - Philox-2x32-10 PRNG (counter-based, HIP-compatible)
 * - lch_farrow_delay kernel: fractional delay + optional Gaussian noise
 *
 * Kernels compiled at runtime via hiprtc.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

namespace lch_farrow {
namespace kernels {

/**
 * @brief HIP kernel source for LchFarrow fractional delay
 *
 * lch_farrow_delay:
 *   Input:  float2_t* (complex signal), N = antennas * points
 *   Output: float2_t* (delayed signal), N = antennas * points
 *   Params: lagrange_matrix (48x5=240 floats), delay_us (per antenna)
 *
 * Algorithm (DelayedFormSignal_Kernel_CORRECT):
 *   read_pos = sample_id - delay_us[antenna] * 1e-6 * sample_rate
 *   center = floor(read_pos), frac = read_pos - center
 *   row = (uint)(frac * 48) % 48
 *   output[n] = sum(L[row][k] * input[center-1+k], k=0..4)
 */
inline const char* GetLchFarrowKernelSource() {
    return R"HIP(

struct float2_t {
    float x;
    float y;
};

// ═══════════════════════════════════════════════════════════════════════
// Philox-2x32-10: counter-based PRNG
// ═══════════════════════════════════════════════════════════════════════

struct uint2_t {
    unsigned int x;
    unsigned int y;
};

__device__ uint2_t philox2x32_round(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_M = 0xD2511F53u;
    unsigned int lo = ctr.x * PHILOX_M;
    unsigned int hi = __umulhi(ctr.x, PHILOX_M);
    uint2_t result;
    result.x = hi ^ key ^ ctr.y;
    result.y = lo;
    return result;
}

__device__ uint2_t philox2x32_10(uint2_t ctr, unsigned int key) {
    const unsigned int PHILOX_BUMP = 0x9E3779B9u;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key); key += PHILOX_BUMP;
    ctr = philox2x32_round(ctr, key);
    return ctr;
}

// ═══════════════════════════════════════════════════════════════════════
// LCH Farrow: fractional delay kernel (Lagrange 48x5)
// ═══════════════════════════════════════════════════════════════════════

extern "C" __launch_bounds__(256)
__global__ void lch_farrow_delay(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    const float* __restrict__ lagrange_matrix,
    const float* __restrict__ delay_us,
    unsigned int antennas,
    unsigned int points,
    float sample_rate,
    float noise_amplitude,
    float norm_val,
    unsigned int noise_seed)
{
    // 2D grid: Y=antenna, X=sample (eliminates div/mod ~40 cycles/thread)
    unsigned int antenna_id = blockIdx.y;
    unsigned int sample_id  = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_id >= points) return;
    unsigned int gid = antenna_id * points + sample_id;

    // Delay in samples
    float delay_samples = delay_us[antenna_id] * 1e-6f * sample_rate;
    // Snap to nearest integer if close (float32 precision fix for integer delays)
    float delay_rounded = floorf(delay_samples + 0.5f);
    float delay_diff = delay_samples - delay_rounded;
    if (delay_diff < 0.0f) delay_diff = -delay_diff;
    if (delay_diff < 0.002f) delay_samples = delay_rounded;

    float read_pos = (float)sample_id - delay_samples;

    // Before signal start -> zero
    if (read_pos < 0.0f) {
        float2_t zero;
        zero.x = 0.0f;
        zero.y = 0.0f;
        output[gid] = zero;
        return;
    }

    // center = floor(read_pos), frac = read_pos - center
    int center = (int)floorf(read_pos);
    float frac = read_pos - (float)center;
    unsigned int row = ((unsigned int)(frac * 48.0f)) % 48u;

    // 5 Lagrange coefficients
    float L0 = lagrange_matrix[row * 5u + 0u];
    float L1 = lagrange_matrix[row * 5u + 1u];
    float L2 = lagrange_matrix[row * 5u + 2u];
    float L3 = lagrange_matrix[row * 5u + 3u];
    float L4 = lagrange_matrix[row * 5u + 4u];

    // Read 5 input samples around center (center-1 .. center+3)
    unsigned int base = antenna_id * points;

    float2_t s0, s1, s2, s3, s4;
    float2_t zero_val;
    zero_val.x = 0.0f;
    zero_val.y = 0.0f;

    int idx0 = center - 1;
    int idx1 = center;
    int idx2 = center + 1;
    int idx3 = center + 2;
    int idx4 = center + 3;

    s0 = (idx0 >= 0 && idx0 < (int)points) ? input[base + (unsigned int)idx0] : zero_val;
    s1 = (idx1 >= 0 && idx1 < (int)points) ? input[base + (unsigned int)idx1] : zero_val;
    s2 = (idx2 >= 0 && idx2 < (int)points) ? input[base + (unsigned int)idx2] : zero_val;
    s3 = (idx3 >= 0 && idx3 < (int)points) ? input[base + (unsigned int)idx3] : zero_val;
    s4 = (idx4 >= 0 && idx4 < (int)points) ? input[base + (unsigned int)idx4] : zero_val;

    // 5-point Lagrange interpolation
    float2_t result;
    result.x = L0 * s0.x + L1 * s1.x + L2 * s2.x + L3 * s3.x + L4 * s4.x;
    result.y = L0 * s0.y + L1 * s1.y + L2 * s2.y + L3 * s3.y + L4 * s4.y;

    // Optional noise (Philox + Box-Muller)
    if (noise_amplitude > 0.0f) {
        uint2_t n_ctr;
        n_ctr.x = gid;
        n_ctr.y = noise_seed;
        uint2_t n_rnd = philox2x32_10(n_ctr, 0xCD9E8D57u);

        float u1 = (float)(n_rnd.x) / 4294967296.0f + 1e-10f;
        float u2 = (float)(n_rnd.y) / 4294967296.0f;

        float r = __fsqrt_rn(-2.0f * __logf(u1));
        float theta = 2.0f * 3.14159265358979323846f * u2;
        float sin_t, cos_t;
        __sincosf(theta, &sin_t, &cos_t);

        float noise_re = noise_amplitude * norm_val * r * cos_t;
        float noise_im = noise_amplitude * norm_val * r * sin_t;

        result.x += noise_re;
        result.y += noise_im;
    }

    output[gid] = result;
}

)HIP";
}

}  // namespace kernels
}  // namespace lch_farrow

#endif  // ENABLE_ROCM
