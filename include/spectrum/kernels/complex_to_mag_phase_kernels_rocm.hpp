#pragma once

/**
 * @file complex_to_mag_phase_kernels_rocm.hpp
 * @brief Standalone HIP kernel source for complex-to-magnitude+phase conversion
 *
 * Separated from fft_processor_kernels_rocm.hpp for:
 * - No pad_data kernel compiled (not needed for direct conversion)
 * - Independent HSACO disk cache key ("c2mp_kernels")
 * - Clean separation of concerns
 *
 * Kernel compiled at runtime via hiprtc.
 * Uses custom float2_t struct to avoid hiprtc built-in type issues.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

namespace fft_processor {
namespace kernels {

/**
 * @brief HIP kernel source: complex_to_mag_phase (standalone)
 *
 * Converts complex (re, im) pairs to (magnitude, phase) pairs.
 * Each thread processes one element.
 * Output is interleaved: float2_t { .x = mag, .y = phase }
 *
 * Optimizations:
 * - __launch_bounds__(BLOCK_SIZE) for better register allocation
 * - __restrict__ pointers for auto-vectorization
 * - __fsqrt_rn and __atan2f fast intrinsics
 */
inline const char* GetComplexToMagPhaseKernelSource() {
    return R"HIP(

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t {
    float x;
    float y;
};

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_mag_phase
// Converts complex data to magnitude + phase (interleaved output)
// One thread per element. 1D grid.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ input,      // Complex input: {re, im}
    float2_t* __restrict__ mag_phase,         // Interleaved output: {mag, phase}
    unsigned int total)                       // Total elements (beam_count * n_point)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

)HIP";
}

/**
 * @brief HIP kernel source: complex_to_magnitude (magnitude-only, no phase)
 *
 * Converts complex (re, im) pairs to magnitude only, multiplied by inv_n.
 * Output is plain float[] — no interleaving.
 * inv_n is computed on host: norm_coeff<0 ? 1.0f/n_point : (norm_coeff>0 ? norm_coeff : 1.0f)
 *
 * One thread per element, 1D grid.
 */
inline const char* GetComplexToMagnitudeKernelSource() {
    return R"HIP(

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t {
    float x;
    float y;
};

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_magnitude
// Converts complex data to magnitude * inv_n (float output, no phase)
// One thread per element. 1D grid.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude(
    const float2_t* __restrict__ input,   // Complex input: {re, im}
    float* __restrict__ output,           // Float output: mag * inv_n
    float inv_n,                          // Normalization factor (host-computed)
    unsigned int total)                   // Total elements (beam_count * n_point)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = __fsqrt_rn(z.x * z.x + z.y * z.y) * inv_n;
}

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_magnitude_squared   (SNR_02 — power spectrum, no sqrt)
// Converts complex data to |X|² * inv_n (for CFAR SNR-estimator).
// ~7× faster than complex_to_magnitude — no transcendental.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude_squared(
    const float2_t* __restrict__ input,   // Complex input: {re, im}
    float* __restrict__ output,           // Float output: (re² + im²) * inv_n
    float inv_n,                          // Normalization factor (host-computed)
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = (z.x * z.x + z.y * z.y) * inv_n;  // NO sqrt!
}

)HIP";
}

/**
 * @brief Combined source: complex_to_mag_phase + complex_to_magnitude
 *
 * For GpuContext::CompileModule() — compiles both kernels in one hiprtc call.
 * Used by refactored ComplexToMagPhaseROCm (Ref03).
 */
inline const char* GetCombinedC2MPKernelSource() {
    return R"HIP(

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t {
    float x;
    float y;
};

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ mag_phase,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude(
    const float2_t* __restrict__ input,
    float* __restrict__ output,
    float inv_n,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = __fsqrt_rn(z.x * z.x + z.y * z.y) * inv_n;
}

// SNR_02: square-law detector — (re² + im²) * inv_n, no sqrt.
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude_squared(
    const float2_t* __restrict__ input,
    float* __restrict__ output,
    float inv_n,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = (z.x * z.x + z.y * z.y) * inv_n;
}

)HIP";
}

}  // namespace kernels
}  // namespace fft_processor

#endif  // ENABLE_ROCM
