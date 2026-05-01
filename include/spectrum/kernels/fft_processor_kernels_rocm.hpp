#pragma once

/**
 * @file fft_processor_kernels_rocm.hpp
 * @brief HIP kernel-source string'и FFTProcessorROCm (high-level: pad/window + post-conversion).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" sources для hiprtc.
 *       Все ядра в ОДНОМ source — компилируются одним hiprtc-вызовом через
 *       GpuContext::CompileModule() (минимизирует количество HSACO-кэшей).
 * @note Состав ядер:
 *         - pad_data                     — n_point → nFFT zero-padding (batch FFT)
 *         - pad_data_windowed            — pad + window (Hann/Hamming/Blackman), SNR_02b
 *         - complex_to_mag_phase         — FFT out → interleaved {mag, phase}
 *         - complex_to_magnitude         — FFT out → |X|·inv_n (ProcessMagnitudesToGPU)
 *         - complex_to_magnitude_squared — FFT out → (re²+im²)·inv_n (SNR_02, ~7× быстрее)
 *
 * История:
 *   - Создан:  2026-02-23
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер;
 *              ранее v2 2026-04-09 добавил SNR_02 squared + SNR_02b window)
 */

#if ENABLE_ROCM

namespace fft_processor {
namespace kernels {

/**
 * @brief Combined HIP kernel source for FFTProcessorROCm
 *
 * All kernels needed by FFTProcessorROCm are in ONE source, compiled with
 * GpuContext::CompileModule() in a single hiprtc call.
 *
 * Added in v2 (SNR_02, SNR_02b):
 *   - pad_data_windowed:                Hann/Hamming/Blackman window inline
 *   - complex_to_magnitude:             magnitude-only output (no phase) — for SNR pipeline
 *   - complex_to_magnitude_squared:     |X|² output (no sqrt) — 7× faster for SNR-estimator
 */
inline const char* GetHIPKernelSource() {
    return R"HIP(

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

struct float2_t {
    float x;
    float y;
};

// ═══════════════════════════════════════════════════════════════
// Kernel: pad_data
// Pads input data from n_point to nFFT with zeros for batch FFT.
// 2D grid: blockIdx.y = beam_id — eliminates int div/mod per thread.
// Zeros in fft_input handled by hipMemsetAsync before launch.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int beam_id = blockIdx.y;
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;  // Only copy valid points; rest already zero

    output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
}

// ═══════════════════════════════════════════════════════════════
// Kernel: pad_data_windowed      (SNR_02b — калибровано Python моделью)
// pad_data + window function (Hann/Hamming/Blackman) inline.
// Applies window to n_point samples, zeros the rest [n_point..nFFT].
// One thread per output element. 2D grid: X=n_point, Y=beam.
//
// window_type: 0=None (unused here), 1=Hann, 2=Hamming, 3=Blackman
// Uses __cosf (fast intrinsic, not slow cosf).
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(256)
extern "C" __global__ void pad_data_windowed(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT,
    int window_type)
{
    unsigned int beam_id = blockIdx.y;
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;  // Only copy valid points; rest already zero by memset

    float2_t z = input[beam_id * n_point + pos];

    // Apply window (inline, no LUT — registers cheaper than memory loads on GPU)
    float w = 1.0f;
    if (window_type == 1) {          // Hann
        float theta = 2.0f * M_PI_F * (float)pos / (float)(n_point - 1);
        w = 0.5f * (1.0f - __cosf(theta));
    } else if (window_type == 2) {   // Hamming
        float theta = 2.0f * M_PI_F * (float)pos / (float)(n_point - 1);
        w = 0.54f - 0.46f * __cosf(theta);
    } else if (window_type == 3) {   // Blackman
        float theta = 2.0f * M_PI_F * (float)pos / (float)(n_point - 1);
        w = 0.42f - 0.5f * __cosf(theta) + 0.08f * __cosf(2.0f * theta);
    }

    float2_t out;
    out.x = z.x * w;
    out.y = z.y * w;
    output[beam_id * nFFT + pos] = out;
}

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_mag_phase
// Converts complex FFT output to magnitude + phase (interleaved).
// One thread per element. Used by ProcessMagPhase.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(256)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float2_t* __restrict__ mag_phase,   // interleaved: .x=mag, .y=phase
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = fft_output[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_magnitude   (for ProcessMagnitudesToGPU)
// Converts complex to magnitude |X| * inv_n (float output, no phase).
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(256)
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

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_magnitude_squared   (SNR_02 — для SNR-estimator)
// Converts complex to |X|² * inv_n (power spectrum, NO sqrt).
// ~7× faster than complex_to_magnitude (no transcendental).
// Used when squared=true in MagnitudeOp::Execute().
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(256)
extern "C" __global__ void complex_to_magnitude_squared(
    const float2_t* __restrict__ input,
    float* __restrict__ output,
    float inv_n,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = (z.x * z.x + z.y * z.y) * inv_n;  // NO sqrt!
}

)HIP";
}

}  // namespace kernels
}  // namespace fft_processor

#endif  // ENABLE_ROCM
