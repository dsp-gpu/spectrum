#pragma once

/**
 * @file fir_kernels_rocm.hpp
 * @brief HIP kernel-source для FIR-фильтра (Direct-form свёртка, complex float).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Kernel: fir_filter_cf32
 *         - 1D grid: total_points = channels × points (один поток = один выходной отсчёт)
 *         - y[ch][n] = Σ h[k] · x[ch][n-k], k=0..num_taps-1
 *         - Без __constant vs __global split: HIP L1/L2 кэш даёт сравнимую производительность
 *           с OpenCL __constant memory.
 * @note Порт из fir_kernels.hpp (OpenCL → HIP/ROCm).
 *
 * История:
 *   - Создан:  2026-02-23
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace filters {
namespace kernels {

/**
 * @brief Get FIR kernel source (HIP/ROCm)
 *
 * Single kernel handles all coefficient sizes (no __constant vs __global split).
 * HIP L1/L2 cache provides similar performance to OpenCL __constant memory.
 *
 * @return Pointer to kernel source string
 */
inline const char* GetFirDirectSource_rocm() {
  return R"HIP(

// ─────────────────────────────────────────────────────────────────────────
// Custom float2 for hiprtc compatibility
// ─────────────────────────────────────────────────────────────────────────

struct float2_t {
    float x;
    float y;
};

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ─────────────────────────────────────────────────────────────────────────
// FIR Direct-Form Convolution (HIP)
//
// Layout: [channels * points] float2_t (channel-sequential)
//   input[ch * points + n] = complex sample (ch, n)
//
// 2D grid: blockIdx.y = channel, blockIdx.x * blockDim.x + threadIdx.x = sample
// Eliminates expensive integer division/modulo (~20 cycles each)
// ─────────────────────────────────────────────────────────────────────────

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void fir_filter_cf32(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    const float* __restrict__ coeffs,
    unsigned int num_taps,
    unsigned int channels,
    unsigned int points)
{
    unsigned int n  = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ch = blockIdx.y;
    if (n >= points || ch >= channels) return;

    unsigned int base = ch * points;

    float acc_x = 0.0f;
    float acc_y = 0.0f;

    // Branch-free inner loop: limit k to valid range
    unsigned int k_max = (n + 1u < num_taps) ? n + 1u : num_taps;

    #pragma unroll 4
    for (unsigned int k = 0; k < k_max; k++) {
        float2_t x = input[base + n - k];
        float h = coeffs[k];
        acc_x += h * x.x;
        acc_y += h * x.y;
    }

    float2_t result;
    result.x = acc_x;
    result.y = acc_y;
    output[base + n] = result;
}

)HIP";
}

}  // namespace kernels
}  // namespace filters
