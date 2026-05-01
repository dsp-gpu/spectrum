#pragma once

/**
 * @file moving_average_kernels_rocm.hpp
 * @brief HIP kernel-source для Moving Average фильтров: SMA / EMA / MMA / DEMA / TEMA.
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для hiprtc.
 *       Все ядра: 1D grid, один поток на канал, sequential loop по points.
 *       Input/output — complex float (float2_t = {re, im}).
 * @note Типы:
 *         - SMA  — Simple MA (ring buffer, equal weights 1/N)
 *         - EMA  — Exponential MA (alpha = 2/(N+1))
 *         - MMA  — Modified MA / Wilder's (alpha = 1/N)
 *         - DEMA — Double EMA (2·EMA1 - EMA2)
 *         - TEMA — Triple EMA (3·EMA1 - 3·EMA2 + EMA3)
 * @note ⚠️ N_WINDOW = compile-time константа через `-DN_WINDOW=<window_size>` из
 *       MovingAverageFilterROCm::CompileKernels(sma_window). Используется только
 *       для SMA ring buffer. НЕ заменять на фиксированное! Причина: thread-local
 *       ring[N] при BLOCK_SIZE=256 → scratch overflow → SIGSEGV на RDNA4.
 *
 * История:
 *   - Создан:  2026-03-01
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace filters {
namespace kernels {

inline const char* GetMovingAverageSource_rocm() {
  return R"HIP(

// ─── Configuration ──────────────────────────────────────────────────────────
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ⚠️ ВАЖНО: N_WINDOW — compile-time константа, устанавливается через hiprtc define
//    (-DN_WINDOW=<window_size>) из MovingAverageFilterROCm::CompileKernels(sma_window).
//    Используется только SMA ring buffer. НЕ ЗАМЕНЯТЬ на фиксированное значение!
//    Причина: thread-local ring[N] при BLOCK_SIZE=256 → scratch overflow → SIGSEGV на RDNA4.
#ifndef N_WINDOW
#define N_WINDOW 10
#endif

// ─── Types ───────────────────────────────────────────────────────────────────
struct float2_t { float x; float y; };

// ═════════════════════════════════════════════════════════════════════════════
// SMA — Simple Moving Average
// Ring buffer in thread-local memory (size = N_WINDOW, set via hiprtc -DN_WINDOW)
// Optimizations:
//   - inv_N precomputed on host, passed as param (avoids per-sample division)
//   - head wraparound via conditional branch (avoids expensive %N ~20 cycles)
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void sma_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    unsigned int N,
    float inv_N)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    float2_t ring[N_WINDOW];  // size = window_size, set via -DN_WINDOW
    float sum_x = 0.0f, sum_y = 0.0f;
    unsigned int head = 0;

    for (unsigned int n = 0; n < points; n++) {
        float2_t x = in[base + n];
        if (n < N) {
            ring[n] = x;
            sum_x += x.x;
            sum_y += x.y;
            // Warmup phase: average over n+1 samples
            float inv = __frcp_rn((float)(n + 1));
            out[base + n].x = sum_x * inv;
            out[base + n].y = sum_y * inv;
        } else {
            float2_t old_val = ring[head];
            ring[head] = x;
            // Conditional branch instead of %N (~20 cycles savings per iteration)
            if (++head >= N) head = 0;
            sum_x += x.x - old_val.x;
            sum_y += x.y - old_val.y;
            out[base + n].x = sum_x * inv_N;
            out[base + n].y = sum_y * inv_N;
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// EMA — Exponential Moving Average
// alpha = 2/(N+1), state: 1x float2_t
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void ema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float one_minus_alpha = 1.0f - alpha;

    float2_t state = in[base];
    out[base] = state;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        state.x = alpha * x.x + one_minus_alpha * state.x;
        state.y = alpha * x.y + one_minus_alpha * state.y;
        out[base + n] = state;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// MMA — Modified (Wilder's) Moving Average
// alpha = 1/N (slower than EMA)
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void mma_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float one_minus_alpha = 1.0f - alpha;

    float2_t state = in[base];
    out[base] = state;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        state.x = alpha * x.x + one_minus_alpha * state.x;
        state.y = alpha * x.y + one_minus_alpha * state.y;
        out[base + n] = state;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DEMA — Double EMA
// DEMA[n] = 2*EMA1[n] - EMA2[n]
// State: 2x float2_t (ema1, ema2)
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void dema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float one_minus_alpha = 1.0f - alpha;

    float2_t x0 = in[base];
    float2_t ema1 = x0, ema2 = x0;
    out[base].x = 2.0f * ema1.x - ema2.x;
    out[base].y = 2.0f * ema1.y - ema2.y;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        ema1.x = alpha * x.x    + one_minus_alpha * ema1.x;
        ema1.y = alpha * x.y    + one_minus_alpha * ema1.y;
        ema2.x = alpha * ema1.x + one_minus_alpha * ema2.x;
        ema2.y = alpha * ema1.y + one_minus_alpha * ema2.y;
        out[base + n].x = 2.0f * ema1.x - ema2.x;
        out[base + n].y = 2.0f * ema1.y - ema2.y;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// TEMA — Triple EMA
// TEMA[n] = 3*EMA1[n] - 3*EMA2[n] + EMA3[n]
// State: 3x float2_t
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void tema_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float alpha)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float one_minus_alpha = 1.0f - alpha;

    float2_t x0 = in[base];
    float2_t ema1 = x0, ema2 = x0, ema3 = x0;
    out[base].x = 3.0f * ema1.x - 3.0f * ema2.x + ema3.x;
    out[base].y = 3.0f * ema1.y - 3.0f * ema2.y + ema3.y;

    for (unsigned int n = 1; n < points; n++) {
        float2_t x = in[base + n];
        ema1.x = alpha * x.x    + one_minus_alpha * ema1.x;
        ema1.y = alpha * x.y    + one_minus_alpha * ema1.y;
        ema2.x = alpha * ema1.x + one_minus_alpha * ema2.x;
        ema2.y = alpha * ema1.y + one_minus_alpha * ema2.y;
        ema3.x = alpha * ema2.x + one_minus_alpha * ema3.x;
        ema3.y = alpha * ema2.y + one_minus_alpha * ema3.y;
        out[base + n].x = 3.0f * ema1.x - 3.0f * ema2.x + ema3.x;
        out[base + n].y = 3.0f * ema1.y - 3.0f * ema2.y + ema3.y;
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace filters
