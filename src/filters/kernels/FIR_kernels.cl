

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

