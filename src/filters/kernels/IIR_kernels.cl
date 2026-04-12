

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
// IIR Biquad Cascade - Direct Form II Transposed (HIP)
//
// One thread = one channel (all samples, all sections)
// Complex signal (float2_t), real biquad coefficients
//
// Direct Form II Transposed:
//   y[n] = b0*x[n] + w1
//   w1   = b1*x[n] - a1*y[n] + w2
//   w2   = b2*x[n] - a2*y[n]
// ─────────────────────────────────────────────────────────────────────────

extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void iir_biquad_cascade_cf32(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    const float* __restrict__ sos_matrix,
    unsigned int num_sections,
    unsigned int channels,
    unsigned int points)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    unsigned int base = ch * points;

    for (unsigned int sec = 0; sec < num_sections; sec++) {

        // Load biquad coefficients for this section
        float b0 = sos_matrix[sec * 5u + 0u];
        float b1 = sos_matrix[sec * 5u + 1u];
        float b2 = sos_matrix[sec * 5u + 2u];
        float a1 = sos_matrix[sec * 5u + 3u];
        float a2 = sos_matrix[sec * 5u + 4u];

        // Source pointer: first section reads input, subsequent read output
        // Extracted outside points-loop to eliminate per-sample branch
        const float2_t* src = (sec == 0u) ? input : output;

        // State (Direct Form II Transposed)
        float w1_x = 0.0f, w1_y = 0.0f;
        float w2_x = 0.0f, w2_y = 0.0f;

        for (unsigned int n = 0; n < points; n++) {
            float2_t x = src[base + n];

            // DFII Transposed
            float2_t y;
            y.x = b0 * x.x + w1_x;
            y.y = b0 * x.y + w1_y;

            w1_x = b1 * x.x - a1 * y.x + w2_x;
            w1_y = b1 * x.y - a1 * y.y + w2_y;

            w2_x = b2 * x.x - a2 * y.x;
            w2_y = b2 * x.y - a2 * y.y;

            output[base + n] = y;
        }
    }
}

