

// ─── Configuration ──────────────────────────────────────────────────────────
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t { float x; float y; };

// ═════════════════════════════════════════════════════════════════════════════
// 1D Scalar Kalman Filter
// Constant-state model: x_pred = x_hat
// Re and Im filtered independently
// Optimizations:
//   - __frcp_rn() for fast reciprocal instead of full division (~2x faster)
//   - K = P_pred * rcp(P_pred + R) — single FMA-friendly operation
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void kalman_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    float Q,
    float R,
    float x0,
    float P0)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;

    // Independent scalar Kalman states for Re and Im
    float x_re = x0, x_im = x0;
    float P_re = P0, P_im = P0;

    for (unsigned int n = 0; n < points; n++) {
        float2_t z = in[base + n];

        // ── Re ──────────────────────────────────────────
        float P_pred_re = P_re + Q;
        // K = P_pred / (P_pred + R) via fast reciprocal
        float K_re = P_pred_re * __frcp_rn(P_pred_re + R);
        x_re = x_re + K_re * (z.x - x_re);
        P_re = (1.0f - K_re) * P_pred_re;

        // ── Im ──────────────────────────────────────────
        float P_pred_im = P_im + Q;
        float K_im = P_pred_im * __frcp_rn(P_pred_im + R);
        x_im = x_im + K_im * (z.y - x_im);
        P_im = (1.0f - K_im) * P_pred_im;

        out[base + n].x = x_re;
        out[base + n].y = x_im;
    }
}

