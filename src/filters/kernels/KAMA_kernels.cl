

// ─── Configuration ──────────────────────────────────────────────────────────
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ⚠️ ВАЖНО: N_WINDOW — compile-time константа, устанавливается через hiprtc define
//    (-DN_WINDOW=<er_period>) из KaufmanFilterROCm::CompileKernel(n_window).
//    НЕ ЗАМЕНЯТЬ на фиксированное значение (128 и т.п.)!
//    Причина: thread-local массив фиксированного размера при BLOCK_SIZE=256
//    вызывает scratch overflow (256 KB/блок) и SIGSEGV на gfx1201 RDNA4.
//    Значение по умолчанию используется только при прямой компиляции вне фильтра.
#ifndef N_WINDOW
#define N_WINDOW 10
#endif

struct float2_t { float x; float y; };

// ═════════════════════════════════════════════════════════════════════════════
// KAMA — Kaufman Adaptive Moving Average
// ER = Direction / Volatility → SC = (ER*(fast-slow)+slow)^2 → update
// Optimizations:
//   - %N replaced with conditional branch (3 locations, ~60 cycles saved/iter)
//   - dir/vol division → __frcp_rn() fast reciprocal
//   - prev_idx computed via simple subtraction + conditional
// ═════════════════════════════════════════════════════════════════════════════
extern "C" __global__ __launch_bounds__(BLOCK_SIZE)
void kaufman_kernel(
    const float2_t* __restrict__ in,
          float2_t* __restrict__ out,
    unsigned int channels,
    unsigned int points,
    unsigned int N,
    float fast_sc,
    float slow_sc)
{
    unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const unsigned int base = ch * points;
    const float eps = 1e-8f;
    const float sc_diff = fast_sc - slow_sc;

    // Passthrough first N samples
    for (unsigned int i = 0; i < N && i < points; i++) {
        out[base + i] = in[base + i];
    }
    if (points <= N) return;

    // ── Sliding window for Volatility: O(1) per sample instead of O(N) ──
    float delta_re[N_WINDOW], delta_im[N_WINDOW];  // size = er_period, set via -DN_WINDOW
    float vol_re = 0.0f, vol_im = 0.0f;
    unsigned int head = 0;

    for (unsigned int i = 0; i < N; i++) {
        float dr = fabsf(in[base + i + 1].x - in[base + i].x);
        float di = fabsf(in[base + i + 1].y - in[base + i].y);
        delta_re[i] = dr;
        delta_im[i] = di;
        vol_re += dr;
        vol_im += di;
    }

    float2_t kama = in[base + N - 1];

    for (unsigned int n = N; n < points; n++) {
        float2_t x = in[base + n];

        // 1. Direction
        float dir_re = fabsf(x.x - in[base + n - N].x);
        float dir_im = fabsf(x.y - in[base + n - N].y);

        // 2. Volatility: sliding window (vol_re, vol_im)

        // 3. ER — branchless
        float er_re = dir_re * __frcp_rn(vol_re + eps);
        float er_im = dir_im * __frcp_rn(vol_im + eps);

        // 4. SC = (ER*(fast-slow)+slow)^2
        float sc_re = er_re * sc_diff + slow_sc;
        float sc_im = er_im * sc_diff + slow_sc;
        sc_re *= sc_re;
        sc_im *= sc_im;

        // 5. Update KAMA
        kama.x = kama.x + sc_re * (x.x - kama.x);
        kama.y = kama.y + sc_im * (x.y - kama.y);
        out[base + n] = kama;

        // 6. Sliding window update
        if (n + 1 < points) {
            float new_dr = fabsf(in[base + n + 1].x - in[base + n].x);
            float new_di = fabsf(in[base + n + 1].y - in[base + n].y);
            vol_re += new_dr - delta_re[head];
            vol_im += new_di - delta_im[head];
            delta_re[head] = new_dr;
            delta_im[head] = new_di;
            if (++head >= N) head = 0;
        }
    }
}

