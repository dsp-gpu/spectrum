#pragma once

/**
 * @file test_kaufman_rocm.hpp
 * @brief ROCm tests for KaufmanFilterROCm (KAMA)
 *
 * MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. gpu_vs_cpu: 8ch x 4096pts, random signal
 *   2. trend: ER~1, fast tracking
 *   3. noise: ER~0, slow/frozen KAMA
 *   4. adaptive_transition: trend -> noise -> trend
 *   5. step_demo: step + trend-noise-step demo (120pts, table output)
 *
 * IMPORTANT: Compiles ONLY with ENABLE_ROCM=1.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01 (migrated 2026-03-23)
 */

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <random>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM

#include <spectrum/filters/kaufman_filter_rocm.hpp>
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"

// test_utils — unified test infrastructure
#include "modules/test_utils/test_utils.hpp"

namespace test_kaufman_rocm {

using namespace filters;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// ═══════════════════════════════════════════════════════════════════════════
// run() — TestRunner (functional style)
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  int gpu_id = 0;

  // Check for ROCm devices
  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    auto& con = ConsoleOutput::GetInstance();
    con.Print(gpu_id, "KAMA[ROCm]", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  KaufmanFilterROCm kauf(&backend);

  TestRunner runner(&backend, "KAMA[ROCm]", gpu_id);

  // ── Test 1: GPU vs CPU — random complex signal ────────────────────

  runner.test("gpu_vs_cpu", [&]() {
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 8;
    const uint32_t points   = 4096;
    const size_t   total    = static_cast<size_t>(channels) * points;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> signal(total);
    for (size_t i = 0; i < total; ++i)
      signal[i] = {dist(rng), dist(rng)};

    auto cpu_result = kauf.ProcessCpu(signal, channels, points);
    auto gpu_result = kauf.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "kama_8ch");
  });

  // ── Test 2: Trend signal — ER~1, fast tracking ───────────────────

  runner.test("trend", [&]() {
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 256;

    // Linear trend: x[n] = n * 0.1
    std::vector<std::complex<float>> signal(points);
    for (uint32_t n = 0; n < points; ++n)
      signal[n] = {static_cast<float>(n) * 0.1f, 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // After warmup (n > 20), KAMA should track trend closely
    float max_lag = 0.0f;
    for (uint32_t n = 20; n < points; ++n) {
      float lag = std::abs(result[n].real() - signal[n].real());
      max_lag = std::max(max_lag, lag);
    }

    return ScalarAbsError(max_lag, 0.0f, 2.0, "max_lag");
  });

  // ── Test 3: Noise signal — ER~0, KAMA almost frozen ──────────────

  runner.test("noise", [&]() -> TestResult {
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 512;

    std::mt19937 rng(99);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<std::complex<float>> signal(points);
    for (uint32_t n = 0; n < points; ++n)
      signal[n] = {noise(rng), 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // After warmup: std(KAMA) should be much smaller than std(signal)
    float mean_sig = 0.0f, mean_kama = 0.0f;
    uint32_t count = 0;
    for (uint32_t n = 100; n < points; ++n) {
      mean_sig  += signal[n].real();
      mean_kama += result[n].real();
      ++count;
    }
    mean_sig  /= count;
    mean_kama /= count;

    float var_sig = 0.0f, var_kama = 0.0f;
    for (uint32_t n = 100; n < points; ++n) {
      float ds = signal[n].real() - mean_sig;
      float dk = result[n].real() - mean_kama;
      var_sig  += ds * ds;
      var_kama += dk * dk;
    }
    float std_sig  = std::sqrt(var_sig / count);
    float std_kama = std::sqrt(var_kama / count);

    float ratio = std_kama / (std_sig + 1e-10f);

    TestResult tr{"noise"};
    tr.add(ratio < 0.2f
        ? PassResult("std_ratio", ratio, 0.2,
                      "std_sig=" + std::to_string(std_sig) + " std_kama=" + std::to_string(std_kama))
        : FailResult("std_ratio", ratio, 0.2,
                      "ratio>0.2, std_sig=" + std::to_string(std_sig) + " std_kama=" + std::to_string(std_kama)));
    return tr;
  });

  // ── Test 4: Adaptive transition — trend -> noise -> trend ────────

  runner.test("adaptive_transition", [&]() -> TestResult {
    kauf.SetParams(10, 2, 30);

    const uint32_t channels = 1;
    const uint32_t points   = 2048;

    std::mt19937 rng(555);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<std::complex<float>> signal(points);

    // Phase 1 [0..511]: linear trend (slope=0.01)
    for (uint32_t n = 0; n < 512; ++n)
      signal[n] = {static_cast<float>(n) * 0.01f, 0.0f};

    // Phase 2 [512..1023]: white noise
    for (uint32_t n = 512; n < 1024; ++n)
      signal[n] = {noise(rng), 0.0f};

    // Phase 3 [1024..2047]: linear trend continuing
    for (uint32_t n = 1024; n < 2048; ++n)
      signal[n] = {static_cast<float>(n - 1024) * 0.01f + 5.0f, 0.0f};

    auto result = kauf.ProcessCpu(signal, channels, points);

    // Check Phase 1 (end): KAMA tracks trend
    float val_500 = result[500].real();
    float exp_500 = 500.0f * 0.01f;

    // Check Phase 2: KAMA is stable (small variation during noise)
    float delta_noise = std::abs(result[1020].real() - result[520].real());

    // Check Phase 3 (end): KAMA recovers tracking
    float val_2040 = result[2040].real();
    float exp_2040 = (2040.0f - 1024.0f) * 0.01f + 5.0f;

    TestResult tr{"adaptive_transition"};
    tr.add(ScalarAbsError(val_500, exp_500, 0.5, "trend_track"));
    tr.add(ScalarAbsError(delta_noise, 0.0f, 3.0, "noise_stability"));
    tr.add(ScalarAbsError(val_2040, exp_2040, 1.0, "trend_recovery"));
    return tr;
  });

  // ── Test 5: Step + trend-noise-step demo (120pts, table output) ──

  runner.test("step_demo", [&]() -> TestResult {
    const uint32_t channels = 1;
    const uint32_t points   = 120;

    // ── Signal A: clean step (20 zeros / 50 ones / 50 zeros) ──
    std::vector<std::complex<float>> sig_step(points, {0.0f, 0.0f});
    for (uint32_t i = 20; i < 70; ++i)
      sig_step[i] = {1.0f, 0.0f};

    kauf.SetParams(10, 2, 30);

    auto out_step = kauf.ProcessCpu(sig_step, channels, points);

    auto& con = ConsoleOutput::GetInstance();
    char buf[128];

    // ── Print step table ──
    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "--- [KAMA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  t  | input | KAMA(10) | note");
    con.Print(gpu_id, "KAMA[ROCm]",
        " ----+-------+----------+-----------------");

    uint32_t sample_t[] = {0,5,10,15,20,22,25,28,30,35,40,50,65,
                           70,72,75,80,90,110,119};
    for (uint32_t i = 0; i < sizeof(sample_t)/sizeof(uint32_t); ++i) {
      uint32_t t = sample_t[i];
      const char* note = "";
      if (t == 20) note = "<-- step up";
      if (t == 70) note = "<-- step down";
      std::snprintf(buf, sizeof(buf), " %3u |  %.1f  |  %6.4f  | %s",
          t, sig_step[t].real(), out_step[t].real(), note);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    }

    // ── Signal B: trend-noise-step (adaptivity demo) ──
    std::vector<std::complex<float>> sig_tnt(120, {0.0f, 0.0f});
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.2f);

    // Phase 1 [0..39]: clean trend 0->1
    for (uint32_t n = 0; n < 40; ++n)
      sig_tnt[n] = {static_cast<float>(n) * 0.025f, 0.0f};

    // Phase 2 [40..79]: white noise
    for (uint32_t n = 40; n < 80; ++n)
      sig_tnt[n] = {noise(rng), 0.0f};

    // Phase 3 [80..119]: step = 1.0 + small noise
    for (uint32_t n = 80; n < 120; ++n)
      sig_tnt[n] = {1.0f + noise(rng) * 0.05f, 0.0f};

    auto out_tnt = kauf.ProcessCpu(sig_tnt, 1, 120);

    // ── Print adaptive table ──
    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "--- [KAMA Demo] Trend->Noise->Step (adaptivity) ---");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  Phase        | Signal   | ER    | KAMA behavior");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [0..39]  trend| 0->1     | ~1.0  | fast tracking");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [40..79] noise| sigma=0.2| ~0.0  | almost frozen");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  [80..119] step| 1.0+eps  | ~1.0  | fast recovery");

    con.Print(gpu_id, "KAMA[ROCm]", "");
    con.Print(gpu_id, "KAMA[ROCm]",
        "  t  | input  | KAMA    | phase");
    con.Print(gpu_id, "KAMA[ROCm]",
        " ----+--------+---------+------");

    uint32_t demo_t[] = {0,10,20,30,39,40,50,60,70,79,80,85,90,100,115,119};
    for (uint32_t i = 0; i < sizeof(demo_t)/sizeof(uint32_t); ++i) {
      uint32_t t = demo_t[i];
      const char* phase = (t < 40) ? "trend" :
                           (t < 80) ? "noise" : "step ";
      std::snprintf(buf, sizeof(buf), " %3u | %6.3f | %6.3f  | %s",
          t, sig_tnt[t].real(), out_tnt[t].real(), phase);
      con.Print(gpu_id, "KAMA[ROCm]", std::string(buf));
    }

    // ── Checks ──
    TestResult tr{"step_demo"};

    // Step plateau: KAMA should reach ~1.0 at t=55
    tr.add(ScalarAbsError(out_step[55].real(), 1.0f, 0.01, "step_plateau_t55"));

    // Trend tracking at t=38
    tr.add(ScalarAbsError(out_tnt[38].real(), 0.950f, 0.1, "trend_t38"));

    // Noise stability: KAMA variation within noise phase (t=55..79) should be small
    float delta_noise = std::abs(out_tnt[79].real() - out_tnt[55].real());
    tr.add(ScalarAbsError(delta_noise, 0.0f, 0.5, "noise_delta"));

    // GPU vs CPU (step signal)
    auto gpu_result = kauf.ProcessFromCPU(sig_step, channels, points);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, points);

    tr.add(AbsError(gpu_data.data(), out_step.data(), points,
                    tolerance::kFilter, "gpu_vs_cpu_step"));

    return tr;
  });

  // ── Summary ─────────────────────────────────────────────────────

  runner.print_summary();
}

}  // namespace test_kaufman_rocm

#else  // !ENABLE_ROCM

namespace test_kaufman_rocm {

inline void run() {
  // SKIPPED: ENABLE_ROCM not defined
}

}  // namespace test_kaufman_rocm

#endif  // ENABLE_ROCM
