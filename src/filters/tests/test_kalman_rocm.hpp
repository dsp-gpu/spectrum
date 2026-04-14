#pragma once

/**
 * @file test_kalman_rocm.hpp
 * @brief ROCm tests for KalmanFilterROCm (1D scalar Kalman)
 *
 * MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. gpu_vs_cpu: 8ch x 4096pts, random signal
 *   2. const_signal: constant + noise, convergence test
 *   3. channel_independence: 256ch, unique constants
 *   4. step_response: step at n=512
 *   5. lfm_radar_demo: 5 targets, beat signal + AWGN + Kalman
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

#include <spectrum/filters/kalman_filter_rocm.hpp>
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_core.hpp"

// test_utils — unified test infrastructure
#include "modules/test_utils/test_utils.hpp"

namespace test_kalman_rocm {

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
    con.Print(gpu_id, "Kalman[ROCm]", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  KalmanFilterROCm kalman(&backend);

  TestRunner runner(&backend, "Kalman[ROCm]", gpu_id);

  // ── Test 1: GPU vs CPU — random complex signal ────────────────────

  runner.test("gpu_vs_cpu", [&]() {
    kalman.SetParams(0.1f, 25.0f, 0.0f, 25.0f);

    const uint32_t channels = 8;
    const uint32_t points   = 4096;
    const size_t   total    = static_cast<size_t>(channels) * points;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<std::complex<float>> signal(total);
    for (size_t i = 0; i < total; ++i)
      signal[i] = {dist(rng), dist(rng)};

    auto cpu_result = kalman.ProcessCpu(signal, channels, points);
    auto gpu_result = kalman.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "kalman_8ch");
  });

  // ── Test 2: Constant signal + noise → convergence ────────────────

  runner.test("const_signal", [&]() -> TestResult {
    kalman.SetParams(0.01f, 25.0f, 0.0f, 25.0f);

    const uint32_t channels = 8;
    const uint32_t points   = 1024;
    const size_t   total    = static_cast<size_t>(channels) * points;
    const float    const_val = 100.0f;
    const float    noise_sigma = 5.0f;

    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, noise_sigma);

    std::vector<std::complex<float>> signal(total);
    for (uint32_t ch = 0; ch < channels; ++ch) {
      size_t base = static_cast<size_t>(ch) * points;
      for (uint32_t n = 0; n < points; ++n) {
        signal[base + n] = {const_val + noise(rng),
                             const_val + noise(rng)};
      }
    }

    auto cpu_result = kalman.ProcessCpu(signal, channels, points);

    // Check: after 200 samples, filtered should be close to const_val
    // and RMS(filtered - const_val) << RMS(noisy - const_val)
    float rms_raw = 0.0f, rms_flt = 0.0f;
    uint32_t count = 0;
    for (uint32_t ch = 0; ch < channels; ++ch) {
      size_t base = static_cast<size_t>(ch) * points;
      for (uint32_t n = 200; n < points; ++n) {
        float err_raw = signal[base + n].real() - const_val;
        float err_flt = cpu_result[base + n].real() - const_val;
        rms_raw += err_raw * err_raw;
        rms_flt += err_flt * err_flt;
        ++count;
      }
    }
    rms_raw = std::sqrt(rms_raw / count);
    rms_flt = std::sqrt(rms_flt / count);

    float improvement_db = 20.0f * std::log10(rms_raw / (rms_flt + 1e-10f));

    TestResult tr{"const_signal"};
    // At least 10 dB improvement
    tr.add(improvement_db > 10.0f
        ? PassResult("improvement_db", improvement_db, 10.0,
                      "RMS_raw=" + std::to_string(rms_raw) + " RMS_flt=" + std::to_string(rms_flt))
        : FailResult("improvement_db", improvement_db, 10.0,
                      "need >10dB, RMS_raw=" + std::to_string(rms_raw) + " RMS_flt=" + std::to_string(rms_flt)));
    return tr;
  });

  // ── Test 3: Channel independence — 256ch, unique constant ────────

  runner.test("channel_independence", [&]() -> TestResult {
    kalman.SetParams(0.1f, 1.0f, 0.0f, 25.0f);

    const uint32_t channels = 256;
    const uint32_t points   = 512;
    const size_t   total    = static_cast<size_t>(channels) * points;

    std::mt19937 rng(777);
    std::normal_distribution<float> noise(0.0f, 0.1f);

    std::vector<std::complex<float>> signal(total);
    for (uint32_t ch = 0; ch < channels; ++ch) {
      float cv = static_cast<float>(ch) * 10.0f;
      size_t base = static_cast<size_t>(ch) * points;
      for (uint32_t n = 0; n < points; ++n) {
        signal[base + n] = {cv + noise(rng), cv + noise(rng)};
      }
    }

    auto gpu_result = kalman.ProcessFromCPU(signal, channels, points);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    // Check last sample: should be close to ch * 10.0f
    TestResult tr{"channel_independence"};
    float max_ch_err = 0.0f;
    for (uint32_t ch = 0; ch < channels; ++ch) {
      float cv = static_cast<float>(ch) * 10.0f;
      size_t idx = static_cast<size_t>(ch) * points + (points - 1);
      float err = std::abs(gpu_data[idx].real() - cv);
      max_ch_err = std::max(max_ch_err, err);
    }
    tr.add(ScalarAbsError(max_ch_err, 0.0f, 1.0, "max_ch_err"));
    return tr;
  });

  // ── Test 4: Step response — step at n=512 ────────────────────────

  runner.test("step_response", [&]() -> TestResult {
    kalman.SetParams(1.0f, 25.0f, 0.0f, 25.0f);

    const uint32_t channels = 1;
    const uint32_t points   = 1024;

    std::vector<std::complex<float>> signal(points, {0.0f, 0.0f});
    for (uint32_t n = 512; n < points; ++n)
      signal[n] = {100.0f, 0.0f};

    auto cpu_result = kalman.ProcessCpu(signal, channels, points);

    // After ~100 samples past the step (n=612), should be > 80
    float val_612 = cpu_result[612].real();
    // At the end (n=1023), should be close to 100
    float val_end = cpu_result[1023].real();

    TestResult tr{"step_response"};
    tr.add(val_612 > 60.0f
        ? PassResult("val_at_612", val_612, 60.0, "should be >60")
        : FailResult("val_at_612", val_612, 60.0, "expected >60"));
    tr.add(ScalarAbsError(val_end, 100.0f, 5.0, "val_at_end"));
    return tr;
  });

  // ── Test 5: LFM Radar demo — 5 targets, beat signal + AWGN ──────

  runner.test("lfm_radar_demo", [&]() -> TestResult {
    // LFM radar parameters
    const float    fs         = 10e6f;
    const float    fdev       = 2e6f;
    const uint32_t N          = 16384;
    const float    Ti         = static_cast<float>(N) / fs;
    const float    mu         = fdev / Ti;
    const float    bin_hz     = fs / static_cast<float>(N);
    const float    c          = 3e8f;
    const float    noise_sigma = 0.30f;

    const uint32_t n_ant = 5;
    const float tau_us[5] = {50.0f, 100.0f, 150.0f, 200.0f, 250.0f};

    float tau[5], f_beat[5], range_km[5];
    for (uint32_t a = 0; a < n_ant; ++a) {
      tau[a]      = tau_us[a] * 1e-6f;
      f_beat[a]   = mu * tau[a];
      range_km[a] = c * tau[a] / 2.0f / 1000.0f;
    }

    // Generate dechirped beat signal + AWGN
    std::vector<std::complex<float>> signal(n_ant * N);
    std::vector<std::complex<float>> clean(n_ant * N);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, noise_sigma);

    for (uint32_t ch = 0; ch < n_ant; ++ch) {
      const float omega = 2.0f * static_cast<float>(M_PI) * f_beat[ch] / fs;
      for (uint32_t n = 0; n < N; ++n) {
        float re_clean = std::cos(omega * n);
        float im_clean = std::sin(omega * n);
        clean[ch * N + n] = {re_clean, im_clean};
        signal[ch * N + n] = {re_clean + noise(rng),
                                im_clean + noise(rng)};
      }
    }

    // Apply Kalman filter (Q=0.01 for enough bandwidth to track all beat freqs)
    kalman.SetParams(0.01f, 0.09f, 0.0f, 0.09f);

    auto gpu_result = kalman.ProcessFromCPU(signal, n_ant, N);
    auto filtered = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, n_ant * N);

    // Print demo tables via ConsoleOutput
    auto& con = ConsoleOutput::GetInstance();
    char buf[256];

    con.Print(gpu_id, "Kalman[ROCm]", "");
    con.Print(gpu_id, "Kalman[ROCm]",
        "================================================================");
    con.Print(gpu_id, "Kalman[ROCm]",
        "       LFM RADAR - KALMAN FILTER DEMONSTRATION");
    con.Print(gpu_id, "Kalman[ROCm]",
        "================================================================");

    std::snprintf(buf, sizeof(buf),
        "  fs=%.1fMHz  fdev=%.1fMHz  N=%u  Ti=%.3fms",
        fs / 1e6f, fdev / 1e6f, N, Ti * 1e3f);
    con.Print(gpu_id, "Kalman[ROCm]", std::string(buf));
    std::snprintf(buf, sizeof(buf),
        "  mu=%.1fMHz/s  dBin=%.1fHz  noise_sigma=%.2f",
        mu / 1e6f, bin_hz, noise_sigma);
    con.Print(gpu_id, "Kalman[ROCm]", std::string(buf));

    // Target table
    con.Print(gpu_id, "Kalman[ROCm]", "");
    con.Print(gpu_id, "Kalman[ROCm]",
        "  Ant | Range     | tau     | f_beat      | FFT bin");
    con.Print(gpu_id, "Kalman[ROCm]",
        "  ----+-----------+---------+-------------+--------");

    for (uint32_t a = 0; a < n_ant; ++a) {
      std::snprintf(buf, sizeof(buf),
          "   %u  | %5.1f km  | %3.0f us  | %6.2f kHz  |  #%3.0f",
          a, range_km[a], tau_us[a],
          f_beat[a] / 1e3f, f_beat[a] / bin_hz);
      con.Print(gpu_id, "Kalman[ROCm]", std::string(buf));
    }

    // Results table
    con.Print(gpu_id, "Kalman[ROCm]", "");
    con.Print(gpu_id, "Kalman[ROCm]",
        "  Kalman: Q=0.01  R=0.09  x0=0  P0=0.09");
    con.Print(gpu_id, "Kalman[ROCm]", "");
    con.Print(gpu_id, "Kalman[ROCm]",
        "  Ant | RMS raw  | RMS filt | Improv.  | dB gain");
    con.Print(gpu_id, "Kalman[ROCm]",
        "  ----+----------+----------+----------+--------");

    float total_db = 0.0f;

    for (uint32_t a = 0; a < n_ant; ++a) {
      float rms_raw = 0.0f, rms_flt = 0.0f;
      uint32_t count = 0;
      size_t base = static_cast<size_t>(a) * N;

      for (uint32_t n = 200; n < N; ++n) {
        float err_raw_re = signal[base + n].real() - clean[base + n].real();
        float err_raw_im = signal[base + n].imag() - clean[base + n].imag();
        float err_flt_re = filtered[base + n].real() - clean[base + n].real();
        float err_flt_im = filtered[base + n].imag() - clean[base + n].imag();
        rms_raw += err_raw_re * err_raw_re + err_raw_im * err_raw_im;
        rms_flt += err_flt_re * err_flt_re + err_flt_im * err_flt_im;
        ++count;
      }
      rms_raw = std::sqrt(rms_raw / (2.0f * count));
      rms_flt = std::sqrt(rms_flt / (2.0f * count));

      float db_gain = 20.0f * std::log10(rms_raw / (rms_flt + 1e-10f));
      total_db += db_gain;

      std::snprintf(buf, sizeof(buf),
          "   %u  |  %.4f  |  %.4f  |  x%.1f     | %+.1f dB",
          a, rms_raw, rms_flt,
          rms_raw / (rms_flt + 1e-10f), db_gain);
      con.Print(gpu_id, "Kalman[ROCm]", std::string(buf));
    }

    float avg_db = total_db / n_ant;

    std::snprintf(buf, sizeof(buf),
        "  Average improvement: %+.1f dB", avg_db);
    con.Print(gpu_id, "Kalman[ROCm]", "");
    con.Print(gpu_id, "Kalman[ROCm]", std::string(buf));
    con.Print(gpu_id, "Kalman[ROCm]",
        "================================================================");

    TestResult tr{"lfm_radar_demo"};
    tr.add(avg_db > 1.0f
        ? PassResult("avg_improvement_db", avg_db, 1.0, "avg >1dB")
        : FailResult("avg_improvement_db", avg_db, 1.0, "expected avg >1dB"));
    return tr;
  });

  // ── Summary ─────────────────────────────────────────────────────

  runner.print_summary();
}

}  // namespace test_kalman_rocm

#else  // !ENABLE_ROCM

namespace test_kalman_rocm {

inline void run() {
  // SKIPPED: ENABLE_ROCM not defined
}

}  // namespace test_kalman_rocm

#endif  // ENABLE_ROCM
