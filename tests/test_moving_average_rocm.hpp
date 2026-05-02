#pragma once

// ============================================================================
// test_moving_average_rocm — тесты MovingAverageFilterROCm (ROCm)
//
// ЧТО:    6 тестов: EMA/SMA/MMA/DEMA/TEMA (8ch×4096pts GPU vs CPU),
//         step_response (5 типов фильтров, 120pts, table output).
// ЗАЧЕМ:  Moving averages — в baseline estimation и CFAR detector.
//         Ошибка здесь = неверный шумовой порог в radar.
// ПОЧЕМУ: ENABLE_ROCM. Эталон CPU. MIGRATED to test_utils 2026-03-23.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_moving_average_rocm.hpp
 * @brief ROCm tests for MovingAverageFilterROCm (SMA, EMA, MMA, DEMA, TEMA)
 *
 * MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 *   1. ema: EMA(N=10) GPU vs CPU, 8ch x 4096pts
 *   2. sma: SMA(N=8) GPU vs CPU, 8ch x 4096pts
 *   3. mma: MMA(N=10) GPU vs CPU
 *   4. dema: DEMA(N=10) GPU vs CPU
 *   5. tema: TEMA(N=10) GPU vs CPU
 *   6. step_response: step demo (120pts, 5 filter types)
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

#include <spectrum/filters/moving_average_filter_rocm.hpp>
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/backends/rocm/rocm_core.hpp>

// test_utils — unified test infrastructure
#include "modules/test_utils/test_utils.hpp"

namespace test_moving_average_rocm {

using namespace filters;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate random complex signal
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> generate_random_signal(
    uint32_t channels, uint32_t points, unsigned int seed = 42)
{
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> signal(total);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < total; ++i) {
    signal[i] = {dist(rng), dist(rng)};
  }
  return signal;
}

// ═══════════════════════════════════════════════════════════════════════════
// run() — TestRunner (functional style)
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  int gpu_id = 0;

  // Check for ROCm devices
  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    auto& con = ConsoleOutput::GetInstance();
    con.Print(gpu_id, "MA[ROCm]", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  MovingAverageFilterROCm filter(&backend);

  TestRunner runner(&backend, "MA[ROCm]", gpu_id);

  // ── Helper lambda: run single MA type GPU vs CPU ──────────────────

  auto test_ma_type = [&](const std::string& test_name, MAType type,
                          uint32_t window_size, uint32_t channels,
                          uint32_t points) {
    runner.test(test_name, [&, type, window_size, channels, points]() {
      filter.SetParams(type, window_size);

      size_t total = static_cast<size_t>(channels) * points;
      auto signal = generate_random_signal(channels, points);

      auto cpu_result = filter.ProcessCpu(signal, channels, points);
      auto gpu_result = filter.ProcessFromCPU(signal, channels, points);

      auto gpu_data = ReadHipBuffer<std::complex<float>>(
          backend.GetNativeQueue(), gpu_result.data, total);

      return AbsError(gpu_data.data(), cpu_result.data(), total,
                      tolerance::kFilter, test_name);
    });
  };

  // ── Test 1: EMA(N=10), 8ch x 4096pts ─────────────────────────────

  test_ma_type("ema", MAType::EMA, 10, 8, 4096);

  // ── Test 2: SMA(N=8), 8ch x 4096pts ──────────────────────────────

  test_ma_type("sma", MAType::SMA, 8, 8, 4096);

  // ── Test 3: MMA(N=10), 8ch x 4096pts ─────────────────────────────

  test_ma_type("mma", MAType::MMA, 10, 8, 4096);

  // ── Test 4: DEMA(N=10), 8ch x 4096pts ────────────────────────────

  test_ma_type("dema", MAType::DEMA, 10, 8, 4096);

  // ── Test 5: TEMA(N=10), 8ch x 4096pts ────────────────────────────

  test_ma_type("tema", MAType::TEMA, 10, 8, 4096);

  // ── Test 6: Step response demo — 120pts step signal ───────────────

  runner.test("step_response", [&]() -> TestResult {
    const uint32_t channels = 1;
    const uint32_t points   = 120;
    const uint32_t N        = 10;

    // Step signal: 20 zeros, 50 ones, 50 zeros
    std::vector<std::complex<float>> sig(points, {0.0f, 0.0f});
    for (uint32_t i = 20; i < 70; ++i)
      sig[i] = {1.0f, 0.0f};

    // Compute all 5 filter types via CPU
    struct FilterResult {
      const char* name;
      MAType      type;
      std::vector<std::complex<float>> out;
    };

    std::vector<FilterResult> results = {
      {"SMA",  MAType::SMA,  {}},
      {"EMA",  MAType::EMA,  {}},
      {"MMA",  MAType::MMA,  {}},
      {"DEMA", MAType::DEMA, {}},
      {"TEMA", MAType::TEMA, {}},
    };

    for (auto& r : results) {
      filter.SetParams(r.type, N);
      r.out = filter.ProcessCpu(sig, channels, points);
    }

    // Print table header
    auto& con = ConsoleOutput::GetInstance();
    con.Print(gpu_id, "MA[ROCm]", "");
    con.Print(gpu_id, "MA[ROCm]",
        "--- [MA Demo] Step signal: 20 zeros / 50 ones / 50 zeros ---");
    con.Print(gpu_id, "MA[ROCm]",
        "  t  | input | SMA(10) | EMA(10) | MMA(10) | DEMA(10)| TEMA(10)");
    con.Print(gpu_id, "MA[ROCm]",
        " ----+-------+---------+---------+---------+---------+---------");

    // Print every 5th sample
    char buf[160];
    for (uint32_t t = 0; t < points; t += 5) {
      std::snprintf(buf, sizeof(buf),
          " %3u |  %.1f  |  %5.3f  |  %5.3f  |  %5.3f  |  %5.3f  |  %5.3f",
          t,
          sig[t].real(),
          results[0].out[t].real(),
          results[1].out[t].real(),
          results[2].out[t].real(),
          results[3].out[t].real(),
          results[4].out[t].real()
      );
      con.Print(gpu_id, "MA[ROCm]", std::string(buf));
    }

    TestResult tr{"step_response"};

    // Verify: all filters reach ~1.0 at mid-plateau (t=55)
    for (auto& r : results) {
      float val_mid = r.out[55].real();
      tr.add(ScalarAbsError(val_mid, 1.0f, 0.05,
                             std::string(r.name) + "_plateau_t55"));
    }

    // TEMA should react faster than EMA at the front
    bool tema_leads = (results[4].out[23].real() > results[1].out[23].real());
    tr.add(tema_leads
        ? PassResult("tema_leads_ema_t23", results[4].out[23].real(), 0.0, "TEMA > EMA at front edge")
        : FailResult("tema_leads_ema_t23", results[4].out[23].real(), 0.0, "TEMA should lead EMA at t=23"));

    // GPU vs CPU check (EMA)
    filter.SetParams(MAType::EMA, N);
    auto gpu_result = filter.ProcessFromCPU(sig, channels, points);
    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, points);

    tr.add(AbsError(gpu_data.data(), results[1].out.data(), points,
                    tolerance::kFilter, "ema_gpu_vs_cpu"));

    return tr;
  });

  // ── Summary ─────────────────────────────────────────────────────

  runner.print_summary();
}

}  // namespace test_moving_average_rocm

#else  // !ENABLE_ROCM

namespace test_moving_average_rocm {

inline void run() {
  // SKIPPED: ENABLE_ROCM not defined
}

}  // namespace test_moving_average_rocm

#endif  // ENABLE_ROCM
