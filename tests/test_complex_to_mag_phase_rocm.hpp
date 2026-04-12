#pragma once

/**
 * @file test_complex_to_mag_phase_rocm.hpp
 * @brief Tests for ComplexToMagPhaseROCm -- direct complex->mag+phase conversion
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 * 1. single_beam_cpu    — CPU→CPU sinusoid, mag/phase accuracy
 * 2. multi_beam_cpu     — 8 beams varying amplitude
 * 3. gpu_input          — external GPU input → CPU output
 * 4. cpu_to_gpu         — ProcessToGPU interleaved output
 * 5. gpu_to_gpu         — full GPU pipeline
 * 6. accuracy           — edge cases (zero, pure real/imag, 3-4-5, large/small)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01 (migrated 2026-03-23)
 */

#if ENABLE_ROCM

#include "complex_to_mag_phase_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include "modules/test_utils/test_utils.hpp"

#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_complex_to_mag_phase_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// Module-specific helper: verify mag/phase pair with phase wrapping
inline void CheckMagPhase(const std::complex<float>* input,
    const float* mag, const float* phase, uint32_t n,
    float& max_mag_err, float& max_phase_err) {
  max_mag_err = 0; max_phase_err = 0;
  for (uint32_t k = 0; k < n; ++k) {
    float cpu_mag = std::abs(input[k]);
    float cpu_ph = std::arg(input[k]);
    float mag_err = std::fabs(cpu_mag - mag[k]);
    float ph_err = std::fabs(cpu_ph - phase[k]);
    if (ph_err > static_cast<float>(M_PI))
      ph_err = 2.0f * static_cast<float>(M_PI) - ph_err;
    max_mag_err = std::max(max_mag_err, mag_err);
    if (cpu_mag > 1e-6f) max_phase_err = std::max(max_phase_err, ph_err);
  }
}

inline void run() {
  int gpu_id = 0;

  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    ConsoleOutput::GetInstance().Print(gpu_id, "C2MP ROCm", "[!] No ROCm devices -- skip");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "C2MP ROCm", gpu_id);

  // ── Test 1: Single beam CPU→CPU ───────────────────────────────

  runner.test("single_beam_cpu", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);
    auto data = refs::GenerateSinusoid(100.0f, 1000.0f, 4096, 2.5f);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point = 4096;

    auto results = converter.Process(data, params);

    TestResult tr{"single_beam_cpu"};
    if (results.empty()) return tr.add(FailResult("empty", 0, 1));

    float me, pe;
    CheckMagPhase(data.data(), results[0].magnitude.data(),
                  results[0].phase.data(), 4096, me, pe);
    tr.add(ScalarAbsError(static_cast<double>(me), 0.0, 1e-3, "mag_err"));
    tr.add(ScalarAbsError(static_cast<double>(pe), 0.0, 1e-3, "phase_err"));
    return tr;
  });

  // ── Test 2: Multi-beam CPU→CPU ────────────────────────────────

  runner.test("multi_beam_cpu", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);

    const uint32_t beams = 8, n = 4096;
    // Generate with varying amplitude
    std::vector<std::complex<float>> data(beams * n);
    for (uint32_t b = 0; b < beams; ++b) {
      float amp = 0.5f + b * 0.5f;
      auto beam = refs::GenerateSinusoid(500.0f, 12000.0f, n, amp);
      std::copy(beam.begin(), beam.end(), data.begin() + b * n);
    }

    MagPhaseParams params;
    params.beam_count = beams;
    params.n_point = n;

    auto results = converter.Process(data, params);

    TestResult tr{"multi_beam_cpu"};
    if (results.size() != beams) return tr.add(FailResult("beams", results.size(), beams));

    float worst = 0;
    for (uint32_t b = 0; b < beams; ++b) {
      for (uint32_t k = 0; k < n; ++k) {
        float err = std::fabs(std::abs(data[b * n + k]) - results[b].magnitude[k]);
        worst = std::max(worst, err);
      }
    }
    tr.add(ScalarAbsError(static_cast<double>(worst), 0.0, 1e-3, "mag_err"));
    return tr;
  });

  // ── Test 3: GPU input → CPU output ────────────────────────────

  runner.test("gpu_input", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);
    auto data = refs::GenerateSinusoid(200.0f, 1000.0f, 2048);

    size_t sz = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(sz);
    backend.MemcpyHostToDevice(gpu_data, data.data(), sz);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point = 2048;

    auto results = converter.Process(gpu_data, params, sz);
    backend.Free(gpu_data);

    TestResult tr{"gpu_input"};
    if (results.empty()) return tr.add(FailResult("empty", 0, 1));

    float me, pe;
    CheckMagPhase(data.data(), results[0].magnitude.data(),
                  results[0].phase.data(), 2048, me, pe);
    tr.add(ScalarAbsError(static_cast<double>(me), 0.0, 1e-3, "mag_err"));
    return tr;
  });

  // ── Test 4: CPU → GPU (ProcessToGPU) ──────────────────────────

  runner.test("cpu_to_gpu", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);
    auto data = refs::GenerateSinusoid(300.0f, 2000.0f, 1024, 3.0f);

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point = 1024;

    void* gpu_out = converter.ProcessToGPU(data, params);

    TestResult tr{"cpu_to_gpu"};
    if (!gpu_out) return tr.add(FailResult("null_ptr", 0, 1));

    std::vector<float> raw(1024 * 2);
    backend.MemcpyDeviceToHost(raw.data(), gpu_out, raw.size() * sizeof(float));
    backend.Free(gpu_out);

    float max_mag_err = 0, max_phase_err = 0;
    for (uint32_t k = 0; k < 1024; ++k) {
      float gpu_mag = raw[k * 2], gpu_ph = raw[k * 2 + 1];
      float cpu_mag = std::abs(data[k]), cpu_ph = std::arg(data[k]);
      max_mag_err = std::max(max_mag_err, std::fabs(cpu_mag - gpu_mag));
      float pe = std::fabs(cpu_ph - gpu_ph);
      if (pe > static_cast<float>(M_PI)) pe = 2.0f * static_cast<float>(M_PI) - pe;
      if (cpu_mag > 1e-6f) max_phase_err = std::max(max_phase_err, pe);
    }
    tr.add(ScalarAbsError(static_cast<double>(max_mag_err), 0.0, 1e-3, "mag_err"));
    tr.add(ScalarAbsError(static_cast<double>(max_phase_err), 0.0, 1e-3, "phase_err"));
    return tr;
  });

  // ── Test 5: GPU → GPU ────────────────────────────────────────

  runner.test("gpu_to_gpu", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);
    const uint32_t beams = 4, n = 2048;

    std::vector<std::complex<float>> data(beams * n);
    for (uint32_t b = 0; b < beams; ++b) {
      float amp = 1.0f + b;
      auto beam = refs::GenerateSinusoid(150.0f, 1000.0f, n, amp);
      std::copy(beam.begin(), beam.end(), data.begin() + b * n);
    }

    size_t sz = data.size() * sizeof(std::complex<float>);
    void* gpu_in = backend.Allocate(sz);
    backend.MemcpyHostToDevice(gpu_in, data.data(), sz);

    MagPhaseParams params;
    params.beam_count = beams;
    params.n_point = n;

    void* gpu_out = converter.ProcessToGPU(gpu_in, params, sz);
    backend.Free(gpu_in);

    TestResult tr{"gpu_to_gpu"};
    if (!gpu_out) return tr.add(FailResult("null_ptr", 0, 1));

    size_t total = beams * n;
    std::vector<float> raw(total * 2);
    backend.MemcpyDeviceToHost(raw.data(), gpu_out, raw.size() * sizeof(float));
    backend.Free(gpu_out);

    float worst = 0;
    for (size_t i = 0; i < total; ++i) {
      worst = std::max(worst, std::fabs(std::abs(data[i]) - raw[i * 2]));
    }
    tr.add(ScalarAbsError(static_cast<double>(worst), 0.0, 1e-3, "mag_err"));
    return tr;
  });

  // ── Test 6: Accuracy — edge cases ─────────────────────────────

  runner.test("accuracy", [&]() -> TestResult {
    ComplexToMagPhaseROCm converter(&backend);

    std::vector<std::complex<float>> data = {
      {0,0}, {1,0}, {-1,0}, {0,1}, {0,-1},
      {1,1}, {-1,1}, {-1,-1}, {1,-1},
      {1000,2000}, {1e-6f,1e-6f}, {3,4}, {0,0}, {5,12}, {1,0.0001f}, {0.0001f,1}
    };
    uint32_t n = static_cast<uint32_t>(data.size());

    MagPhaseParams params;
    params.beam_count = 1;
    params.n_point = n;

    auto results = converter.Process(data, params);

    TestResult tr{"accuracy"};
    if (results.empty()) return tr.add(FailResult("empty", 0, 1));

    float me, pe;
    CheckMagPhase(data.data(), results[0].magnitude.data(),
                  results[0].phase.data(), n, me, pe);
    tr.add(ScalarAbsError(static_cast<double>(me), 0.0, 1e-2, "mag_err"));
    tr.add(ScalarAbsError(static_cast<double>(pe), 0.0, 1e-2, "phase_err"));

    // 3-4-5 triangle
    tr.add(ScalarAbsError(static_cast<double>(results[0].magnitude[11]), 5.0, 1e-3, "mag_3_4_5"));
    return tr;
  });

  runner.print_summary();
}

}  // namespace test_complex_to_mag_phase_rocm

#endif  // ENABLE_ROCM
