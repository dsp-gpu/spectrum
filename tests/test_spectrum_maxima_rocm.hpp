#pragma once

/**
 * @file test_spectrum_maxima_rocm.hpp
 * @brief ROCm tests for SpectrumProcessorROCm (peak finding)
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 * 1. one_peak          — CW 100 Hz, single peak search
 * 2. two_peaks         — two CW signals, TWO_PEAKS mode
 * 3. find_all_maxima   — 3 sinusoids → 3 peaks
 * 4. all_maxima_fft    — pre-computed FFT with known peaks
 * 5. batch_16_beams    — 16 beams with unique frequencies
 * 6. compare_opencl    — ROCm vs OpenCL cross-validation (skipped)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (migrated 2026-03-23)
 */

#include <vector>
#include <complex>
#include <cmath>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM
#include "processors/spectrum_processor_rocm.hpp"
#include "factory/spectrum_processor_factory.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "common/backend_type.hpp"

#include "modules/test_utils/test_utils.hpp"
#endif

namespace test_spectrum_maxima_rocm {

#if ENABLE_ROCM

using namespace antenna_fft;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// Module-specific helpers (not generic enough for test_utils)

inline std::vector<std::complex<float>> GenerateCW(
    float freq, float sample_rate, uint32_t n_point,
    uint32_t beam_count = 1, float amplitude = 1.0f) {
  std::vector<std::complex<float>> data(static_cast<size_t>(beam_count) * n_point);
  for (uint32_t b = 0; b < beam_count; ++b)
    for (uint32_t t = 0; t < n_point; ++t) {
      float ph = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
      data[b * n_point + t] = {amplitude * std::sin(ph), 0.0f};
    }
  return data;
}

inline std::vector<std::complex<float>> GenerateMultiCW(
    const std::vector<float>& freqs, float sample_rate, uint32_t n_point,
    uint32_t beam_count = 1) {
  std::vector<std::complex<float>> data(static_cast<size_t>(beam_count) * n_point);
  for (uint32_t b = 0; b < beam_count; ++b)
    for (uint32_t t = 0; t < n_point; ++t) {
      float val = 0;
      for (float f : freqs)
        val += std::sin(2.0f * static_cast<float>(M_PI) * f * t / sample_rate);
      data[b * n_point + t] = {val, 0.0f};
    }
  return data;
}

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  try { backend.Initialize(gpu_id); }
  catch (...) { return; }

  TestRunner runner(&backend, "SpecMaxima ROCm", gpu_id);

  // ── Test 1: ONE_PEAK — CW 100 Hz ─────────────────────────────

  runner.test("one_peak", [&]() -> TestResult {
    const uint32_t n_point = 1000;
    const float sample_rate = 10000.0f, freq = 100.0f;
    const uint32_t antenna_count = 4;

    auto data = GenerateCW(freq, sample_rate, n_point, antenna_count);

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, &backend);
    processor->Initialize(params);
    auto results = processor->ProcessFromCPU(data);

    TestResult tr{"one_peak"};
    if (results.size() != antenna_count)
      return tr.add(FailResult("count", results.size(), antenna_count));

    for (const auto& r : results) {
      float error = std::abs(r.interpolated.refined_frequency - freq);
      tr.add(ScalarAbsError(static_cast<double>(error), 0.0, 5.0, "beam" + std::to_string(r.antenna_id)));
    }
    return tr;
  });

  // ── Test 2: TWO_PEAKS ─────────────────────────────────────────

  runner.test("two_peaks", [&]() -> TestResult {
    auto data = GenerateMultiCW({100.0f, 300.0f}, 10000.0f, 1000, 2);

    SpectrumParams params;
    params.antenna_count = 2;
    params.n_point = 1000;
    params.repeat_count = 1;
    params.sample_rate = 10000.0f;
    params.peak_mode = PeakSearchMode::TWO_PEAKS;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, &backend);
    processor->Initialize(params);
    auto results = processor->ProcessFromCPU(data);

    TestResult tr{"two_peaks"};
    tr.add(ValidationResult{results.size() == 4, "count",
        static_cast<double>(results.size()), 4.0, ""});
    return tr;
  });

  // ── Test 3: FindAllMaxima — 3 frequencies ─────────────────────

  runner.test("find_all_maxima", [&]() -> TestResult {
    auto data = GenerateMultiCW({50.0f, 120.0f, 200.0f}, 1000.0f, 1024, 2);

    SpectrumParams params;
    params.antenna_count = 2;
    params.n_point = 1024;
    params.repeat_count = 1;
    params.sample_rate = 1000.0f;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, &backend);
    processor->Initialize(params);
    auto result = processor->FindAllMaximaFromCPU(data, OutputDestination::CPU, 1, 0);

    TestResult tr{"find_all_maxima"};
    tr.add(ValidationResult{result.total_maxima > 0, "total_maxima",
        static_cast<double>(result.total_maxima), 1.0, ""});
    for (const auto& beam : result.beams)
      tr.add(ValidationResult{beam.num_maxima >= 3, "beam" + std::to_string(beam.antenna_id),
          static_cast<double>(beam.num_maxima), 3.0, ""});
    return tr;
  });

  // ── Test 4: AllMaximaFromCPU — pre-computed FFT ───────────────

  runner.test("all_maxima_fft", [&]() -> TestResult {
    const uint32_t nFFT = 1024;

    std::vector<std::complex<float>> fft_data(nFFT, {0.1f, 0.0f});
    fft_data[50] = {100.0f, 0.0f};
    fft_data[120] = {80.0f, 0.0f};
    fft_data[200] = {60.0f, 0.0f};

    SpectrumParams params;
    params.antenna_count = 1;
    params.n_point = nFFT;
    params.repeat_count = 1;
    params.sample_rate = 1000.0f;
    params.nFFT = nFFT;
    params.base_fft = nFFT;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, &backend);
    processor->Initialize(params);
    auto result = processor->AllMaximaFromCPU(fft_data, 1, nFFT, 1000.0f,
                                               OutputDestination::CPU, 1, nFFT / 2);

    TestResult tr{"all_maxima_fft"};
    tr.add(ValidationResult{result.total_maxima >= 3, "total",
        static_cast<double>(result.total_maxima), 3.0, ""});
    return tr;
  });

  // ── Test 5: Batch 16 beams ────────────────────────────────────

  runner.test("batch_16_beams", [&]() -> TestResult {
    const uint32_t n_point = 1000, antenna_count = 16;
    const float sample_rate = 10000.0f;

    std::vector<std::complex<float>> data(static_cast<size_t>(antenna_count) * n_point);
    for (uint32_t b = 0; b < antenna_count; ++b) {
      float freq = 100.0f + b * 10.0f;
      for (uint32_t t = 0; t < n_point; ++t) {
        float ph = 2.0f * static_cast<float>(M_PI) * freq * t / sample_rate;
        data[b * n_point + t] = {std::sin(ph), 0.0f};
      }
    }

    SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point = n_point;
    params.repeat_count = 1;
    params.sample_rate = sample_rate;
    params.peak_mode = PeakSearchMode::ONE_PEAK;
    params.memory_limit = 0.8f;

    auto processor = SpectrumProcessorFactory::Create(BackendType::ROCm, &backend);
    processor->Initialize(params);
    auto results = processor->ProcessFromCPU(data);

    TestResult tr{"batch_16_beams"};
    if (results.size() != antenna_count)
      return tr.add(FailResult("count", results.size(), antenna_count));

    for (const auto& r : results) {
      float expected = 100.0f + r.antenna_id * 10.0f;
      float error = std::abs(r.interpolated.refined_frequency - expected);
      tr.add(ScalarAbsError(static_cast<double>(error), 0.0, 10.0,
          "beam" + std::to_string(r.antenna_id)));
    }
    return tr;
  });

  // ── Test 6: Compare with OpenCL (skipped) ─────────────────────

  runner.test("compare_opencl", [&]() {
    throw SkipTest("OpenCL backend not available in this test");
    return ValidationResult{true, "skip", 0, 0, ""};
  });

  runner.print_summary();
}

#else

inline void run() {}

#endif  // ENABLE_ROCM

}  // namespace test_spectrum_maxima_rocm
