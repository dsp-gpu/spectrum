#pragma once

/**
 * @file test_fft_processor_rocm.hpp
 * @brief Tests for FFTProcessorROCm -- hipFFT-based FFT processing
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 * 1. single_beam_complex  — known tone at 100 Hz → peak at expected bin
 * 2. multi_beam_batch     — 8 beams with freq_step
 * 3. mag_phase_consistency — mag/phase vs complex output
 * 4. mag_phase_freq       — frequency array validation
 * 5. gpu_input            — void* device pointer input
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (migrated 2026-03-23)
 */

#if ENABLE_ROCM

#include "fft_processor_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"

#include "modules/test_utils/test_utils.hpp"

#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_processor_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

inline void run() {
  int gpu_id = 0;

  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    ConsoleOutput::GetInstance().Print(gpu_id, "FFT ROCm", "[!] No ROCm devices -- skip");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  TestRunner runner(&backend, "FFT ROCm", gpu_id);

  // ── Test 1: Single beam complex — known tone at 100 Hz ────────

  runner.test("single_beam_complex", [&]() -> TestResult {
    FFTProcessorROCm fft(&backend);

    auto data = refs::GenerateSinusoid(100.0f, 1000.0f, 1024);

    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = 1024;
    params.sample_rate = 1000.0f;
    params.output_mode = FFTOutputMode::COMPLEX;

    auto results = fft.ProcessComplex(data, params);

    TestResult tr{"single_beam_complex"};
    if (results.empty()) return tr.add(FailResult("size", 0, 1));

    uint32_t nFFT = results[0].nFFT;
    size_t expected_bin = static_cast<size_t>(100.0f * nFFT / 1000.0f);
    size_t peak_bin = refs::FindPeakBinComplex(results[0].spectrum.data(), nFFT / 2);

    tr.add(ValidationResult{peak_bin == expected_bin, "peak_bin",
        static_cast<double>(peak_bin), static_cast<double>(expected_bin),
        "peak=" + std::to_string(peak_bin) + " exp=" + std::to_string(expected_bin)});
    return tr;
  });

  // ── Test 2: Multi-beam batch (8 beams) ────────────────────────

  runner.test("multi_beam_batch", [&]() -> TestResult {
    FFTProcessorROCm fft(&backend);

    const uint32_t beam_count = 8;
    const uint32_t n_point = 1024;
    const float sample_rate = 1000.0f;
    const float base_freq = 50.0f, freq_step = 25.0f;

    // Multi-beam with different frequencies per beam
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
      float freq = base_freq + b * freq_step;
      for (size_t i = 0; i < n_point; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float ph = 2.0f * static_cast<float>(M_PI) * freq * t;
        data[b * n_point + i] = {std::cos(ph), std::sin(ph)};
      }
    }

    FFTProcessorParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = FFTOutputMode::COMPLEX;

    auto results = fft.ProcessComplex(data, params);

    TestResult tr{"multi_beam_batch"};
    if (results.size() != beam_count)
      return tr.add(FailResult("beam_count", results.size(), beam_count));

    for (uint32_t b = 0; b < beam_count; ++b) {
      float freq = base_freq + b * freq_step;
      uint32_t nFFT = results[b].nFFT;
      size_t expected = static_cast<size_t>(
          std::lround(freq * static_cast<float>(nFFT) / sample_rate));
      size_t peak = refs::FindPeakBinComplex(results[b].spectrum.data(), nFFT / 2);
      // ±1 bin tolerance
      bool ok = (peak + 1 >= expected && peak <= expected + 1);
      tr.add(ValidationResult{ok, "beam" + std::to_string(b),
          static_cast<double>(peak), static_cast<double>(expected), ""});
    }
    return tr;
  });

  // ── Test 3: MagPhase consistency with complex ─────────────────

  runner.test("mag_phase_consistency", [&]() -> TestResult {
    auto data = refs::GenerateSinusoid(200.0f, 1000.0f, 512);

    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = 512;
    params.sample_rate = 1000.0f;

    FFTProcessorROCm fft_c(&backend);
    params.output_mode = FFTOutputMode::COMPLEX;
    auto cr = fft_c.ProcessComplex(data, params);

    FFTProcessorROCm fft_mp(&backend);
    params.output_mode = FFTOutputMode::MAGNITUDE_PHASE;
    auto mr = fft_mp.ProcessMagPhase(data, params);

    TestResult tr{"mag_phase_consistency"};
    if (cr.empty() || mr.empty())
      return tr.add(FailResult("empty", 0, 1));

    uint32_t nFFT = cr[0].nFFT;
    float max_mag_err = 0, max_phase_err = 0;
    for (uint32_t k = 0; k < nFFT; ++k) {
      float cpu_mag = std::abs(cr[0].spectrum[k]);
      float cpu_ph = std::arg(cr[0].spectrum[k]);
      float mag_err = std::fabs(cpu_mag - mr[0].magnitude[k]);
      float ph_err = std::fabs(cpu_ph - mr[0].phase[k]);
      if (ph_err > static_cast<float>(M_PI))
        ph_err = 2.0f * static_cast<float>(M_PI) - ph_err;
      max_mag_err = std::max(max_mag_err, mag_err);
      if (cpu_mag > 1e-3f) max_phase_err = std::max(max_phase_err, ph_err);
    }

    tr.add(ScalarAbsError(static_cast<double>(max_mag_err), 0.0, 1e-2, "mag_err"));
    tr.add(ScalarAbsError(static_cast<double>(max_phase_err), 0.0, 1e-2, "phase_err"));
    return tr;
  });

  // ── Test 4: MagPhaseFreq — frequency array ────────────────────

  runner.test("mag_phase_freq", [&]() -> TestResult {
    FFTProcessorROCm fft(&backend);
    auto data = refs::GenerateSinusoid(150.0f, 1000.0f, 1024);

    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = 1024;
    params.sample_rate = 1000.0f;
    params.output_mode = FFTOutputMode::MAGNITUDE_PHASE_FREQ;

    auto results = fft.ProcessMagPhase(data, params);

    TestResult tr{"mag_phase_freq"};
    if (results.empty() || results[0].frequency.empty())
      return tr.add(FailResult("empty", 0, 1));

    uint32_t nFFT = results[0].nFFT;
    float freq_step = 1000.0f / static_cast<float>(nFFT);

    // Verify frequency array
    float max_freq_err = 0;
    for (uint32_t k = 0; k < nFFT; ++k) {
      float expected = static_cast<float>(k) * freq_step;
      max_freq_err = std::max(max_freq_err,
          std::fabs(results[0].frequency[k] - expected));
    }
    tr.add(ScalarAbsError(static_cast<double>(max_freq_err), 0.0, 1e-4, "freq_array"));

    // Peak frequency
    size_t peak_bin = refs::FindPeakBin(results[0].magnitude.data(), nFFT / 2);
    float peak_freq = results[0].frequency[peak_bin];
    tr.add(ScalarAbsError(static_cast<double>(peak_freq), 150.0, freq_step, "peak_freq"));
    return tr;
  });

  // ── Test 5: GPU input (void*) ─────────────────────────────────

  runner.test("gpu_input", [&]() -> TestResult {
    FFTProcessorROCm fft(&backend);
    auto data = refs::GenerateSinusoid(100.0f, 1000.0f, 1024);

    size_t data_size = data.size() * sizeof(std::complex<float>);
    void* gpu_data = backend.Allocate(data_size);
    backend.MemcpyHostToDevice(gpu_data, data.data(), data_size);

    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = 1024;
    params.sample_rate = 1000.0f;
    params.output_mode = FFTOutputMode::COMPLEX;

    auto results = fft.ProcessComplex(gpu_data, params, data_size);
    backend.Free(gpu_data);

    TestResult tr{"gpu_input"};
    if (results.empty())
      return tr.add(FailResult("empty", 0, 1));

    uint32_t nFFT = results[0].nFFT;
    size_t expected = static_cast<size_t>(100.0f * nFFT / 1000.0f);
    size_t peak = refs::FindPeakBinComplex(results[0].spectrum.data(), nFFT / 2);
    tr.add(ValidationResult{peak == expected, "peak_bin",
        static_cast<double>(peak), static_cast<double>(expected), ""});
    return tr;
  });

  runner.print_summary();
}

}  // namespace test_fft_processor_rocm

#endif  // ENABLE_ROCM
