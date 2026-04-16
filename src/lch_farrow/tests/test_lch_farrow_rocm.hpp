#pragma once

/**
 * @file test_lch_farrow_rocm.hpp
 * @brief ROCm tests for LchFarrowROCm fractional delay processor
 *
 * Tests:
 *   1. Zero delay: output == input (within tolerance)
 *   2. Integer delay (5 samples): GPU vs CPU reference
 *   3. Fractional delay (2.7 samples): GPU vs CPU reference
 *   4. Multi-antenna: 4 antennas with different delays
 *
 * NOT RUN on Windows — compile-only check.
 * Will be tested on Linux + AMD GPU (Radeon 9070 / MI100).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <sstream>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM
#include "lch_farrow_rocm.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include <core/services/console_output.hpp>
#endif

namespace test_lch_farrow_rocm {

#if ENABLE_ROCM

using namespace lch_farrow;
using namespace drv_gpu_lib;

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate CW signal (flat vector for ProcessFromCPU)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> generate_cw_flat(
    uint32_t antennas, uint32_t points, float fs, float freq,
    float amplitude = 1.0f)
{
  std::vector<std::complex<float>> signal(
      static_cast<size_t>(antennas) * points);
  for (uint32_t a = 0; a < antennas; ++a) {
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / fs;
      float phase = 2.0f * static_cast<float>(M_PI) * freq * t;
      signal[a * points + n] = std::complex<float>(
          amplitude * std::cos(phase), amplitude * std::sin(phase));
    }
  }
  return signal;
}

// Helper: flat → 2D (for ProcessCpu)
inline std::vector<std::vector<std::complex<float>>> flat_to_2d(
    const std::vector<std::complex<float>>& flat,
    uint32_t antennas, uint32_t points)
{
  std::vector<std::vector<std::complex<float>>> result(antennas);
  for (uint32_t a = 0; a < antennas; ++a) {
    result[a].assign(flat.begin() + a * points,
                     flat.begin() + (a + 1) * points);
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Zero delay — output == input
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_zero_delay(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    LchFarrowROCm processor(&backend);

    const float fs = 1e6f;
    const uint32_t points = 4096;
    const float freq = 50000.0f;
    const uint32_t antennas = 1;

    auto signal = generate_cw_flat(antennas, points, fs, freq);

    processor.SetDelays({0.0f});
    processor.SetSampleRate(fs);

    auto result = processor.ProcessFromCPU(signal, antennas, points);

    // Read output from GPU
    std::vector<std::complex<float>> output(points);
    hipError_t err = hipMemcpyDtoH(output.data(), result.data,
                                     points * sizeof(std::complex<float>));
    hipFree(result.data);

    if (err != hipSuccess) {
      con.Print(gpu_id, "LchFarrow[ROCm]", "[X] Zero Delay: hipMemcpyDtoH failed");
      return false;
    }

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - signal[n]);
      if (e > max_err) max_err = e;
    }

    bool passed = (max_err < 1e-4f);
    std::ostringstream oss;
    oss << "Test 1 (zero delay): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-4)");
    con.Print(gpu_id, "LchFarrow[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "LchFarrow[ROCm]",
              "[X] Zero Delay EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Integer delay (5 samples) — GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_integer_delay(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    LchFarrowROCm processor(&backend);

    const float fs = 1e6f;
    const uint32_t points = 4096;
    const float freq = 50000.0f;
    const uint32_t antennas = 1;
    const float delay_us = 5.0f;  // 5 us at 1 MHz = 5 samples

    auto signal = generate_cw_flat(antennas, points, fs, freq);

    processor.SetDelays({delay_us});
    processor.SetSampleRate(fs);

    // GPU
    auto result = processor.ProcessFromCPU(signal, antennas, points);

    std::vector<std::complex<float>> output(points);
    hipMemcpyDtoH(output.data(), result.data,
                    points * sizeof(std::complex<float>));
    hipFree(result.data);

    // CPU reference
    auto input_2d = flat_to_2d(signal, antennas, points);
    processor.SetDelays({delay_us});
    auto cpu_ref = processor.ProcessCpu(input_2d, antennas, points);

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - cpu_ref[0][n]);
      if (e > max_err) max_err = e;
    }

    bool passed = (max_err < 1e-2f);
    std::ostringstream oss;
    oss << "Test 2 (integer delay 5): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-2)");
    con.Print(gpu_id, "LchFarrow[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "LchFarrow[ROCm]",
              "[X] Integer Delay EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Fractional delay (2.7 samples) — GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_fractional_delay(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    LchFarrowROCm processor(&backend);

    const float fs = 1e6f;
    const uint32_t points = 4096;
    const float freq = 50000.0f;
    const uint32_t antennas = 1;
    const float delay_us = 2.7f;  // 2.7 us at 1 MHz = 2.7 samples

    auto signal = generate_cw_flat(antennas, points, fs, freq);

    processor.SetDelays({delay_us});
    processor.SetSampleRate(fs);

    // GPU
    auto result = processor.ProcessFromCPU(signal, antennas, points);

    std::vector<std::complex<float>> output(points);
    hipMemcpyDtoH(output.data(), result.data,
                    points * sizeof(std::complex<float>));
    hipFree(result.data);

    // CPU reference
    auto input_2d = flat_to_2d(signal, antennas, points);
    processor.SetDelays({delay_us});
    auto cpu_ref = processor.ProcessCpu(input_2d, antennas, points);

    float max_err = 0.0f;
    for (uint32_t n = 0; n < points; ++n) {
      float e = std::abs(output[n] - cpu_ref[0][n]);
      if (e > max_err) max_err = e;
    }

    bool passed = (max_err < 1e-2f);
    std::ostringstream oss;
    oss << "Test 3 (fractional delay 2.7): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-2)");
    con.Print(gpu_id, "LchFarrow[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "LchFarrow[ROCm]",
              "[X] Fractional Delay EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Multi-antenna — 4 antennas with different delays
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_multi_antenna(ConsoleOutput& con, int gpu_id) {
  try {
    ROCmBackend backend;
    backend.Initialize(gpu_id);

    LchFarrowROCm processor(&backend);

    const float fs = 1e6f;
    const uint32_t points = 2048;
    const float freq = 50000.0f;
    const uint32_t antennas = 4;
    std::vector<float> delays = {0.3f, 1.7f, 3.3f, 4.9f};

    auto signal = generate_cw_flat(antennas, points, fs, freq);

    processor.SetDelays(delays);
    processor.SetSampleRate(fs);

    // GPU
    auto result = processor.ProcessFromCPU(signal, antennas, points);

    std::vector<std::complex<float>> output(
        static_cast<size_t>(antennas) * points);
    hipMemcpyDtoH(output.data(), result.data,
                    output.size() * sizeof(std::complex<float>));
    hipFree(result.data);

    // CPU reference
    auto input_2d = flat_to_2d(signal, antennas, points);
    processor.SetDelays(delays);
    auto cpu_ref = processor.ProcessCpu(input_2d, antennas, points);

    float max_err = 0.0f;
    for (uint32_t a = 0; a < antennas; ++a) {
      for (uint32_t n = 0; n < points; ++n) {
        float e = std::abs(output[a * points + n] - cpu_ref[a][n]);
        if (e > max_err) max_err = e;
      }
    }

    bool passed = (max_err < 1e-2f);
    std::ostringstream oss;
    oss << "Test 4 (multi-antenna 4x): max_err = " << std::scientific
        << std::setprecision(2) << max_err
        << (passed ? " PASSED" : " FAILED (expected < 1e-2)");
    con.Print(gpu_id, "LchFarrow[ROCm]", oss.str());
    return passed;
  } catch (const std::exception& e) {
    con.Print(gpu_id, "LchFarrow[ROCm]",
              "[X] Multi-Antenna EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Start();
  int gpu_id = 0;

  con.Print(gpu_id, "LchFarrow[ROCm]", "");
  con.Print(gpu_id, "LchFarrow[ROCm]", "============================================");
  con.Print(gpu_id, "LchFarrow[ROCm]", "  LchFarrowROCm Tests (HIP)");
  con.Print(gpu_id, "LchFarrow[ROCm]", "============================================");

  int passed = 0;
  int total = 4;

  if (test_zero_delay(con, gpu_id))      ++passed;
  if (test_integer_delay(con, gpu_id))   ++passed;
  if (test_fractional_delay(con, gpu_id)) ++passed;
  if (test_multi_antenna(con, gpu_id))   ++passed;

  con.Print(gpu_id, "LchFarrow[ROCm]",
            "Results: " + std::to_string(passed) + "/" +
            std::to_string(total) + " passed");
  con.Print(gpu_id, "LchFarrow[ROCm]", "============================================");
  con.Print(gpu_id, "LchFarrow[ROCm]", "");
  con.WaitEmpty();
}

#else  // !ENABLE_ROCM

inline void run() {
  std::cout << "\n[test_lch_farrow_rocm] SKIPPED: ENABLE_ROCM not defined\n";
}

#endif  // ENABLE_ROCM

}  // namespace test_lch_farrow_rocm
