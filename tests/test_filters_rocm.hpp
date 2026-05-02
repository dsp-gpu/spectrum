#pragma once

// ============================================================================
// test_filters_rocm — тесты FirFilterROCm + IirFilterROCm (ROCm)
//
// ЧТО:    GPU vs CPU reference, multichannel, void* input.
//         FIR: 64/256-tap. IIR: Butterworth multi-section.
// ЗАЧЕМ:  Фильтры — базовая DSP-операция. Ошибки в фазе/амплитуде
//         дают неверный спектр после FFT.
// ПОЧЕМУ: ENABLE_ROCM. Эталон — CPU reference implementation.
//
// История: Создан: 2026-04-12
// ============================================================================

/**
 * @file test_filters_rocm.hpp
 * @brief Тесты FirFilterROCm + IirFilterROCm — GPU vs CPU reference, multichannel, void* input
 * @note Test fixture, не публичный API. Запускается через all_test.hpp.
 *       Компилируется ТОЛЬКО при ENABLE_ROCM=1. FIR 64/256-tap, IIR Butterworth multi-section.
 */

#include <vector>
#include <complex>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if ENABLE_ROCM

#include <spectrum/filters/fir_filter_rocm.hpp>
#include <spectrum/filters/iir_filter_rocm.hpp>
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/backends/rocm/rocm_core.hpp>

// test_utils — unified test infrastructure
#include "modules/test_utils/test_utils.hpp"

namespace test_filters_rocm {

using namespace filters;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;

// ═══════════════════════════════════════════════════════════════════════════
// Pre-computed coefficients
// ═══════════════════════════════════════════════════════════════════════════

// scipy.signal.firwin(64, 0.1, window='hamming')
static const std::vector<float> kTestFirCoeffs64 = {
  -0.000157f, -0.000332f, -0.000459f, -0.000399f, -0.000000f,
   0.000850f,  0.002169f,  0.003849f,  0.005627f,  0.007078f,
   0.007634f,  0.006647f,  0.003540f, -0.002061f, -0.010202f,
  -0.020558f, -0.032375f, -0.044476f, -0.055375f, -0.063450f,
  -0.067076f, -0.064848f, -0.055800f, -0.039588f, -0.016579f,
   0.012488f,  0.046355f,  0.083311f,  0.121273f,  0.157982f,
   0.191169f,  0.218728f,  0.238881f,  0.250291f,  0.252151f,
   0.244226f,  0.226902f,  0.201177f,  0.168596f,  0.131141f,
   0.091041f,  0.050605f,  0.012058f, -0.022485f, -0.051250f,
  -0.073004f, -0.087189f, -0.093864f, -0.093644f, -0.087573f,
  -0.076953f, -0.063233f, -0.047833f, -0.032032f, -0.016901f,
  -0.003260f,  0.008352f,  0.017614f,  0.024418f,  0.028835f,
   0.031065f,  0.031385f,  0.030094f,  0.027480f
};

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate test signal (CW 100Hz + CW 5000Hz)
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> generate_test_signal(
    uint32_t channels, uint32_t points, float sample_rate)
{
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> signal(total);

  float f_low  = 100.0f;
  float f_high = 5000.0f;

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;
    float phase_offset = static_cast<float>(ch) * 0.1f;
    for (uint32_t n = 0; n < points; ++n) {
      float t = static_cast<float>(n) / sample_rate;
      float re = std::cos(2.0f * static_cast<float>(M_PI) * f_low * t + phase_offset)
               + 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * f_high * t);
      float im = std::sin(2.0f * static_cast<float>(M_PI) * f_low * t + phase_offset)
               + 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * f_high * t);
      signal[base + n] = {re, im};
    }
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
    con.Print(gpu_id, "Filters[ROCm]", "[!] No ROCm devices found -- skipping tests");
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  FirFilterROCm fir(&backend);
  IirFilterROCm iir(&backend);

  TestRunner runner(&backend, "Filters[ROCm]", gpu_id);

  // ── Test 1: FIR basic — 64 taps, 8ch x 4096pts ─────────────────

  runner.test("fir_basic", [&]() {
    fir.SetCoefficients(kTestFirCoeffs64);

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = fir.ProcessCpu(signal, channels, points);
    auto gpu_result = fir.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "fir_64tap_8ch");
  });

  // ── Test 2: FIR large — 256 taps, 16ch x 8192pts ───────────────

  runner.test("fir_large", [&]() {
    // Generate simple 256-tap LP coefficients (sinc * hamming)
    const uint32_t num_taps = 256;
    std::vector<float> coeffs(num_taps);
    float fc = 0.1f;
    int M = static_cast<int>(num_taps) - 1;
    for (uint32_t i = 0; i < num_taps; ++i) {
      float n_offset = static_cast<float>(i) - static_cast<float>(M) / 2.0f;
      if (std::abs(n_offset) < 1e-6f) {
        coeffs[i] = 2.0f * fc;
      } else {
        coeffs[i] = std::sin(2.0f * static_cast<float>(M_PI) * fc * n_offset)
                     / (static_cast<float>(M_PI) * n_offset);
      }
      // Hamming window
      float w = 0.54f - 0.46f * std::cos(2.0f * static_cast<float>(M_PI) *
                                           static_cast<float>(i) / static_cast<float>(M));
      coeffs[i] *= w;
    }

    fir.SetCoefficients(coeffs);

    const uint32_t channels    = 16;
    const uint32_t points      = 8192;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = fir.ProcessCpu(signal, channels, points);
    auto gpu_result = fir.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "fir_256tap_16ch");
  });

  // ── Test 3: FIR Process (void* input) — direct GPU pointer ──────

  runner.test("fir_gpu_ptr", [&]() {
    fir.SetCoefficients(kTestFirCoeffs64);

    const uint32_t channels = 4;
    const uint32_t points   = 2048;
    const size_t   total    = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = fir.ProcessCpu(signal, channels, points);

    // Manually upload to GPU
    hipStream_t stream = static_cast<hipStream_t>(backend.GetNativeQueue());
    size_t data_size = total * sizeof(std::complex<float>);
    void* input_ptr = nullptr;
    hipMalloc(&input_ptr, data_size);
    hipMemcpyHtoDAsync(input_ptr,
                        const_cast<std::complex<float>*>(signal.data()),
                        data_size, stream);
    hipStreamSynchronize(stream);

    // Process with void* ptr
    auto gpu_result = fir.Process(input_ptr, channels, points);
    hipFree(input_ptr);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "fir_void_ptr");
  });

  // ── Test 4: IIR basic — Butterworth 2nd order (1 section) ───────

  runner.test("iir_basic", [&]() {
    // Butterworth 2nd order LP, fc=0.1: butter(2, 0.1, output='sos')
    BiquadSection sec0;
    sec0.b0 = 0.02008337f;
    sec0.b1 = 0.04016673f;
    sec0.b2 = 0.02008337f;
    sec0.a1 = -1.56101808f;
    sec0.a2 = 0.64135154f;
    iir.SetBiquadSections({sec0});

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);
    auto gpu_result = iir.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "iir_1sec_8ch");
  });

  // ── Test 5: IIR multi-section — 4th order Butterworth ───────────

  runner.test("iir_multi_section", [&]() {
    // Butterworth 4th order LP, fc=0.1: butter(4, 0.1, output='sos')
    BiquadSection sec0;
    sec0.b0 = 0.00482434f;
    sec0.b1 = 0.00964869f;
    sec0.b2 = 0.00482434f;
    sec0.a1 = -1.64745998f;
    sec0.a2 = 0.70089678f;

    BiquadSection sec1;
    sec1.b0 = 0.00482434f;
    sec1.b1 = 0.00964869f;
    sec1.b2 = 0.00482434f;
    sec1.a1 = -1.47454166f;
    sec1.a2 = 0.58712864f;

    iir.SetBiquadSections({sec0, sec1});

    const uint32_t channels    = 8;
    const uint32_t points      = 4096;
    const size_t   total       = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);
    auto gpu_result = iir.ProcessFromCPU(signal, channels, points);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "iir_2sec_8ch");
  });

  // ── Test 6: IIR Process (void* input) — direct GPU pointer ──────

  runner.test("iir_gpu_ptr", [&]() {
    BiquadSection sec0;
    sec0.b0 = 0.02008337f;
    sec0.b1 = 0.04016673f;
    sec0.b2 = 0.02008337f;
    sec0.a1 = -1.56101808f;
    sec0.a2 = 0.64135154f;
    iir.SetBiquadSections({sec0});

    const uint32_t channels = 4;
    const uint32_t points   = 2048;
    const size_t   total    = static_cast<size_t>(channels) * points;

    auto signal = generate_test_signal(channels, points, 50000.0f);
    auto cpu_result = iir.ProcessCpu(signal, channels, points);

    // Manually upload
    hipStream_t stream = static_cast<hipStream_t>(backend.GetNativeQueue());
    size_t data_size = total * sizeof(std::complex<float>);
    void* input_ptr = nullptr;
    hipMalloc(&input_ptr, data_size);
    hipMemcpyHtoDAsync(input_ptr,
                        const_cast<std::complex<float>*>(signal.data()),
                        data_size, stream);
    hipStreamSynchronize(stream);

    auto gpu_result = iir.Process(input_ptr, channels, points);
    hipFree(input_ptr);

    auto gpu_data = ReadHipBuffer<std::complex<float>>(
        backend.GetNativeQueue(), gpu_result.data, total);

    return AbsError(gpu_data.data(), cpu_result.data(), total,
                    tolerance::kFilter, "iir_void_ptr");
  });

  // ── Summary ─────────────────────────────────────────────────────

  runner.print_summary();
}

}  // namespace test_filters_rocm

#else  // !ENABLE_ROCM

namespace test_filters_rocm {

inline void run() {
  // SKIPPED: ENABLE_ROCM not defined
}

}  // namespace test_filters_rocm

#endif  // ENABLE_ROCM
