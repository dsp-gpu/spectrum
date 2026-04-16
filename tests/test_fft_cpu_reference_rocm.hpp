#pragma once

/**
 * @file test_fft_cpu_reference_rocm.hpp
 * @brief GPU FFT vs CPU reference (pocketfft) — валидация точности
 *
 * Тесты:
 *   1. c2c_sinusoid     — complex sinusoid: GPU hipFFT C2C vs CPU pocketfft C2C
 *   2. r2c_real_signal  — real signal: GPU (pad zero-imag + C2C) vs CPU pocketfft R2C
 *   3. c2c_multi_beam   — multi-beam: 4 луча с разными частотами
 *   4. magnitude_match  — |X_gpu| vs |X_cpu| поэлементно
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-16
 */

#if ENABLE_ROCM

#include <spectrum/fft_processor_rocm.hpp>
#include <spectrum/utils/cpu_fft.hpp>
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/services/console_output.hpp>

#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_fft_cpu_reference {

using namespace fft_processor;
using namespace drv_gpu_lib;
using namespace spectrum_utils;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

inline std::vector<std::complex<float>> gen_complex_tone(
    float freq, float sample_rate, size_t n_point, float amplitude = 1.0f) {
  std::vector<std::complex<float>> data(n_point);
  for (size_t i = 0; i < n_point; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    float ph = 2.0f * static_cast<float>(M_PI) * freq * t;
    data[i] = {amplitude * std::cos(ph), amplitude * std::sin(ph)};
  }
  return data;
}

inline std::vector<float> gen_real_tone(
    float freq, float sample_rate, size_t n_point, float amplitude = 1.0f) {
  std::vector<float> data(n_point);
  for (size_t i = 0; i < n_point; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    data[i] = amplitude * std::sin(2.0f * static_cast<float>(M_PI) * freq * t);
  }
  return data;
}

/// Максимальная абсолютная ошибка между двумя комплексными векторами
inline float max_abs_error(const std::vector<std::complex<float>>& a,
                           const std::vector<std::complex<float>>& b) {
  size_t n = std::min(a.size(), b.size());
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(a[i] - b[i]));
  }
  return max_err;
}

/// Максимальная относительная ошибка магнитуд (игнорируем bins с малой амплитудой)
inline float max_relative_mag_error(const std::vector<float>& gpu_mag,
                                    const std::vector<float>& cpu_mag,
                                    float threshold = 1e-3f) {
  size_t n = std::min(gpu_mag.size(), cpu_mag.size());
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    if (cpu_mag[i] > threshold) {
      float rel = std::fabs(gpu_mag[i] - cpu_mag[i]) / cpu_mag[i];
      max_err = std::max(max_err, rel);
    }
  }
  return max_err;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: C2C — complex sinusoid, GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_c2c_sinusoid(ConsoleOutput& con, int gpu_id,
                               IBackend* backend) {
  try {
    const float freq = 100.0f;
    const float sample_rate = 1000.0f;
    const uint32_t n_point = 1024;

    auto data = gen_complex_tone(freq, sample_rate, n_point);

    // GPU FFT (hipFFT C2C)
    FFTProcessorROCm fft(backend);
    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = FFTOutputMode::COMPLEX;
    auto gpu_results = fft.ProcessComplex(data, params);

    if (gpu_results.empty()) {
      con.Print(gpu_id, "FFT-Ref", "[X] c2c_sinusoid: empty GPU result");
      return false;
    }

    // CPU FFT (pocketfft C2C)
    auto cpu_spectrum = CpuFFT::ForwardC2C(data);

    // Сравнение: GPU nFFT может быть >= n_point (zero-padded to power of 2)
    uint32_t nFFT = gpu_results[0].nFFT;
    std::vector<std::complex<float>> gpu_spectrum = gpu_results[0].spectrum;

    // Если GPU сделал zero-pad, CPU тоже нужно до nFFT
    if (nFFT > n_point) {
      std::vector<std::complex<float>> padded(nFFT, {0, 0});
      std::copy(data.begin(), data.end(), padded.begin());
      cpu_spectrum = CpuFFT::ForwardC2C(padded);
    }

    float err = max_abs_error(gpu_spectrum, cpu_spectrum);

    // Нормируем на амплитуду пика (~N для единичной синусоиды)
    float peak_amp = static_cast<float>(nFFT);
    float rel_err = err / peak_amp;

    std::ostringstream oss;
    oss << "c2c_sinusoid: abs_err=" << std::scientific << std::setprecision(2) << err
        << " rel_err=" << rel_err
        << (rel_err < 1e-5f ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FFT-Ref", oss.str());
    return rel_err < 1e-5f;

  } catch (const std::exception& e) {
    con.Print(gpu_id, "FFT-Ref", "[X] c2c_sinusoid EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: R2C — real signal, GPU vs CPU
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_r2c_real_signal(ConsoleOutput& con, int gpu_id,
                                  IBackend* backend) {
  try {
    const float freq = 250.0f;
    const float sample_rate = 2000.0f;
    const uint32_t n_point = 2048;

    auto real_data = gen_real_tone(freq, sample_rate, n_point);

    // CPU R2C (pocketfft) — полный спектр
    auto cpu_spectrum = CpuFFT::ForwardR2C_Full(real_data);

    // GPU: оборачиваем real в complex (imag=0) и делаем C2C
    std::vector<std::complex<float>> complex_data(n_point);
    for (size_t i = 0; i < n_point; ++i) {
      complex_data[i] = {real_data[i], 0.0f};
    }

    FFTProcessorROCm fft(backend);
    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = FFTOutputMode::COMPLEX;
    auto gpu_results = fft.ProcessComplex(complex_data, params);

    if (gpu_results.empty()) {
      con.Print(gpu_id, "FFT-Ref", "[X] r2c_real_signal: empty GPU result");
      return false;
    }

    uint32_t nFFT = gpu_results[0].nFFT;

    // Если GPU pad'ит, pad CPU тоже
    if (nFFT > n_point) {
      std::vector<float> padded(nFFT, 0.0f);
      std::copy(real_data.begin(), real_data.end(), padded.begin());
      cpu_spectrum = CpuFFT::ForwardR2C_Full(padded);
    }

    // Сравнение магнитуд (фаза может отличаться из-за padding)
    auto gpu_mag = CpuFFT::Magnitude(gpu_results[0].spectrum);
    auto cpu_mag = CpuFFT::Magnitude(cpu_spectrum);

    float rel_err = max_relative_mag_error(gpu_mag, cpu_mag, 1.0f);

    std::ostringstream oss;
    oss << "r2c_real_signal: max_rel_mag_err=" << std::scientific << std::setprecision(2) << rel_err
        << (rel_err < 1e-4f ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FFT-Ref", oss.str());
    return rel_err < 1e-4f;

  } catch (const std::exception& e) {
    con.Print(gpu_id, "FFT-Ref", "[X] r2c_real_signal EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: C2C multi-beam — 4 луча с разными частотами
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_c2c_multi_beam(ConsoleOutput& con, int gpu_id,
                                 IBackend* backend) {
  try {
    const uint32_t beam_count = 4;
    const uint32_t n_point = 512;
    const float sample_rate = 1000.0f;
    const float freqs[] = {50.0f, 125.0f, 200.0f, 375.0f};

    // Генерация multi-beam
    std::vector<std::complex<float>> data(beam_count * n_point);
    for (uint32_t b = 0; b < beam_count; ++b) {
      auto tone = gen_complex_tone(freqs[b], sample_rate, n_point);
      std::copy(tone.begin(), tone.end(), data.begin() + b * n_point);
    }

    // GPU
    FFTProcessorROCm fft(backend);
    FFTProcessorParams params;
    params.beam_count = beam_count;
    params.n_point = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = FFTOutputMode::COMPLEX;
    auto gpu_results = fft.ProcessComplex(data, params);

    if (gpu_results.size() != beam_count) {
      con.Print(gpu_id, "FFT-Ref", "[X] c2c_multi_beam: beam_count mismatch");
      return false;
    }

    float max_err = 0;
    for (uint32_t b = 0; b < beam_count; ++b) {
      uint32_t nFFT = gpu_results[b].nFFT;

      // CPU для этого луча (с padding если надо)
      auto beam_data = gen_complex_tone(freqs[b], sample_rate, n_point);
      if (nFFT > n_point) {
        beam_data.resize(nFFT, {0, 0});
      }
      auto cpu_spectrum = CpuFFT::ForwardC2C(beam_data);

      float err = max_abs_error(gpu_results[b].spectrum, cpu_spectrum);
      float peak_amp = static_cast<float>(nFFT);
      max_err = std::max(max_err, err / peak_amp);
    }

    std::ostringstream oss;
    oss << "c2c_multi_beam: max_rel_err=" << std::scientific << std::setprecision(2) << max_err
        << (max_err < 1e-5f ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FFT-Ref", oss.str());
    return max_err < 1e-5f;

  } catch (const std::exception& e) {
    con.Print(gpu_id, "FFT-Ref", "[X] c2c_multi_beam EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Magnitude match — |X_gpu| vs |X_cpu|
// ═══════════════════════════════════════════════════════════════════════════

inline bool test_magnitude_match(ConsoleOutput& con, int gpu_id,
                                  IBackend* backend) {
  try {
    const uint32_t n_point = 1024;
    const float sample_rate = 4000.0f;

    // Multi-tone signal
    std::vector<std::complex<float>> data(n_point);
    for (size_t i = 0; i < n_point; ++i) {
      float t = static_cast<float>(i) / sample_rate;
      float re = std::cos(2.0f * static_cast<float>(M_PI) * 100.0f * t) +
                 0.5f * std::cos(2.0f * static_cast<float>(M_PI) * 500.0f * t) +
                 0.3f * std::cos(2.0f * static_cast<float>(M_PI) * 1200.0f * t);
      float im = std::sin(2.0f * static_cast<float>(M_PI) * 100.0f * t) +
                 0.5f * std::sin(2.0f * static_cast<float>(M_PI) * 500.0f * t) +
                 0.3f * std::sin(2.0f * static_cast<float>(M_PI) * 1200.0f * t);
      data[i] = {re, im};
    }

    // GPU MagPhase
    FFTProcessorROCm fft(backend);
    FFTProcessorParams params;
    params.beam_count = 1;
    params.n_point = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = FFTOutputMode::MAGNITUDE_PHASE;
    auto gpu_results = fft.ProcessMagPhase(data, params);

    if (gpu_results.empty()) {
      con.Print(gpu_id, "FFT-Ref", "[X] magnitude_match: empty GPU result");
      return false;
    }

    // CPU reference
    uint32_t nFFT = gpu_results[0].nFFT;
    auto padded = data;
    padded.resize(nFFT, {0, 0});
    auto cpu_spectrum = CpuFFT::ForwardC2C(padded);
    auto cpu_mag = CpuFFT::Magnitude(cpu_spectrum);

    // Сравнение
    float max_err = 0;
    for (size_t i = 0; i < std::min(gpu_results[0].magnitude.size(), cpu_mag.size()); ++i) {
      if (cpu_mag[i] > 1.0f) {
        float rel = std::fabs(gpu_results[0].magnitude[i] - cpu_mag[i]) / cpu_mag[i];
        max_err = std::max(max_err, rel);
      }
    }

    std::ostringstream oss;
    oss << "magnitude_match: max_rel_err=" << std::scientific << std::setprecision(2) << max_err
        << (max_err < 1e-4f ? " PASSED" : " FAILED");
    con.Print(gpu_id, "FFT-Ref", oss.str());
    return max_err < 1e-4f;

  } catch (const std::exception& e) {
    con.Print(gpu_id, "FFT-Ref", "[X] magnitude_match EXCEPTION: " + std::string(e.what()));
    return false;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

inline void run() {
  auto& con = ConsoleOutput::GetInstance();
  con.Start();
  int gpu_id = 0;

  int device_count = ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    con.Print(gpu_id, "FFT-Ref", "[!] No ROCm devices — skip");
    con.WaitEmpty();
    return;
  }

  ROCmBackend backend;
  backend.Initialize(gpu_id);

  con.Print(gpu_id, "FFT-Ref", "");
  con.Print(gpu_id, "FFT-Ref", "============================================");
  con.Print(gpu_id, "FFT-Ref", "  GPU FFT vs CPU pocketfft Reference Tests");
  con.Print(gpu_id, "FFT-Ref", "============================================");

  int passed = 0, total = 4;

  if (test_c2c_sinusoid(con, gpu_id, &backend))     ++passed;
  if (test_r2c_real_signal(con, gpu_id, &backend))   ++passed;
  if (test_c2c_multi_beam(con, gpu_id, &backend))    ++passed;
  if (test_magnitude_match(con, gpu_id, &backend))   ++passed;

  con.Print(gpu_id, "FFT-Ref",
      "Results: " + std::to_string(passed) + "/" + std::to_string(total) + " passed");
  con.Print(gpu_id, "FFT-Ref", "============================================");
  con.Print(gpu_id, "FFT-Ref", "");
  con.WaitEmpty();
}

#else  // !ENABLE_ROCM

inline void run() {
  // stub
}

#endif  // ENABLE_ROCM

}  // namespace test_fft_cpu_reference
