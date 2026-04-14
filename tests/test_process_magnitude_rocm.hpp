#pragma once

/**
 * @file test_process_magnitude_rocm.hpp
 * @brief Tests for ComplexToMagPhaseROCm::ProcessMagnitude / ProcessMagnitudeToGPU
 *
 * ✅ MIGRATED to test_utils (2026-03-23)
 *
 * Tests:
 * 1. gpu_input_no_norm     — GPU void* input, norm_coeff=1
 * 2. managed_norm_by_n     — hipMallocManaged, norm_coeff=-1 (÷N)
 * 3. norm_zero_signal      — norm_coeff=0, zero signal
 * 4. to_gpu                — ProcessMagnitudeToGPU output stays on GPU
 * 5. multi_beam_managed    — 4 beams × 4096 pts (managed memory)
 * 6. to_buffer             — ProcessMagnitudeToBuffer zero allocations
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-11 (migrated 2026-03-23)
 */

#if ENABLE_ROCM

#include <spectrum/complex_to_mag_phase_rocm.hpp>
#include "test_helpers_rocm.hpp"
#include <core/backends/rocm/rocm_backend.hpp>

#include "modules/test_utils/test_utils.hpp"

#include <hip/hip_runtime.h>
#include <vector>
#include <complex>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace test_process_magnitude_rocm {

using namespace fft_processor;
using namespace drv_gpu_lib;
using namespace gpu_test_utils;
using namespace test_helpers_rocm;

// Module-specific helpers

inline std::vector<std::complex<float>> MakeSinusoid(
    size_t n, float amp = 1.0f, float freq = 1000.0f, float fs = 44100.0f) {
  std::vector<std::complex<float>> data(n);
  for (size_t i = 0; i < n; ++i) {
    float ph = 2.0f * static_cast<float>(M_PI) * freq * static_cast<float>(i) / fs;
    data[i] = {amp * std::cos(ph), amp * std::sin(ph)};
  }
  return data;
}

inline std::vector<float> RefMagnitudes(const std::vector<std::complex<float>>& d, float inv_n) {
  std::vector<float> r(d.size());
  for (size_t i = 0; i < d.size(); ++i) r[i] = std::abs(d[i]) * inv_n;
  return r;
}

inline bool AllClose(const std::vector<float>& a, const std::vector<float>& b, float atol = 1e-4f) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::abs(a[i] - b[i]) > atol + 1e-6f * std::abs(b[i])) return false;
  return true;
}

inline void run() {
  int gpu_id = 0;

  ROCmBackend backend;
  backend.Initialize(gpu_id);
  ComplexToMagPhaseROCm proc(&backend);

  TestRunner runner(&backend, "ProcMag ROCm", gpu_id);

  // ── Test 1: GPU void* input, norm=1 ───────────────────────────

  runner.test("gpu_input_no_norm", [&]() {
    constexpr uint32_t N = 4096;
    auto src = MakeSinusoid(N, 2.5f);

    void* gpu_ptr = nullptr;
    (void)hipMalloc(&gpu_ptr, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_ptr, src.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1; params.n_point = N; params.norm_coeff = 1.0f;

    auto results = proc.ProcessMagnitude(gpu_ptr, params, N * sizeof(std::complex<float>));
    (void)hipFree(gpu_ptr);

    auto ref = RefMagnitudes(src, 1.0f);
    bool ok = results.size() == 1 && AllClose(results[0].magnitude, ref);
    return ValidationResult{ok, "allclose", ok ? 0.0 : 1.0, 0.0, ""};
  });

  // ── Test 2: Managed memory + norm=-1 (÷N) ────────────────────

  runner.test("managed_norm_by_n", [&]() {
    constexpr uint32_t N = 2048;
    auto managed = AllocateManagedForTest(N * sizeof(std::complex<float>));
    auto* ptr = static_cast<std::complex<float>*>(managed);
    auto src = MakeSinusoid(N, 3.0f);
    for (size_t i = 0; i < N; ++i) ptr[i] = src[i];

    auto input = MakeManagedInput(managed, 1, N);
    MagPhaseParams params;
    params.beam_count = 1; params.n_point = N; params.norm_coeff = -1.0f;

    auto results = proc.ProcessMagnitude(input.data, params, input.gpu_memory_bytes);
    (void)hipFree(managed);

    auto ref = RefMagnitudes(src, 1.0f / N);
    bool ok = results.size() == 1 && AllClose(results[0].magnitude, ref);
    return ValidationResult{ok, "allclose", ok ? 0.0 : 1.0, 0.0, ""};
  });

  // ── Test 3: norm=0, zero signal ───────────────────────────────

  runner.test("norm_zero_signal", [&]() {
    constexpr uint32_t N = 512;
    std::vector<std::complex<float>> zeros(N, {0, 0});

    void* gpu_ptr = nullptr;
    (void)hipMalloc(&gpu_ptr, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_ptr, zeros.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1; params.n_point = N; params.norm_coeff = 0.0f;

    auto results = proc.ProcessMagnitude(gpu_ptr, params);
    (void)hipFree(gpu_ptr);

    float sum = 0;
    if (results.size() == 1) for (float v : results[0].magnitude) sum += std::abs(v);
    return ValidationResult{sum < 1e-6f, "all_zero", sum, 0.0, ""};
  });

  // ── Test 4: ProcessMagnitudeToGPU ─────────────────────────────

  runner.test("to_gpu", [&]() {
    constexpr uint32_t N = 1024;
    auto src = MakeSinusoid(N);

    void* gpu_in = nullptr;
    (void)hipMalloc(&gpu_in, N * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_in, src.data(), N * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    MagPhaseParams params;
    params.beam_count = 1; params.n_point = N; params.norm_coeff = 1.0f;

    void* gpu_out = proc.ProcessMagnitudeToGPU(gpu_in, params);
    (void)hipFree(gpu_in);

    std::vector<float> result(N);
    (void)hipMemcpy(result.data(), gpu_out, N * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipFree(gpu_out);

    bool ok = AllClose(result, RefMagnitudes(src, 1.0f));
    return ValidationResult{ok, "allclose", ok ? 0.0 : 1.0, 0.0, ""};
  });

  // ── Test 5: Multi-beam managed 4×4096 ─────────────────────────

  runner.test("multi_beam_managed", [&]() {
    constexpr uint32_t B = 4, N = 4096;
    constexpr size_t T = B * N;

    auto managed = AllocateManagedForTest(T * sizeof(std::complex<float>));
    auto* ptr = static_cast<std::complex<float>*>(managed);
    std::vector<std::complex<float>> src(T);
    for (uint32_t b = 0; b < B; ++b) {
      float amp = (b + 1) * 0.5f;
      auto beam = MakeSinusoid(N, amp);
      for (uint32_t k = 0; k < N; ++k) { ptr[b*N+k] = beam[k]; src[b*N+k] = beam[k]; }
    }

    auto input = MakeManagedInput(managed, B, N);
    MagPhaseParams params;
    params.beam_count = B; params.n_point = N; params.norm_coeff = -1.0f;

    auto results = proc.ProcessMagnitude(input.data, params, input.gpu_memory_bytes);
    (void)hipFree(managed);

    bool ok = results.size() == B;
    float inv_n = 1.0f / N;
    for (uint32_t b = 0; b < B && ok; ++b) {
      std::vector<std::complex<float>> beam_src(src.begin() + b*N, src.begin() + (b+1)*N);
      ok = AllClose(results[b].magnitude, RefMagnitudes(beam_src, inv_n));
    }
    return ValidationResult{ok, "allclose", ok ? 0.0 : 1.0, 0.0, ""};
  });

  // ── Test 6: ProcessMagnitudeToBuffer — zero allocs ────────────

  runner.test("to_buffer", [&]() {
    constexpr uint32_t B = 2, N = 2048;
    constexpr size_t T = B * N;
    auto src = MakeSinusoid(T, 1.5f);

    void* gpu_in = nullptr;
    (void)hipMalloc(&gpu_in, T * sizeof(std::complex<float>));
    (void)hipMemcpy(gpu_in, src.data(), T * sizeof(std::complex<float>), hipMemcpyHostToDevice);

    void* gpu_out = nullptr;
    (void)hipMalloc(&gpu_out, T * sizeof(float));

    MagPhaseParams params;
    params.beam_count = B; params.n_point = N; params.norm_coeff = 0.0f;

    proc.ProcessMagnitudeToBuffer(gpu_in, gpu_out, params);
    void* gpu_ref = proc.ProcessMagnitudeToGPU(gpu_in, params);

    std::vector<float> buf(T), ref(T);
    (void)hipMemcpy(buf.data(), gpu_out, T * sizeof(float), hipMemcpyDeviceToHost);
    (void)hipMemcpy(ref.data(), gpu_ref, T * sizeof(float), hipMemcpyDeviceToHost);

    (void)hipFree(gpu_in);
    (void)hipFree(gpu_out);
    (void)hipFree(gpu_ref);

    bool ok = AllClose(buf, ref);
    return ValidationResult{ok, "buf_vs_ref", ok ? 0.0 : 1.0, 0.0, ""};
  });

  runner.print_summary();
}

}  // namespace test_process_magnitude_rocm

#endif  // ENABLE_ROCM
