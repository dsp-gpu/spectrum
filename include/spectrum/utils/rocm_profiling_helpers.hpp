#pragma once

/**
 * @file rocm_profiling_helpers.hpp
 * @brief Shared helpers for converting HIP events / wall-clock to ROCmProfilingData
 *
 * Used by FFTProcessorROCm, SpectrumProcessorROCm, and other fft_func classes.
 * Eliminates copy-paste of MakeROCmDataFromEvents / MakeROCmDataFromClock.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-22
 */

#if ENABLE_ROCM

#include <core/services/profiling_types.hpp>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>

namespace fft_func_utils {

/// Convert hipEvent pair → ROCmProfilingData.
///
/// @note Read-only: НЕ вызывает hipEventDestroy.
///       Владение событиями остаётся за вызывающим (обычно ScopedHipEvent).
///       Раньше делался destroy — приводило к double-free с RAII-клиентами.
///       2026-04-15: destroy убран, все клиенты используют ScopedHipEvent.
inline drv_gpu_lib::ROCmProfilingData MakeROCmDataFromEvents(
    hipEvent_t ev_start, hipEvent_t ev_end,
    uint32_t kind, const char* op_string = "") {
  hipEventSynchronize(ev_end);
  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, ev_start, ev_end);

  drv_gpu_lib::ROCmProfilingData d{};
  uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_ms * 1e6f);
  d.queued_ns = 0; d.submit_ns = 0;
  d.start_ns = 0; d.end_ns = elapsed_ns; d.complete_ns = elapsed_ns;
  d.kind = kind; d.op_string = op_string;
  return d;
}

/// Convert wall-clock time pair → ROCmProfilingData (for synchronous CPU-side operations)
inline drv_gpu_lib::ROCmProfilingData MakeROCmDataFromClock(
    std::chrono::high_resolution_clock::time_point t_start,
    std::chrono::high_resolution_clock::time_point t_end,
    uint32_t kind, const char* op_string = "") {
  uint64_t elapsed_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
  drv_gpu_lib::ROCmProfilingData d{};
  d.queued_ns = 0; d.submit_ns = 0;
  d.start_ns = 0; d.end_ns = elapsed_ns; d.complete_ns = elapsed_ns;
  d.kind = kind; d.op_string = op_string;
  return d;
}

}  // namespace fft_func_utils

#endif  // ENABLE_ROCM
