#pragma once

// ============================================================================
// rocm_profiling_helpers — конвертеры hipEvent / wall-clock в ROCmProfilingData
//
// ЧТО:    Две inline-функции в namespace fft_func_utils:
//           - MakeROCmDataFromEvents(ev_start, ev_end, kind, op) — для GPU-замеров;
//           - MakeROCmDataFromClock(t_start, t_end, kind, op)    — для CPU-операций.
//         Возвращают готовый ROCmProfilingData для подачи в ProfilingFacade::Record.
//
// ЗАЧЕМ:  FFTProcessorROCm, SpectrumProcessorROCm, фильтры, lch_farrow_rocm —
//         все одинаково собирают hipEventElapsedTime и заполняют структуру.
//         Раньше это копипастилось в каждом классе. Helpers устраняют дублирование.
//
// ПОЧЕМУ: - Inline + header-only → zero overhead, один include во всех клиентах.
//         - Read-only по событиям: НЕ вызывают hipEventDestroy (с 2026-04-15).
//           Раньше делался destroy → double-free с ScopedHipEvent. Сейчас
//           владение остаётся за caller'ом (обычно ScopedHipEvent через RAII).
//         - Свободные функции (не методы класса) → caller не плодит instance,
//           использует напрямую без singleton/factory boilerplate.
//
// Использование:
//   drv_gpu_lib::ScopedHipEvent ev_s, ev_e;
//   ev_s.CreateOrThrow(); ev_e.CreateOrThrow();
//   hipEventRecord(ev_s.get(), stream);
//   // ... kernel launch ...
//   hipEventRecord(ev_e.get(), stream);
//   auto data = fft_func_utils::MakeROCmDataFromEvents(
//       ev_s.get(), ev_e.get(), 0, "FFT_C2C");
//   ProfilingFacade::GetInstance().Record(gpu_id, "spectrum", "fft", data);
//
// История:
//   - Создан:  2026-03-22 (устранение копипасты в fft_func классах)
//   - Изменён: 2026-04-15 (убран hipEventDestroy — все клиенты на ScopedHipEvent)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/profiling_types.hpp>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdint>

namespace fft_func_utils {

/// Конвертирует пару hipEvent_t → ROCmProfilingData (для GPU-замеров).
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

/// Конвертирует пару wall-clock timestamp'ов → ROCmProfilingData (для синхронных CPU-операций).
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
