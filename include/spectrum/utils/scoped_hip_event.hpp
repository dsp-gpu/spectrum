#pragma once

/**
 * @file scoped_hip_event.hpp
 * @brief ScopedHipEvent — RAII-обёртка над hipEvent_t (exception-safe)
 *
 * Зачем:
 *   В hot-path (FFT, filters, lch_farrow) событиями профилируются 3-6 стадий
 *   подряд. Если между hipEventCreate и последним hipEventDestroy
 *   что-то бросает исключение (hipfftExecC2C, kernel launch, runtime_error),
 *   ранее созданные события утекают.
 *
 *   ScopedHipEvent гарантирует hipEventDestroy в деструкторе через RAII.
 *
 * Использование:
 *   ScopedHipEvent ev_up_s, ev_up_e;
 *   ev_up_s.Create(); ev_up_e.Create();
 *   hipEventRecord(ev_up_s.get(), stream);
 *   UploadData(...);
 *   hipEventRecord(ev_up_e.get(), stream);
 *   // При исключении — события корректно освобождаются
 *
 * Move-only: копирование запрещено, перемещение передаёт владение.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-14
 */

#if ENABLE_ROCM

#include <hip/hip_runtime.h>

namespace fft_processor {

class ScopedHipEvent {
public:
  ScopedHipEvent() = default;

  ~ScopedHipEvent() {
    if (event_) {
      hipEventDestroy(event_);
    }
  }

  ScopedHipEvent(const ScopedHipEvent&) = delete;
  ScopedHipEvent& operator=(const ScopedHipEvent&) = delete;

  ScopedHipEvent(ScopedHipEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  ScopedHipEvent& operator=(ScopedHipEvent&& other) noexcept {
    if (this != &other) {
      if (event_) hipEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  /// Создать hipEvent_t. Повторный вызов — сначала уничтожит старое.
  /// @return hipError_t из hipEventCreate
  hipError_t Create() {
    if (event_) {
      hipEventDestroy(event_);
      event_ = nullptr;
    }
    return hipEventCreate(&event_);
  }

  hipEvent_t get() const { return event_; }
  bool valid() const { return event_ != nullptr; }

private:
  hipEvent_t event_ = nullptr;
};

}  // namespace fft_processor

#endif  // ENABLE_ROCM
