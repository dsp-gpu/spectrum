#pragma once

/**
 * @file spectrum_profiling.hpp
 * @brief Локальные данные профилирования модуля fft_func (spectrum)
 *
 * Для централизованного профилирования (JSON, MD) используется GPUProfiler
 * с OpenCLProfilingData. ProfilingData — локальный формат для GetProfilingData().
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

namespace antenna_fft {

/// Локальные данные профилирования (время в мс) для GetProfilingData()
struct ProfilingData {
    double upload_time_ms = 0.0;
    double fft_time_ms = 0.0;
    double post_kernel_time_ms = 0.0;
    double download_time_ms = 0.0;
    double total_time_ms = 0.0;
};

}  // namespace antenna_fft
