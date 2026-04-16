#pragma once

/**
 * @file cpu_fft.hpp
 * @brief CpuFFT — CPU-reference FFT через pocketfft (header-only)
 *
 * Простая обёртка над pocketfft для:
 *   1. Валидации GPU-результатов в тестах (GPU vs CPU)
 *   2. Fallback для отладки (без GPU)
 *
 * Два режима:
 *   - C2C (complex-to-complex): вход complex<float>, выход complex<float>, размер N
 *   - R2C (real-to-complex):    вход float, выход complex<float>, размер N/2+1
 *
 * @note pocketfft: BSD license, header-only, Max-Planck-Society
 * @see third_party/pocketfft/pocketfft_hdronly.h
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-16
 */

#include <complex>
#include <vector>
#include <cstddef>
#include <stdexcept>

// pocketfft — single-header library, спрятан в third_party/
// Отключаем multithreading: CPU FFT используется только в тестах,
// один поток достаточен и избавляет от лишнего #include <thread>.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

namespace spectrum_utils {

// ════════════════════════════════════════════════════════════════════════════
// CpuFFT — статический класс-обёртка
// ════════════════════════════════════════════════════════════════════════════

class CpuFFT {
public:
  // ──────────────────────────────────────────────────────────────────────
  // C2C: complex<float> → complex<float>  (размер входа = размер выхода = N)
  // ──────────────────────────────────────────────────────────────────────

  /**
   * @brief Forward FFT: complex → complex
   * @param input  Вектор complex<float> длиной N
   * @return Вектор complex<float> длиной N (спектр)
   */
  static std::vector<std::complex<float>> ForwardC2C(
      const std::vector<std::complex<float>>& input)
  {
    if (input.empty())
      throw std::runtime_error("CpuFFT::ForwardC2C: empty input");

    const size_t N = input.size();
    std::vector<std::complex<float>> output(N);

    pocketfft::shape_t  shape  = { N };
    pocketfft::stride_t stride_in  = { static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
    pocketfft::stride_t stride_out = stride_in;
    pocketfft::shape_t  axes   = { 0 };

    pocketfft::c2c<float>(
        shape, stride_in, stride_out, axes,
        /*forward=*/true,
        input.data(), output.data(),
        /*fct=*/1.0f);

    return output;
  }

  /**
   * @brief Inverse FFT: complex → complex (с нормировкой 1/N)
   * @param input  Вектор complex<float> длиной N (спектр)
   * @return Вектор complex<float> длиной N (сигнал)
   */
  static std::vector<std::complex<float>> InverseC2C(
      const std::vector<std::complex<float>>& input)
  {
    if (input.empty())
      throw std::runtime_error("CpuFFT::InverseC2C: empty input");

    const size_t N = input.size();
    std::vector<std::complex<float>> output(N);

    pocketfft::shape_t  shape  = { N };
    pocketfft::stride_t stride_in  = { static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
    pocketfft::stride_t stride_out = stride_in;
    pocketfft::shape_t  axes   = { 0 };

    pocketfft::c2c<float>(
        shape, stride_in, stride_out, axes,
        /*forward=*/false,
        input.data(), output.data(),
        /*fct=*/1.0f / static_cast<float>(N));

    return output;
  }

  // ──────────────────────────────────────────────────────────────────────
  // R2C: float → complex<float>  (вход N, выход N/2+1)
  // ──────────────────────────────────────────────────────────────────────

  /**
   * @brief Forward FFT: real → complex (half-spectrum)
   * @param input  Вектор float длиной N
   * @return Вектор complex<float> длиной N/2+1
   *
   * Реальный сигнал → спектр с эрмитовой симметрией.
   * Возвращает только положительные частоты (0..N/2).
   */
  static std::vector<std::complex<float>> ForwardR2C(
      const std::vector<float>& input)
  {
    if (input.empty())
      throw std::runtime_error("CpuFFT::ForwardR2C: empty input");

    const size_t N = input.size();
    const size_t N_out = N / 2 + 1;
    std::vector<std::complex<float>> output(N_out);

    pocketfft::shape_t  shape_in  = { N };
    pocketfft::stride_t stride_in  = { static_cast<ptrdiff_t>(sizeof(float)) };
    pocketfft::stride_t stride_out = { static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };

    pocketfft::r2c<float>(
        shape_in, stride_in, stride_out,
        /*axis=*/0,
        /*forward=*/true,
        input.data(), output.data(),
        /*fct=*/1.0f);

    return output;
  }

  /**
   * @brief Forward FFT: real → full complex spectrum (N элементов)
   * @param input  Вектор float длиной N
   * @return Вектор complex<float> длиной N (полный спектр с зеркальной частью)
   *
   * Как ForwardR2C, но достраивает отрицательные частоты:
   *   output[k] = conj(output[N-k]) для k > N/2
   *
   * Формат совместим с hipFFT C2C output.
   */
  static std::vector<std::complex<float>> ForwardR2C_Full(
      const std::vector<float>& input)
  {
    auto half = ForwardR2C(input);
    const size_t N = input.size();
    std::vector<std::complex<float>> full(N);

    // Копируем положительные частоты
    for (size_t i = 0; i < half.size(); ++i) {
      full[i] = half[i];
    }
    // Зеркалим отрицательные (эрмитова симметрия)
    for (size_t i = half.size(); i < N; ++i) {
      full[i] = std::conj(half[N - i]);
    }
    return full;
  }

  // ──────────────────────────────────────────────────────────────────────
  // Magnitude: |X[k]| из спектра
  // ──────────────────────────────────────────────────────────────────────

  /**
   * @brief Вычислить магнитуды из комплексного спектра
   * @param spectrum Вектор complex<float>
   * @param squared  true → |X|², false → |X|
   * @return Вектор float с магнитудами
   */
  static std::vector<float> Magnitude(
      const std::vector<std::complex<float>>& spectrum,
      bool squared = false)
  {
    std::vector<float> mag(spectrum.size());
    for (size_t i = 0; i < spectrum.size(); ++i) {
      float m = std::norm(spectrum[i]);  // re² + im²
      mag[i] = squared ? m : std::sqrt(m);
    }
    return mag;
  }
};

}  // namespace spectrum_utils
