#pragma once

// ============================================================================
// CpuFFT — CPU-эталон FFT через pocketfft (header-only, для тестов)
//
// ЧТО:    Статический класс-обёртка над pocketfft с 4 операциями: ForwardC2C,
//         InverseC2C, ForwardR2C (half-spectrum N/2+1), ForwardR2C_Full
//         (с эрмитовой симметрией — совместимо с hipFFT C2C output) +
//         Magnitude (|X[k]| или |X[k]|² по флагу squared).
//
// ЗАЧЕМ:  GPU-реализации FFT (FFTProcessorROCm) нуждаются в reference
//         для unit-тестов (GPU vs CPU sanity checks) и в fallback'е для
//         отладки на машинах без GPU. Без CPU-эталона тесты могли бы
//         только проверять, что GPU не падает — не корректность.
//
// ПОЧЕМУ: - pocketfft (BSD, Max-Planck-Society) — header-only, без зависимостей,
//           проверенная численная корректность (используется в SciPy).
//         - POCKETFFT_NO_MULTITHREADING — выключаем threads: используется в
//           тестах, один поток достаточен и убирает зависимость от <thread>.
//         - Все методы static → не нужен instance, можно вызывать как
//           CpuFFT::ForwardC2C(input). Класс — только namespace-прикрытие.
//         - InverseC2C нормализует на 1/N (как hipFFT с inverse_normalize=true);
//           ForwardC2C — без нормализации (стандарт FFTW).
//         - ForwardR2C_Full — отдельно от ForwardR2C: hipFFT в C2C-режиме
//           выдаёт полный спектр N точек, и для прямого сравнения нужно
//           CPU достроить отрицательные частоты через conj.
//
// Использование:
//   #include <spectrum/utils/cpu_fft.hpp>
//   using spectrum_utils::CpuFFT;
//
//   auto spectrum = CpuFFT::ForwardC2C(signal);          // complex → complex
//   auto magnitude = CpuFFT::Magnitude(spectrum, false); // |X[k]|
//   auto recovered = CpuFFT::InverseC2C(spectrum);       // обратный FFT
//
// История:
//   - Создан: 2026-04-16 (унификация CPU-reference для всех тестов spectrum)
// ============================================================================

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
