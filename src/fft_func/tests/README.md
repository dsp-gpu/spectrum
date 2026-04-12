# fft_func — Тесты

> **Модуль**: `modules/fft_func/`
> **Обновлено**: 2026-03-11
> **Объединяет**: бывшие модули `fft_processor` и `fft_maxima`

## Обзор

Тесты для FFT-обработки (`FFTProcessorROCm`, `FFTProcessor`) и поиска максимумов спектра (`SpectrumMaximaFinder`, `SpectrumProcessorROCm`). Все тесты — header-only (`*.hpp`), вызываются из `src/main.cpp` → `all_test.hpp`.

---

## Тесты по платформе

### ROCm / hipFFT (активные, ENABLE_ROCM=1)

| Файл | Namespace | Описание |
|------|-----------|----------|
| `test_fft_processor_rocm.hpp` | `test_fft_processor_rocm` | Базовые тесты FFTProcessorROCm (hipFFT) |
| `test_complex_to_mag_phase_rocm.hpp` | `test_complex_to_mag_phase_rocm` | Тест ComplexToMagPhaseROCm |
| `test_fft_matrix_rocm.hpp` | `test_fft_matrix_rocm` | Матричный бенчмарк: таблица beams × nFFT |
| `test_spectrum_maxima_rocm.hpp` | `test_spectrum_maxima_rocm` | SpectrumProcessorROCm: ONE_PEAK / TWO_PEAKS |
| `test_find_all_maxima.hpp` | `test_find_all_maxima` | FindAllMaxima / AllMaxima pipeline |
| `test_gpu_generator_integration.hpp` | `test_gpu_generator_integration` | CwGenerator → SpectrumMaximaFinder (GPU→GPU) |

### OpenCL / clFFT (только ENABLE_CLFFT=1, ветка opencl-clfft)

| Файл | Namespace | Описание |
|------|-----------|----------|
| `test_fft_processor.hpp` | `test_fft_processor` | FFTProcessor (OpenCL/clFFT) |
| `test_fft_vs_cpu.hpp` | `test_fft_vs_cpu` | Сравнение GPU FFT с CPU-референсом |
| `test_spectrum_maxima.hpp` | `test_spectrum_maxima` | SpectrumMaximaFinder OpenCL |

### Benchmark (закомментировано, запускать отдельно)

| Файл | Описание |
|------|----------|
| `test_fft_benchmark.hpp` | Бенчмарк FFTProcessor (OpenCL) |
| `test_fft_benchmark_rocm.hpp` | Бенчмарк FFTProcessorROCm |
| `test_fft_maxima_benchmark.hpp` | Бенчмарк SpectrumMaximaFinder (OpenCL) |
| `test_fft_maxima_benchmark_rocm.hpp` | Бенчмарк SpectrumProcessorROCm |

## Запуск

```bash
# Только fft_func тесты
./GPUWorkLib fft_func

# Все тесты
./GPUWorkLib all
```
