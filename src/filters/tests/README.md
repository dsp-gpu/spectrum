# Filters Module Tests

## test_fir_basic.hpp
- **Low-pass FIR**: 64 taps, fc=0.1 (normalized), Hamming window
- **Signal**: 8 channels, 4096 points, CW 100Hz + CW 5000Hz, fs=50kHz
- **Validation**: GPU `Process()` vs `ProcessCpu()`, max error < 1e-3
- **Coefficients**: Pre-computed from `scipy.signal.firwin(64, 0.1)`

## test_iir_basic.hpp
- **Butterworth 2nd order LP**: fc=0.1 (normalized), 1 biquad section
- **Signal**: Same as FIR test
- **Validation**: GPU `Process()` vs `ProcessCpu()`, max error < 1e-3
- **Coefficients**: Pre-computed from `scipy.signal.butter(2, 0.1, output='sos')`

## test_filters_rocm.hpp
- ROCm equivalents: `FirFilterROCm` + `IirFilterROCm`
- Запускается только на Linux + AMD GPU (`ENABLE_ROCM=1`)

---

## filters_benchmark.hpp
**Namespace**: `test_filters`
**Дата**: 2026-03-01

Benchmark-классы наследники `GpuBenchmarkBase` (OpenCL):

| Класс | Метод | Стадия | Результаты |
|-------|-------|--------|------------|
| `FirFilterBenchmark` | `Process(cl_mem)` | `Kernel` | `Results/Profiler/GPU_00_FirFilter/` |
| `IirFilterBenchmark` | `Process(cl_mem)` | `Kernel` | `Results/Profiler/GPU_00_IirFilter/` |

- Входной `cl_mem` буфер фиксирован между прогонами
- Выходной `cl_mem` освобождается после каждого вызова
- 5 warmup + 20 замерных → PrintReport + ExportJSON + ExportMarkdown

## test_filters_benchmark.hpp
**Namespace**: `test_filters_benchmark`
**Дата**: 2026-03-01

Test runner для OpenCL бенчмарков:
- **8 каналов × 4096 точек**, fs=50 кГц, CW сигналы
- OpenCL init с `CL_QUEUE_PROFILING_ENABLE` (обязательно для cl_event timing!)
- FIR коэффициенты: `kTestFirCoeffs64` из `test_fir_basic.hpp`
- IIR: Butterworth 2nd order LP (те же что в `test_iir_basic.hpp`)
- Если `is_prof=false` в configGPU.json — выводит `[SKIP]`

## filters_benchmark_rocm.hpp
**Namespace**: `test_filters_rocm`
**Дата**: 2026-03-01
**Условие**: `#if ENABLE_ROCM`

ROCm benchmark-классы наследники `GpuBenchmarkBase`:

| Класс | Метод | Стадии | Результаты |
|-------|-------|--------|------------|
| `FirFilterROCmBenchmark` | `ProcessFromCPU` | Upload(H2D) + Kernel | `Results/Profiler/GPU_00_FirFilter_ROCm/` |
| `IirFilterROCmBenchmark` | `ProcessFromCPU` | Upload(H2D) + Kernel | `Results/Profiler/GPU_00_IirFilter_ROCm/` |

## test_filters_benchmark_rocm.hpp
**Namespace**: `test_filters_benchmark_rocm`
**Дата**: 2026-03-01
**Условие**: `#if ENABLE_ROCM`

ROCm test runner:
- Если нет AMD GPU — `[SKIP]`, не падает
- 5 warmup + 20 замерных → PrintReport + ExportJSON + ExportMarkdown

---

## Running Tests

From `src/main.cpp`:
```cpp
#include "modules/filters/tests/all_test.hpp"
// ...
filters_all_test::run();
```

Включить/отключить тесты в `all_test.hpp`:
```cpp
filters::tests::run_fir_basic();          // FIR basic test
filters::tests::run_iir_basic();          // IIR basic test
test_filters_rocm::run();                 // ROCm tests (Linux+AMD)
// test_filters_benchmark::run();         // OpenCL benchmark
// test_filters_benchmark_rocm::run();    // ROCm benchmark
```
