# LchFarrow — Tests

Тесты и бенчмарки модуля `lch_farrow`.

Точка входа: [`all_test.hpp`](all_test.hpp) — включается из `src/main.cpp`.

---

## Функциональные тесты

| Файл | Что тестирует |
|------|--------------|
| [`test_lch_farrow.hpp`](test_lch_farrow.hpp) | LchFarrow (OpenCL): нулевая задержка, целая задержка (5 сэмп.), дробная задержка (2.7 сэмп.) — GPU vs CPU |
| [`test_lch_farrow_rocm.hpp`](test_lch_farrow_rocm.hpp) | LchFarrowROCm (HIP): те же три сценария на ROCm/AMD GPU |

---

## Бенчмарки OpenCL (GpuBenchmarkBase)

### lch_farrow_benchmark.hpp

Benchmark-класс для OpenCL-реализации:

| Класс | Процессор | Стейджи | output_dir |
|-------|-----------|---------|-----------|
| `LchFarrowBenchmark` | `LchFarrow::Process()` | `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow/` |

**Стейджи** (профилируемые события):
- `Upload_delay` — `clEnqueueWriteBuffer` (задержки → GPU)
- `Kernel` — `lch_farrow_delay` (OpenCL kernel)

> ℹ️ `input_buf` загружается **один раз** до бенчмарка — не является частью замера.

### test_lch_farrow_benchmark.hpp

Namespace: `test_lch_farrow_benchmark`

Параметры:
- `antennas = 8`, `points = 4096`, `sample_rate = 1 MHz`
- `delays = {0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9}` мкс
- `n_warmup = 5`, `n_runs = 20`

---

## Бенчмарки ROCm (ENABLE_ROCM=1)

### lch_farrow_benchmark_rocm.hpp

Benchmark-класс для ROCm-реализации:

| Класс | Процессор | Стейджи | output_dir |
|-------|-----------|---------|-----------|
| `LchFarrowBenchmarkROCm` | `LchFarrowROCm::ProcessFromCPU()` | `Upload_input`, `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow_ROCm/` |

**Стейджи** (профилируемые события):
- `Upload_input` — `hipMemcpyHtoDAsync` (входной сигнал → GPU)
- `Upload_delay` — `hipMemcpyHtoDAsync` (задержки → GPU)
- `Kernel` — `lch_farrow_delay` (HIP kernel)

### test_lch_farrow_benchmark_rocm.hpp

Namespace: `test_lch_farrow_benchmark_rocm`

Параметры:
- `antennas = 8`, `points = 4096`, `sample_rate = 1 MHz`
- `delays = {0.3, 1.7, 2.1, 3.5, 4.0, 5.3, 6.7, 7.9}` мкс
- `n_warmup = 5`, `n_runs = 20`

Если нет AMD GPU → `[SKIP]`.

---

## Как запустить

Раскомментировать нужные вызовы в `all_test.hpp`:

```cpp
// OpenCL LchFarrow Benchmark
test_lch_farrow_benchmark::run();

// ROCm LchFarrowROCm Benchmark (только Linux + AMD GPU)
#if ENABLE_ROCM
test_lch_farrow_benchmark_rocm::run();
#endif
```

⚠️ **Требования**:
- В `configGPU.json` установить `"is_prof": true`
- OpenCL queue создаётся с `CL_QUEUE_PROFILING_ENABLE` (test runner делает автоматически)
- ROCm бенчмарк — только Linux + AMD GPU с установленным ROCm

---

## Результаты

Результаты сохраняются в `Results/Profiler/GPU_00_LchFarrow*/`:
- `report.md` — Markdown-отчёт (min/max/avg по событиям)
- `report.json` — JSON для автоматической обработки

Вывод управляется через `GPUProfiler`:
- `bench.Report()` → `PrintReport()` + `ExportMarkdown()` + `ExportJSON()`
