# Filters — Краткий справочник

> FIR/IIR + Скользящие средние (SMA/EMA/MMA/DEMA/TEMA) + Kalman 1D + KAMA: OpenCL + ROCm (AMD, ENABLE_ROCM=1)

**Namespace**: `filters` | **Каталог**: `modules/filters/`

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Фильтрация — это подавление нежелательных частот и шума в сигнале. Модуль позволяет применять фильтры к многоканальным данным (N антенн × M точек) на GPU — всё параллельно, без цикла по каналам.

---

### FIR фильтр — "прямой" классический фильтр

**Что делает**: для каждого выходного отсчёта берёт N последних входных отсчётов и взвешивает их через заданные коэффициенты (тапы). Простой, стабильный, всегда понятно что происходит.

**Когда брать**: полосовая фильтрация, подавление помех, выделение сигнала на фоне шума. Коэффициенты рассчитывают через scipy (firwin, firls). Хорош при ≥ 8 каналах — тогда GPU реально быстрее CPU.

---

### IIR фильтр — рекурсивный фильтр (биквад каскад)

**Что делает**: фильтр с памятью — каждый выход зависит не только от входов, но и от предыдущих выходов. Реализуется как каскад биквад-секций (Direct Form II Transposed).

**Когда брать**: когда нужна крутая АЧХ при малом числе коэффициентов. Например, Баттерворт 2-4 порядка. Коэффициенты из scipy: `butter()`, `cheby1()`. Внимание: при одном канале IIR медленнее CPU — выгоден только при ≥ 8 каналах.

---

### Скользящие средние (MovingAverageFilterROCm) — сглаживание

**Что делает**: сглаживает сигнал по окну N отсчётов. Пять вариантов:
- **SMA** — простое среднее по кольцевому буферу. Самый медленный отклик, но честный.
- **EMA** — экспоненциальное взвешивание: последние отсчёты важнее. Быстрее реагирует на изменения.
- **MMA** — как EMA, но вес фиксирован как 1/N.
- **DEMA / TEMA** — двойное/тройное EMA: убирают запаздывание EMA, быстрее следуют за трендом.

**Когда брать**: сглаживание огибающей, подавление высокочастотного шума без фазовых искажений. Эффективно при ≥ 64 каналах.

---

### Kalman фильтр (KalmanFilterROCm) — оптимальное сглаживание

**Что делает**: оценивает «истинное» значение сигнала, зная уровень шума измерений (R) и динамику процесса (Q). Адаптивно подбирает коэффициент сглаживания — при малом шуме доверяет измерениям, при большом — своей модели.

**Когда брать**: трекинг амплитуды или фазы beat-сигнала в LFM радаре. Когда нужно не просто сгладить, а получить оптимальную оценку в шумной среде. Re и Im обрабатываются независимо.

**Настройка**: Q/R = 0.01 → сильное сглаживание; Q/R = 1 → быстрая реакция на изменения.

---

### KAMA — адаптивная скользящая средняя Кауфмана (KaufmanFilterROCm)

**Что делает**: скользящая средняя, которая сама подстраивает скорость: при чистом тренде (Efficiency Ratio ≈ 1) следует за сигналом быстро, при шуме (ER ≈ 0) почти замирает.

**Когда брать**: когда сигнал чередует периоды тренда и хаоса, и фиксированный N для скользящей средней не подходит. Эффективно при ≥ 64 каналах.

---

## Алгоритмы

```
FIR:  y[ch][n] = Σ h[k] · x[ch][n-k],  k=0..N-1   (direct-form)
IIR:  y[n] = b0·x[n] + w1              (DFII-T, biquad cascade)
      w1   = b1·x[n] - a1·y[n] + w2
      w2   = b2·x[n] - a2·y[n]
```

---

## Классы

| Класс | Backend | Назначение |
|-------|---------|------------|
| `FirFilter` | OpenCL | FIR direct-form, cl_mem |
| `IirFilter` | OpenCL | IIR biquad cascade DFII-T, cl_mem |
| `FirFilterROCm` | ROCm/HIP | FIR, hiprtc, void* ptr |
| `IirFilterROCm` | ROCm/HIP | IIR biquad, hiprtc, void* ptr |
| `MovingAverageFilterROCm` | ROCm/HIP | SMA/EMA/MMA/DEMA/TEMA, void* ptr |
| `KalmanFilterROCm` | ROCm/HIP | 1D scalar Kalman (Re/Im), void* ptr |
| `KaufmanFilterROCm` | ROCm/HIP | KAMA адаптивная MA по ER, void* ptr |

---

## Быстрый старт

### C++ — FirFilter (OpenCL)

```cpp
#include "filters/fir_filter.hpp"

filters::FirFilter fir(backend);
fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});  // или LoadConfig(json)

auto result = fir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);  // caller owns!
```

### C++ — IirFilter (OpenCL)

```cpp
#include "filters/iir_filter.hpp"

filters::BiquadSection sec;
sec.b0 = 0.02008337f;  sec.b1 = 0.04016673f;  sec.b2 = 0.02008337f;
sec.a1 = -1.56101808f; sec.a2 = 0.64135154f;  // Butterworth 2nd order, fc=0.1

filters::IirFilter iir(backend);
iir.SetBiquadSections({sec});

auto result = iir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);
```

### C++ — FirFilterROCm (Linux + AMD GPU)

```cpp
#include "filters/fir_filter_rocm.hpp"

filters::FirFilterROCm fir(rocm_backend);
fir.SetCoefficients(coeffs);

// Из CPU (upload + process)
auto res = fir.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!

// Из GPU ptr
auto res2 = fir.Process(gpu_ptr, channels, points);
hipFree(res2.data);
```

### Python — FirFilter (OpenCL)

```python
import gpuworklib as gw
import scipy.signal as sig

ctx = gw.GPUContext(0)
fir = gw.FirFilter(ctx)

taps = sig.firwin(64, 0.1)
fir.set_coefficients(taps.tolist())

result = fir.process(signal)  # (channels, points) или 1D complex64
```

### Python — IirFilter (OpenCL)

```python
iir = gw.IirFilter(ctx)
sos = sig.butter(2, 0.1, output='sos')
sections = [
    {'b0': float(r[0]), 'b1': float(r[1]), 'b2': float(r[2]),
     'a1': float(r[4]), 'a2': float(r[5])}
    for r in sos
]
iir.set_sections(sections)
result = iir.process(signal)  # (channels, points) complex64
```

### Python — FirFilterROCm (Linux + AMD GPU)

```python
ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilterROCm(ctx)
fir.set_coefficients(sig.firwin(64, 0.1).tolist())
result = fir.process(data)  # np.ndarray complex64
```

### C++ — MovingAverageFilterROCm (Linux + AMD GPU)

```cpp
#include "filters/moving_average_filter_rocm.hpp"

filters::MovingAverageFilterROCm ma(rocm_backend);
ma.SetParams(filters::MAType::EMA, 10);  // или SMA, MMA, DEMA, TEMA

auto res = ma.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**MAType**: `SMA` (ring buffer, max N=128), `EMA` (α=2/(N+1)), `MMA` (α=1/N), `DEMA` (2×EMA1-EMA2), `TEMA` (3×EMA1-3×EMA2+EMA3).

### C++ — KalmanFilterROCm (Linux + AMD GPU)

```cpp
#include "filters/kalman_filter_rocm.hpp"

filters::KalmanFilterROCm kalman(rocm_backend);
// SetParams(Q, R, x0, P0)
// Q/R << 1: сильное сглаживание | Q/R >> 1: быстрая реакция
kalman.SetParams(0.1f, 25.0f, 0.0f, 25.0f);

auto res = kalman.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**Применение**: сглаживание огибающей/фазы beat-сигнала, трекинг амплитуды.

### C++ — KaufmanFilterROCm (Linux + AMD GPU)

```cpp
#include "filters/kaufman_filter_rocm.hpp"

filters::KaufmanFilterROCm kauf(rocm_backend);
// SetParams(er_period, fast_period, slow_period) — max er_period=128
kauf.SetParams(10, 2, 30);  // стандартные параметры Kaufman

auto res = kauf.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**Поведение**: тренд (ER≈1) → быстрое следование; шум (ER≈0) → KAMA почти заморожен.

---

## Ключевые нюансы

| Параметр | FIR (OpenCL) | IIR (OpenCL) |
|----------|-------------|-------------|
| NDRange | 2D `(ch, ⌈pts/256⌉×256)` | 1D `(ch,)` |
| Коэффициенты | `__constant` ≤ 16 000 тапов, иначе `__global` | `__constant` SOS-матрица |
| Параллелизм | По каналам И семплам | Только по каналам |
| Рекомендовано | ≥ 8 каналов | ≥ 8 каналов |

---

## On-disk kernel cache

FirFilter и IirFilter используют DrvGPU KernelCacheService:
- **Первый запуск:** компиляция → Save в `modules/filters/kernels/bin/`
- **Повторный:** Load binary (~1 мс вместо ~50 мс компиляции)
- **Fallback:** при отсутствии cache — компиляция из source

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| ⚠️ | `clReleaseMemObject(result.data)` — caller owns (OpenCL) |
| ⚠️ | `hipFree(result.data)` — caller owns (ROCm) |
| ⚠️ | IIR single-channel = медленнее CPU! GPU выгоден только при ≥ 8 каналах |
| ⚠️ | SOS scipy: `a1=row[4], a2=row[5]` (пропускаем `a0=row[3]`, он всегда 1) |
| ⚠️ | `kMaxConstantTaps = 16000`: при больше → автоматически `__global` (медленнее) |
| ⚠️ | Бенчмарк: OpenCL queue **обязательно** с `CL_QUEUE_PROFILING_ENABLE` |
| ⚠️ | ROCm FirFilter на Windows — compile-only stub (throws runtime_error) |
| ⚠️ | SMA max N=128, KAMA er_period max=128 (ring buffer в регистрах) |
| ⚠️ | Kalman Re/Im независимо — корректно для AWGN, не для коррелированного шума |
| ⚠️ | Kalman tuning: Q ≈ noise_sigma² / 100, R = noise_sigma². Q/R=0.01 → сглаживание |
| ⚠️ | MA/Kalman/KAMA эффективны при ≥ 64 каналах (1 thread per channel) |

---

## Тесты

| Файл | Что тестирует |
|------|---------------|
| `tests/test_fir_basic.hpp` | OpenCL FIR 64-tap, GPU vs CPU, 8ch × 4096pts |
| `tests/test_iir_basic.hpp` | OpenCL IIR Butterworth 2nd, GPU vs CPU, 8ch × 4096pts |
| `tests/test_filters_rocm.hpp` | ROCm: 6 тестов FIR/IIR (Linux + AMD GPU) |
| `tests/test_moving_average_rocm.hpp` | ROCm: 6 тестов MA (SMA/EMA/MMA/DEMA/TEMA + step demo) |
| `tests/test_kalman_rocm.hpp` | ROCm: 5 тестов Kalman (GPU vs CPU, convergence, LFM radar demo) |
| `tests/test_kaufman_rocm.hpp` | ROCm: 5 тестов KAMA (trend/noise/adaptive) |
| `tests/filters_benchmark.hpp` | OpenCL: FirFilterBenchmark, IirFilterBenchmark |
| `Python_test/filters/test_filters_stage1.py` | Python: 5 тестов FIR+IIR vs scipy |
| `Python_test/filters/test_fir_filter_rocm.py` | Python ROCm FIR: 5 тестов |
| `Python_test/filters/test_iir_filter_rocm.py` | Python ROCm IIR: multi-section, GPU ptr |
| `Python_test/filters/test_moving_average_rocm.py` | Python ROCm MA: все 5 типов (bindings не зарег.) |
| `Python_test/filters/test_kalman_rocm.py` | Python ROCm Kalman: GPU vs CPU, convergence (bindings не зарег.) |
| `Python_test/filters/test_kaufman_rocm.py` | Python ROCm KAMA: adaptive behavior (bindings не зарег.) |

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| ⚠ | `clReleaseMemObject(result.data)` — caller owns (OpenCL) |
| ⚠ | `hipFree(result.data)` — caller owns (ROCm) |
| ⚠ | IIR single-channel = медленнее CPU! GPU выгоден только при ≥ 8 каналах |
| ⚠ | SOS scipy: `a1=row[4], a2=row[5]` (пропускаем `a0=row[3]`, он всегда 1) |
| ⚠ | SMA max N=128, KAMA er_period max=128 (ring buffer в регистрах) |
| ⚠ | MA/Kalman/KAMA эффективны при ≥ 64 каналах |
| ⚠ | **Python bindings для MA/Kalman/KAMA НЕ зарегистрированы** в `gpu_worklib_bindings.cpp` |

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты с rationale
- [API.md](API.md) — полный API-справочник: все классы, сигнатуры, цепочки вызовов
- [gpu_filters_research.md](gpu_filters_research.md) — Overlap-Save/Add, tiled FIR — будущие алгоритмы

---

*Обновлено: 2026-03-09*
