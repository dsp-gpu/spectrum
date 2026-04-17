# Spectrum — Quick (объединённый: FFT + Filters + LCH Farrow)

> Репо `spectrum` объединяет три компонента из DSP-GPU: **fft_func** (БПФ-пайплайн), **filters** (FIR/IIR/адаптивные) и **lch_farrow** (LCH + Farrow дробная задержка).

---

## Компонент: FFT Pipeline


> GPU FFT + поиск максимумов спектра: ROCm/hipFFT (ветка main, AMD RDNA4+)

**Namespace**: `fft_processor` + `antenna_fft` | **Каталог**: `spectrum/`

---

#### Концепция — зачем и что это такое

**Зачем нужен модуль?**
В пайплайне ЦОС часто нужно: взять IQ-сигнал, посчитать его спектр (FFT) и найти в нём частоты сигналов. `fft_func` делает это на GPU — принимает данные с CPU или GPU, возвращает спектр или позиции пиков. Это центральный FFT-модуль всего проекта: его используют гетеродин, статистика, стратегии.

**Модуль появился слиянием** двух бывших модулей (`fft_processor` + `fft_maxima`). Старая документация лежит в `Doc/Modules/~!/` только как архив.

---

#### FFTProcessorROCm — пакетный FFT

**Что делает**: принимает плоский массив `[beam_count × n_point]` комплексных отсчётов, выполняет FFT на GPU для всех лучей параллельно, возвращает спектр.

**Когда брать**: нужен полный спектр (для визуализации, дальнейшего анализа, передачи в другой модуль).

**Три режима вывода**:
- `COMPLEX` — комплексный спектр `X[k]` (для поэлементных операций)
- `MAGNITUDE_PHASE` — `|X[k]|` + `arg(X[k])` в **радианах**
- `MAGNITUDE_PHASE_FREQ` — то же + `f[k]` в Гц

**Ключевой факт**: FFT не нормируется — результат идентичен `np.fft.fft()`. Для физической амплитуды делить на `n_point`.

**Zero-padding**: `nFFT = nextPow2(n_point) × repeat_count`. По умолчанию `repeat_count=1` (без дополнительного padding). Увеличивает разрешение по частоте, не добавляет информацию.

**Оптимизации**: Ref03 архитектура (GpuContext + BufferSet<4>). LRU-2 plan cache — два разных размера батча не пересоздают план. HSACO дисковый кеш — первый запуск ~200 мс, дальше ~1 мс.

---

#### ComplexToMagPhaseROCm — IQ → амплитуда+фаза (без FFT)

**Что делает**: прямое поэлементное вычисление `|z|` и `atan2(im, re)` — никакого FFT. Чисто математическое преобразование сигнала во временной области.

**Когда брать**: нужна огибающая сигнала (амплитуда и фаза отсчёт за отсчётом), а не спектр. Также: pipeline статистики (ProcessMagnitudeToBuffer → compute_statistics).

**Методы**:
- `Process()` — CPU/GPU → CPU (`MagPhaseResult` или `MagnitudeResult`)
- `ProcessToGPU()` — CPU/GPU → GPU (interleaved `{mag, phase}` float2, **CALLER OWNS!**)
- `ProcessMagnitude()` — только амплитуда (без фазы), с нормализацией
- `ProcessMagnitudeToBuffer()` — zero-alloc: пишет в чужой буфер (для pipeline)

**norm_coeff**: `0` или `1.0` = без нормировки; `-1.0` = делить на `n_point`; `>0` = умножить.

---

#### SpectrumMaximaFinder — поиск пиков

**Что делает**: принимает сырой сигнал или готовый FFT-спектр, находит позиции максимумов в нём с параболической интерполяцией для точности до долей бина.

**Когда брать**: нужна не полная спектрограмма, а конкретные частоты — например, найти f_beat в дечирпе, определить несущую принятого сигнала.

**Три режима**:

| Метод | Вход | Когда использовать |
|-------|------|--------------------|
| `Process(ONE_PEAK/TWO_PEAKS)` | Сырой сигнал | Известно, что нужен 1 или 2 пика — минимальный overhead |
| `FindAllMaxima()` | Сырой сигнал | Неизвестно сколько пиков, нужны все локальные максимумы |
| `AllMaxima()` | FFT-спектр (готовый) | FFT уже посчитан, нужны только пики — без повторного FFT |

**Параболическая интерполяция** — только для `peak[0]` (главный пик). Остальные пики: `bin × fs/nFFT`.

**批чинговая обработка**: BatchManager автоматически разбивает большие данные по `memory_limit`.

---

#### Аналогии и связи

```
signal_generators  →  fft_func(FFTProcessorROCm)  →  fft_func(SpectrumMaximaFinder)
(генерирует IQ)        (считает спектр)               (находит пики)
                          ↕
                   heterodyne (SpectrumMaximaFinder встроен)
                          ↕
                   statistics (ComplexToMagPhaseROCm.ProcessMagnitudeToBuffer)
                          ↕
                   strategies (FFT + MaximaFinder pipeline)
```

---

#### Критические отличия двух типов фазы

| Класс / структура | Поле | Единицы |
|-------------------|------|---------|
| `FFTMagPhaseResult.phase` | phase[k] | **РАДИАНЫ** `[-π, π]` |
| `ComplexToMagPhaseROCm` (MagPhaseResult.phase) | phase[k] | **РАДИАНЫ** `[-π, π]` |
| `MaxValue.phase` | phase | **ГРАДУСЫ** `[-180°, 180°]` |

Не перепутай при сравнении результатов!

---

#### Ограничения

- **Только ROCm** в ветке `main` (AMD GPU, RDNA4+ gfx1201+). clFFT мёртв на RDNA4+ — все OpenCL тесты закомментированы.
- **AllMaxima: `n_point = nFFT`** — передавать размер FFT, не n_point сигнала.
- **Python: find_all_maxima принимает FFT**, а не сырой сигнал. Сначала `fft.process_complex()`.
- **ProcessToGPU — caller owns**: ОБЯЗАТЕЛЬНО освободить `backend->Free(ptr)` или `hipFree(ptr)`.
- **OutputDestination::GPU — caller owns**: `hipFree(result.gpu_maxima)` и `hipFree(result.gpu_counts)`.

---

#### Быстрый старт

##### C++ — FFTProcessorROCm

```cpp
#include <spectrum/fft_processor_rocm.hpp>

fft_processor::FFTProcessorROCm fft(backend);

fft_processor::FFTProcessorParams params;
params.beam_count  = 8;
params.n_point     = 1024;
params.sample_rate = 1e6f;
params.output_mode = fft_processor::FFTOutputMode::MAGNITUDE_PHASE;

std::vector<std::complex<float>> data(8 * 1024);
auto results = fft.ProcessMagPhase(data, params);
// results[b].magnitude[k], results[b].phase[k] — фаза в РАДИАНАХ
```

##### C++ — SpectrumMaximaFinder (ONE_PEAK)

```cpp
#include <spectrum/interface/spectrum_maxima_types.h>
#include <spectrum/interface/spectrum_input_data.hpp>

antenna_fft::SpectrumMaximaFinder finder(backend);

antenna_fft::InputData<std::vector<std::complex<float>>> input{
    .antenna_count = 5,
    .n_point = 1000,
    .data = my_signal,
    .sample_rate = 10000.0f,
    .repeat_count = 2
};

auto results = finder.Process(input,
    antenna_fft::PeakSearchMode::ONE_PEAK,
    antenna_fft::DriverType::ROCm);

for (const auto& r : results)
    printf("Beam %u: freq=%.1f Hz, phase=%.1f deg\n",
           r.antenna_id,
           r.interpolated.refined_frequency,
           r.interpolated.phase);  // ГРАДУСЫ в MaxValue!
```

##### C++ — ComplexToMagPhaseROCm

```cpp
#include <spectrum/complex_to_mag_phase_rocm.hpp>

fft_processor::ComplexToMagPhaseROCm converter(backend);
fft_processor::MagPhaseParams params;
params.beam_count = 4;
params.n_point    = 2048;

auto results = converter.Process(data, params);
// results[b].phase[k] — фаза в РАДИАНАХ
```

##### Python — FFTProcessorROCm

```python
import dsp_spectrum
import numpy as np

ctx = dsp_spectrum.ROCmGPUContext(0)
fft = dsp_spectrum.FFTProcessorROCm(ctx)

### 8 лучей, 1024 точки
signal = np.random.randn(8 * 1024).view(np.complex64)

### Комплексный спектр (flat → flat)
spectrum = fft.process_complex(signal, 1e6, beam_count=8, n_point=1024)

### Или 2D → 2D
spectrum = fft.process_complex(signal.reshape(8, 1024), 1e6)

### Mag+Phase
result = fft.process_mag_phase(signal, 1e6, beam_count=8, include_freq=True)
### result['magnitude'], result['phase'] (РАДИАНЫ!), result['frequency']
```

##### Python — SpectrumMaximaFinder

```python
ctx = dsp_spectrum.ROCmGPUContext(0)
fft = dsp_spectrum.FFTProcessorROCm(ctx)
finder = dsp_spectrum.SpectrumMaximaFinderROCm(ctx)

signal = np.zeros(1024, dtype=np.complex64)
### ... заполнить ...

### ШАГ 1: FFT (обязательно сначала!)
spectrum = fft.process_complex(signal, 1000.0)

### ШАГ 2: найти все пики
result = finder.find_all_maxima(spectrum, sample_rate=1000.0)
print(result['frequencies'])  # Hz
```

---

#### Таблица: какой класс выбрать

| Задача | Класс |
|--------|-------|
| Полный спектр для N лучей | `FFTProcessorROCm` |
| Найти 1-2 пика по частоте | `SpectrumMaximaFinder.Process(ONE_PEAK/TWO_PEAKS)` |
| Найти все пики (неизвестно сколько) | `SpectrumMaximaFinder.FindAllMaxima()` |
| FFT уже есть, нужны только пики | `SpectrumMaximaFinder.AllMaxima()` |
| Огибающая/фаза во временной области | `ComplexToMagPhaseROCm.Process()` |
| IQ → амплитуда, остаться на GPU | `ComplexToMagPhaseROCm.ProcessMagnitudeToGPU()` |
| Pipeline с нулевыми аллокациями | `ComplexToMagPhaseROCm.ProcessMagnitudeToBuffer()` |

---

#### Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты
- [API.md](API.md) — полный API-справочник (сигнатуры, параметры)
- [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) — оптимизация HIP/ROCm ядер

---

*Обновлено: 2026-03-28*

---

## Компонент: Filters


> FIR/IIR + Скользящие средние (SMA/EMA/MMA/DEMA/TEMA) + Kalman 1D + KAMA: OpenCL + ROCm (AMD, ENABLE_ROCM=1)

**Namespace**: `filters` | **Каталог**: `spectrum/`

---

#### Концепция — зачем и что это такое

**Зачем нужен модуль?**
Фильтрация — это подавление нежелательных частот и шума в сигнале. Модуль позволяет применять фильтры к многоканальным данным (N антенн × M точек) на GPU — всё параллельно, без цикла по каналам.

---

##### FIR фильтр — "прямой" классический фильтр

**Что делает**: для каждого выходного отсчёта берёт N последних входных отсчётов и взвешивает их через заданные коэффициенты (тапы). Простой, стабильный, всегда понятно что происходит.

**Когда брать**: полосовая фильтрация, подавление помех, выделение сигнала на фоне шума. Коэффициенты рассчитывают через scipy (firwin, firls). Хорош при ≥ 8 каналах — тогда GPU реально быстрее CPU.

---

##### IIR фильтр — рекурсивный фильтр (биквад каскад)

**Что делает**: фильтр с памятью — каждый выход зависит не только от входов, но и от предыдущих выходов. Реализуется как каскад биквад-секций (Direct Form II Transposed).

**Когда брать**: когда нужна крутая АЧХ при малом числе коэффициентов. Например, Баттерворт 2-4 порядка. Коэффициенты из scipy: `butter()`, `cheby1()`. Внимание: при одном канале IIR медленнее CPU — выгоден только при ≥ 8 каналах.

---

##### Скользящие средние (MovingAverageFilterROCm) — сглаживание

**Что делает**: сглаживает сигнал по окну N отсчётов. Пять вариантов:
- **SMA** — простое среднее по кольцевому буферу. Самый медленный отклик, но честный.
- **EMA** — экспоненциальное взвешивание: последние отсчёты важнее. Быстрее реагирует на изменения.
- **MMA** — как EMA, но вес фиксирован как 1/N.
- **DEMA / TEMA** — двойное/тройное EMA: убирают запаздывание EMA, быстрее следуют за трендом.

**Когда брать**: сглаживание огибающей, подавление высокочастотного шума без фазовых искажений. Эффективно при ≥ 64 каналах.

---

##### Kalman фильтр (KalmanFilterROCm) — оптимальное сглаживание

**Что делает**: оценивает «истинное» значение сигнала, зная уровень шума измерений (R) и динамику процесса (Q). Адаптивно подбирает коэффициент сглаживания — при малом шуме доверяет измерениям, при большом — своей модели.

**Когда брать**: трекинг амплитуды или фазы beat-сигнала в LFM радаре. Когда нужно не просто сгладить, а получить оптимальную оценку в шумной среде. Re и Im обрабатываются независимо.

**Настройка**: Q/R = 0.01 → сильное сглаживание; Q/R = 1 → быстрая реакция на изменения.

---

##### KAMA — адаптивная скользящая средняя Кауфмана (KaufmanFilterROCm)

**Что делает**: скользящая средняя, которая сама подстраивает скорость: при чистом тренде (Efficiency Ratio ≈ 1) следует за сигналом быстро, при шуме (ER ≈ 0) почти замирает.

**Когда брать**: когда сигнал чередует периоды тренда и хаоса, и фиксированный N для скользящей средней не подходит. Эффективно при ≥ 64 каналах.

---

#### Алгоритмы

```
FIR:  y[ch][n] = Σ h[k] · x[ch][n-k],  k=0..N-1   (direct-form)
IIR:  y[n] = b0·x[n] + w1              (DFII-T, biquad cascade)
      w1   = b1·x[n] - a1·y[n] + w2
      w2   = b2·x[n] - a2·y[n]
```

---

#### Классы

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

#### Быстрый старт

##### C++ — FirFilter (OpenCL)

```cpp
#include <spectrum/filters/fir_filter.hpp>

filters::FirFilter fir(backend);
fir.SetCoefficients({0.1f, 0.2f, 0.4f, 0.2f, 0.1f});  // или LoadConfig(json)

auto result = fir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);  // caller owns!
```

##### C++ — IirFilter (OpenCL)

```cpp
#include <spectrum/filters/iir_filter.hpp>

filters::BiquadSection sec;
sec.b0 = 0.02008337f;  sec.b1 = 0.04016673f;  sec.b2 = 0.02008337f;
sec.a1 = -1.56101808f; sec.a2 = 0.64135154f;  // Butterworth 2nd order, fc=0.1

filters::IirFilter iir(backend);
iir.SetBiquadSections({sec});

auto result = iir.Process(input_buf, channels, points);
clReleaseMemObject(result.data);
```

##### C++ — FirFilterROCm (Linux + AMD GPU)

```cpp
#include <spectrum/filters/fir_filter_rocm.hpp>

filters::FirFilterROCm fir(rocm_backend);
fir.SetCoefficients(coeffs);

// Из CPU (upload + process)
auto res = fir.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!

// Из GPU ptr
auto res2 = fir.Process(gpu_ptr, channels, points);
hipFree(res2.data);
```

##### Python — FirFilter (OpenCL)

```python
import dsp_spectrum as gw
import scipy.signal as sig

ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilter(ctx)

taps = sig.firwin(64, 0.1)
fir.set_coefficients(taps.tolist())

result = fir.process(signal)  # (channels, points) или 1D complex64
```

##### Python — IirFilter (OpenCL)

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

##### Python — FirFilterROCm (Linux + AMD GPU)

```python
ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilterROCm(ctx)
fir.set_coefficients(sig.firwin(64, 0.1).tolist())
result = fir.process(data)  # np.ndarray complex64
```

##### C++ — MovingAverageFilterROCm (Linux + AMD GPU)

```cpp
#include <spectrum/filters/moving_average_filter_rocm.hpp>

filters::MovingAverageFilterROCm ma(rocm_backend);
ma.SetParams(filters::MAType::EMA, 10);  // или SMA, MMA, DEMA, TEMA

auto res = ma.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**MAType**: `SMA` (ring buffer, max N=128), `EMA` (α=2/(N+1)), `MMA` (α=1/N), `DEMA` (2×EMA1-EMA2), `TEMA` (3×EMA1-3×EMA2+EMA3).

##### C++ — KalmanFilterROCm (Linux + AMD GPU)

```cpp
#include <spectrum/filters/kalman_filter_rocm.hpp>

filters::KalmanFilterROCm kalman(rocm_backend);
// SetParams(Q, R, x0, P0)
// Q/R << 1: сильное сглаживание | Q/R >> 1: быстрая реакция
kalman.SetParams(0.1f, 25.0f, 0.0f, 25.0f);

auto res = kalman.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**Применение**: сглаживание огибающей/фазы beat-сигнала, трекинг амплитуды.

##### C++ — KaufmanFilterROCm (Linux + AMD GPU)

```cpp
#include <spectrum/filters/kaufman_filter_rocm.hpp>

filters::KaufmanFilterROCm kauf(rocm_backend);
// SetParams(er_period, fast_period, slow_period) — max er_period=128
kauf.SetParams(10, 2, 30);  // стандартные параметры Kaufman

auto res = kauf.ProcessFromCPU(cpu_data, channels, points);
hipFree(res.data);  // caller owns!
```

**Поведение**: тренд (ER≈1) → быстрое следование; шум (ER≈0) → KAMA почти заморожен.

---

#### Ключевые нюансы

| Параметр | FIR (OpenCL) | IIR (OpenCL) |
|----------|-------------|-------------|
| NDRange | 2D `(ch, ⌈pts/256⌉×256)` | 1D `(ch,)` |
| Коэффициенты | `__constant` ≤ 16 000 тапов, иначе `__global` | `__constant` SOS-матрица |
| Параллелизм | По каналам И семплам | Только по каналам |
| Рекомендовано | ≥ 8 каналов | ≥ 8 каналов |

---

#### On-disk kernel cache

FirFilter и IirFilter используют core KernelCacheService:
- **Первый запуск:** компиляция → Save в `spectrum/kernels/bin/`
- **Повторный:** Load binary (~1 мс вместо ~50 мс компиляции)
- **Fallback:** при отсутствии cache — компиляция из source

---

#### Важные ловушки

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

#### Тесты

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

#### Важные ловушки

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

#### Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты с rationale
- [API.md](API.md) — полный API-справочник: все классы, сигнатуры, цепочки вызовов
- [gpu_filters_research.md](gpu_filters_research.md) — Overlap-Save/Add, tiled FIR — будущие алгоритмы

---

*Обновлено: 2026-03-09*

---

## Компонент: LCH Farrow


> Дробная задержка Lagrange 48×5 на GPU

---

#### Концепция — зачем и что это такое

**Зачем нужен модуль?**
Когда антенная решётка принимает сигнал, он приходит на каждую антенну в немного разное время — из-за разных расстояний до источника. Разница во времени прихода (Time Difference of Arrival, TDOA) — ключевая информация для определения направления на источник.

Проблема: задержка, как правило, не кратна периоду дискретизации. Например, 2.7 мкс при частоте 1 МГц = 2.7 отсчёта. Стандартным сдвигом массива это не воспроизведёшь — нужна дробная задержка.

---

##### Что такое фильтр Лагранжа (Farrow)

Дробная задержка реализуется через интерполирующий фильтр: берёт 5 соседних отсчётов и вычисляет «виртуальный» отсчёт между ними — с субсэмпловой точностью. Коэффициенты фильтра зависят от дробной части задержки.

Структура 48×5: 48 предвычисленных наборов коэффициентов (на каждые 1/48 от шага дискретизации), 5 тапов каждый. Таблица хранится в GPU-памяти и используется для линейного интерполирования нужного набора.

---

##### Практическое применение

**DelayedFormSignalGenerator** внутри себя использует именно LchFarrow для применения задержки между каналами. Если генератор даёт задержку через Farrow — это значит, что сигнал каждой антенны реально сдвинут с субсэмпловой точностью.

Для других задач (когда нужно задержать уже существующий сигнал — не сгенерированный) используй LchFarrow или LchFarrowROCm напрямую.

---

##### OpenCL vs ROCm

**LchFarrow (OpenCL)** — работает на любом GPU. Входные данные — `cl_mem` (уже на GPU).

**LchFarrowROCm** — только AMD. Умеет принимать данные с CPU (`ProcessFromCPU`) и с GPU-указателя. Чуть другой интерфейс.

---

##### Ограничение

Входной сигнал должен быть достаточно длинным (больше 4 отсчётов + задержка). При задержке больше длины сигнала — выход будет нулевым.

---

#### Алгоритм

```
delay_samples = delay_us * 1e-6 * sample_rate
read_pos = n - delay_samples
if read_pos < 0 → output[n] = 0
center = floor(read_pos), frac = read_pos - center
row = int(frac * 48) % 48
output[n] = sum(L[row][k] * input[center - 1 + k], k=0..4)
```

> ⚠️ `row` — по `frac` позиции чтения, **не** по μ задержки!

---

#### Быстрый старт

##### C++ — OpenCL

```cpp
#include <spectrum/lch_farrow.hpp>

lch_farrow::LchFarrow proc(backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

auto result = proc.Process(input_buf, antennas, points);
// caller: clReleaseMemObject(result.data)
```

##### C++ — ROCm (ENABLE_ROCM=1, Linux)

```cpp
#include <spectrum/lch_farrow_rocm.hpp>

lch_farrow::LchFarrowROCm proc(rocm_backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

// CPU→GPU (ProcessFromCPU — плоский вектор)
auto result = proc.ProcessFromCPU(flat_signal, antennas, points);
// caller: hipFree(result.data)
```

##### Python

```python
proc = dsp_spectrum.LchFarrow(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
delayed = proc.process(signal)
```

---

#### Стейджи профилирования

| Backend | Стейджи | output_dir |
|---------|---------|-----------|
| OpenCL | `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow/` |
| ROCm | `Upload_input`, `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow_ROCm/` |

---

#### Тесты

| Файл | Тесты |
|------|-------|
| `tests/test_lch_farrow.hpp` | OpenCL: 3 теста (zero, int5, frac2.7) |
| `tests/test_lch_farrow_rocm.hpp` | ROCm: 4 теста (+ multi-antenna) |
| `Python_test/lch_farrow/test_lch_farrow.py` | Python: 5 тестов |

---

#### Важные ловушки

| # | Ловушка |
|---|---------|
| 1 | `clReleaseMemObject(result.data)` — caller owns, не забыть |
| 2 | `hipFree(result.data)` — то же для ROCm |
| 3 | `SetDelays` размер должен == `antennas` — иначе exception |
| 4 | Задержки в **мкс**, не сэмплах и не секундах |
| 5 | ROCm первый запуск ~100-200 мс (hiprtc compile) |
| 6 | `row` считается по `frac` позиции чтения, **не** по дробной части задержки μ |

---

#### Ссылки

- [Full.md](Full.md) — математика, C4, детальный разбор тестов
- [API.md](API.md) — полный API reference C++ и Python
- [Python API](../../Python/lch_farrow_api.md) | [tests/README.md](../../../spectrum/tests/README.md)

---

*Обновлено: 2026-03-09*

---

