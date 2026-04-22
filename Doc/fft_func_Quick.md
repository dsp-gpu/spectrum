# fft_func — Краткий справочник

> GPU FFT + поиск максимумов спектра: ROCm/hipFFT (ветка main, AMD RDNA4+)

**Namespace**: `fft_processor` + `antenna_fft` | **Каталог**: `modules/fft_func/`

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
В пайплайне ЦОС часто нужно: взять IQ-сигнал, посчитать его спектр (FFT) и найти в нём частоты сигналов. `fft_func` делает это на GPU — принимает данные с CPU или GPU, возвращает спектр или позиции пиков. Это центральный FFT-модуль всего проекта: его используют гетеродин, статистика, стратегии.

**Модуль появился слиянием** двух бывших модулей (`fft_processor` + `fft_maxima`). Старая документация лежит в `Doc/Modules/~!/` только как архив.

---

## FFTProcessorROCm — пакетный FFT

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

## ComplexToMagPhaseROCm — IQ → амплитуда+фаза (без FFT)

**Что делает**: прямое поэлементное вычисление `|z|` и `atan2(im, re)` — никакого FFT. Чисто математическое преобразование сигнала во временной области.

**Когда брать**: нужна огибающая сигнала (амплитуда и фаза отсчёт за отсчётом), а не спектр. Также: pipeline статистики (ProcessMagnitudeToBuffer → compute_statistics).

**Методы**:
- `Process()` — CPU/GPU → CPU (`MagPhaseResult` или `MagnitudeResult`)
- `ProcessToGPU()` — CPU/GPU → GPU (interleaved `{mag, phase}` float2, **CALLER OWNS!**)
- `ProcessMagnitude()` — только амплитуда (без фазы), с нормализацией
- `ProcessMagnitudeToBuffer()` — zero-alloc: пишет в чужой буфер (для pipeline)

**norm_coeff**: `0` или `1.0` = без нормировки; `-1.0` = делить на `n_point`; `>0` = умножить.

---

## SpectrumMaximaFinder — поиск пиков

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

## Аналогии и связи

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

## Критические отличия двух типов фазы

| Класс / структура | Поле | Единицы |
|-------------------|------|---------|
| `FFTMagPhaseResult.phase` | phase[k] | **РАДИАНЫ** `[-π, π]` |
| `ComplexToMagPhaseROCm` (MagPhaseResult.phase) | phase[k] | **РАДИАНЫ** `[-π, π]` |
| `MaxValue.phase` | phase | **ГРАДУСЫ** `[-180°, 180°]` |

Не перепутай при сравнении результатов!

---

## Ограничения

- **Только ROCm** в ветке `main` (AMD GPU, RDNA4+ gfx1201+). clFFT мёртв на RDNA4+ — все OpenCL тесты закомментированы.
- **AllMaxima: `n_point = nFFT`** — передавать размер FFT, не n_point сигнала.
- **Python: find_all_maxima принимает FFT**, а не сырой сигнал. Сначала `fft.process_complex()`.
- **ProcessToGPU — caller owns**: ОБЯЗАТЕЛЬНО освободить `backend->Free(ptr)` или `hipFree(ptr)`.
- **OutputDestination::GPU — caller owns**: `hipFree(result.gpu_maxima)` и `hipFree(result.gpu_counts)`.

---

## Быстрый старт

### C++ — FFTProcessorROCm

```cpp
#include "fft_processor_rocm.hpp"

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

### C++ — SpectrumMaximaFinder (ONE_PEAK)

```cpp
#include "interface/spectrum_maxima_types.h"
#include "interface/spectrum_input_data.hpp"

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

### C++ — ComplexToMagPhaseROCm

```cpp
#include "complex_to_mag_phase_rocm.hpp"

fft_processor::ComplexToMagPhaseROCm converter(backend);
fft_processor::MagPhaseParams params;
params.beam_count = 4;
params.n_point    = 2048;

auto results = converter.Process(data, params);
// results[b].phase[k] — фаза в РАДИАНАХ
```

### Python — FFTProcessorROCm

```python
import gpuworklib
import numpy as np

ctx = gpuworklib.ROCmGPUContext(0)
fft = gpuworklib.FFTProcessorROCm(ctx)

# 8 лучей, 1024 точки
signal = np.random.randn(8 * 1024).view(np.complex64)

# Комплексный спектр (flat → flat)
spectrum = fft.process_complex(signal, 1e6, beam_count=8, n_point=1024)

# Или 2D → 2D
spectrum = fft.process_complex(signal.reshape(8, 1024), 1e6)

# Mag+Phase
result = fft.process_mag_phase(signal, 1e6, beam_count=8, include_freq=True)
# result['magnitude'], result['phase'] (РАДИАНЫ!), result['frequency']
```

### Python — SpectrumMaximaFinder

```python
ctx = gpuworklib.ROCmGPUContext(0)
fft = gpuworklib.FFTProcessorROCm(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

signal = np.zeros(1024, dtype=np.complex64)
# ... заполнить ...

# ШАГ 1: FFT (обязательно сначала!)
spectrum = fft.process_complex(signal, 1000.0)

# ШАГ 2: найти все пики
result = finder.find_all_maxima(spectrum, sample_rate=1000.0)
print(result['frequencies'])  # Hz
```

---

## Таблица: какой класс выбрать

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

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, все тесты
- [API.md](API.md) — полный API-справочник (сигнатуры, параметры)
- [Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md](../../../Doc_Addition/Info_ROCm_HIP_Optimization_Guide.md) — оптимизация HIP/ROCm ядер

---

*Обновлено: 2026-03-28*
