# LchFarrow — Краткий справочник

> Дробная задержка Lagrange 48×5 на GPU

---

## Концепция — зачем и что это такое

**Зачем нужен модуль?**
Когда антенная решётка принимает сигнал, он приходит на каждую антенну в немного разное время — из-за разных расстояний до источника. Разница во времени прихода (Time Difference of Arrival, TDOA) — ключевая информация для определения направления на источник.

Проблема: задержка, как правило, не кратна периоду дискретизации. Например, 2.7 мкс при частоте 1 МГц = 2.7 отсчёта. Стандартным сдвигом массива это не воспроизведёшь — нужна дробная задержка.

---

### Что такое фильтр Лагранжа (Farrow)

Дробная задержка реализуется через интерполирующий фильтр: берёт 5 соседних отсчётов и вычисляет «виртуальный» отсчёт между ними — с субсэмпловой точностью. Коэффициенты фильтра зависят от дробной части задержки.

Структура 48×5: 48 предвычисленных наборов коэффициентов (на каждые 1/48 от шага дискретизации), 5 тапов каждый. Таблица хранится в GPU-памяти и используется для линейного интерполирования нужного набора.

---

### Практическое применение

**DelayedFormSignalGenerator** внутри себя использует именно LchFarrow для применения задержки между каналами. Если генератор даёт задержку через Farrow — это значит, что сигнал каждой антенны реально сдвинут с субсэмпловой точностью.

Для других задач (когда нужно задержать уже существующий сигнал — не сгенерированный) используй LchFarrow или LchFarrowROCm напрямую.

---

### OpenCL vs ROCm

**LchFarrow (OpenCL)** — работает на любом GPU. Входные данные — `cl_mem` (уже на GPU).

**LchFarrowROCm** — только AMD. Умеет принимать данные с CPU (`ProcessFromCPU`) и с GPU-указателя. Чуть другой интерфейс.

---

### Ограничение

Входной сигнал должен быть достаточно длинным (больше 4 отсчётов + задержка). При задержке больше длины сигнала — выход будет нулевым.

---

## Алгоритм

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

## Быстрый старт

### C++ — OpenCL

```cpp
#include "lch_farrow.hpp"

lch_farrow::LchFarrow proc(backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

auto result = proc.Process(input_buf, antennas, points);
// caller: clReleaseMemObject(result.data)
```

### C++ — ROCm (ENABLE_ROCM=1, Linux)

```cpp
#include "lch_farrow_rocm.hpp"

lch_farrow::LchFarrowROCm proc(rocm_backend);
proc.SetSampleRate(1e6f);
proc.SetDelays({0.0f, 2.7f, 5.0f});

// CPU→GPU (ProcessFromCPU — плоский вектор)
auto result = proc.ProcessFromCPU(flat_signal, antennas, points);
// caller: hipFree(result.data)
```

### Python

```python
proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
delayed = proc.process(signal)
```

---

## Стейджи профилирования

| Backend | Стейджи | output_dir |
|---------|---------|-----------|
| OpenCL | `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow/` |
| ROCm | `Upload_input`, `Upload_delay`, `Kernel` | `Results/Profiler/GPU_00_LchFarrow_ROCm/` |

---

## Тесты

| Файл | Тесты |
|------|-------|
| `tests/test_lch_farrow.hpp` | OpenCL: 3 теста (zero, int5, frac2.7) |
| `tests/test_lch_farrow_rocm.hpp` | ROCm: 4 теста (+ multi-antenna) |
| `Python_test/lch_farrow/test_lch_farrow.py` | Python: 5 тестов |

---

## Важные ловушки

| # | Ловушка |
|---|---------|
| 1 | `clReleaseMemObject(result.data)` — caller owns, не забыть |
| 2 | `hipFree(result.data)` — то же для ROCm |
| 3 | `SetDelays` размер должен == `antennas` — иначе exception |
| 4 | Задержки в **мкс**, не сэмплах и не секундах |
| 5 | ROCm первый запуск ~100-200 мс (hiprtc compile) |
| 6 | `row` считается по `frac` позиции чтения, **не** по дробной части задержки μ |

---

## Ссылки

- [Full.md](Full.md) — математика, C4, детальный разбор тестов
- [API.md](API.md) — полный API reference C++ и Python
- [Python API](../../Python/lch_farrow_api.md) | [tests/README.md](../../../modules/lch_farrow/tests/README.md)

---

*Обновлено: 2026-03-09*
