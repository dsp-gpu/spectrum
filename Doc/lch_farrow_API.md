# LchFarrow — API Reference

> Дробная задержка Lagrange 48×5 на GPU. OpenCL + ROCm.

**Namespace**: `lch_farrow`
**Каталог**: `modules/lch_farrow/`

---

## Подключение

```cpp
// C++ — OpenCL
#include "lch_farrow.hpp"

// C++ — ROCm (только ENABLE_ROCM=1, Linux + AMD GPU)
#include "lch_farrow_rocm.hpp"
```

```python
import gpuworklib

proc_ocl  = gpuworklib.LchFarrow(ctx)       # ctx: GPUContext
proc_rocm = gpuworklib.LchFarrowROCm(ctx)   # ctx: ROCmGPUContext
```

---

## Типы данных

### ProfEvents (OpenCL)

```cpp
using lch_farrow::ProfEvents =
    std::vector<std::pair<const char*, cl_event>>;
```

Передать в `Process()` для сбора `cl_event`. `nullptr` = production (нулевой overhead).
Стейджи: `"Upload_delay"`, `"Kernel"`.

### ROCmProfEvents

```cpp
using lch_farrow::ROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

Стейджи `ProcessFromCPU`: `"Upload_input"`, `"Upload_delay"`, `"Kernel"`.

### InputData

```cpp
template<typename T>
struct drv_gpu_lib::InputData {
    T        data;               // cl_mem или void* — caller owns
    uint32_t antenna_count;
    uint32_t n_point;
    float    sample_rate;
    size_t   gpu_memory_bytes;
};
```

---

## Класс LchFarrow (OpenCL)

### Конструктор

```cpp
explicit LchFarrow(drv_gpu_lib::IBackend* backend);
```

Инициализирует OpenCL context/queue/device, загружает встроенную матрицу, компилирует kernel (`-cl-fast-relaxed-math`), загружает матрицу в GPU. Бросает `std::runtime_error` если backend null.

### Move-семантика

```cpp
LchFarrow(LchFarrow&&) noexcept;
LchFarrow& operator=(LchFarrow&&) noexcept;
// Copy удалён
```

### Setters

```cpp
void SetDelays(const std::vector<float>& delay_us);
void SetSampleRate(float sample_rate);
void SetNoise(float noise_amplitude,
              float norm_val = 0.7071067811865476f,
              uint32_t noise_seed = 0);
void LoadMatrix(const std::string& json_path);
```

| Метод | Параметр | Примечание |
|-------|----------|------------|
| `SetDelays` | задержки в **мкс** | size должен == antennas при вызове Process |
| `SetSampleRate` | Гц | default 1e6 |
| `SetNoise` | amp=0 → без шума; seed=0 → auto | Philox-2x32-10 + Box-Muller |
| `LoadMatrix` | JSON `{"data": [[...]]}` 240 float32 | re-upload GPU; `std::runtime_error` при ошибке |

### Process (GPU→GPU)

```cpp
drv_gpu_lib::InputData<cl_mem> Process(
    cl_mem input_buf,
    uint32_t antennas,
    uint32_t points,
    ProfEvents* prof_events = nullptr);
```

`input_buf` — не освобождается. Вызывает `clFinish`. Возвращает `InputData<cl_mem>` — **caller обязан** `clReleaseMemObject(result.data)`.

Бросает `std::runtime_error` при ошибке OpenCL; `std::invalid_argument` если `delay_us_.size() != antennas`.

### ProcessCpu (CPU reference)

```cpp
std::vector<std::vector<std::complex<float>>> ProcessCpu(
    const std::vector<std::vector<std::complex<float>>>& input,
    uint32_t antennas, uint32_t points);
```

CPU-реализация того же алгоритма для верификации. `input[antenna][sample]`.

### Getters

```cpp
const std::vector<float>& GetDelays() const;
float GetSampleRate() const;
```

---

## Класс LchFarrowROCm (ENABLE_ROCM=1)

На Windows без `ENABLE_ROCM` — stub, все методы бросают `std::runtime_error("LchFarrowROCm: ROCm not enabled")`.

### Конструктор

```cpp
explicit LchFarrowROCm(drv_gpu_lib::IBackend* backend);
```

Получает `hipStream_t`, компилирует через hiprtc, использует HSACO disk cache. Холодный старт ~100-200 мс, горячий ~1-5 мс.

### Setters

Идентичны `LchFarrow` (SetDelays, SetSampleRate, SetNoise, LoadMatrix).

### Process (GPU→GPU)

```cpp
drv_gpu_lib::InputData<void*> Process(
    void* input_ptr, uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events = nullptr);
```

Принимает HIP device pointer. Стейджи: `Upload_delay`, `Kernel`. **Caller** — `hipFree(result.data)`.

### ProcessFromCPU (CPU→GPU)

```cpp
drv_gpu_lib::InputData<void*> ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events = nullptr);
```

Плоский вектор `data[antennas * points]`. Загружает на GPU (временный буфер, освобождается внутри), обрабатывает. Стейджи: `Upload_input`, `Upload_delay`, `Kernel`. **Caller** — `hipFree(result.data)`.

### ProcessCpu, Getters

Идентичны `LchFarrow`.

---

## Сравнение OpenCL vs ROCm

| Аспект | LchFarrow | LchFarrowROCm |
|--------|-----------|---------------|
| Входной handle | `cl_mem` | `void*` (HIP device ptr) |
| CPU→GPU | нет (нужно самому upload) | `ProcessFromCPU(flat_vector)` |
| Очередь | `cl_command_queue` | `hipStream_t` |
| Компиляция | `clBuildProgram` от .cl файла | `hiprtc` + HSACO кэш |
| Upload_input стейдж | нет | есть в `ProcessFromCPU` |
| Освобождение output | `clReleaseMemObject` | `hipFree` |
| Первый запуск | быстро | ~100-200 мс |

---

## Python API

### LchFarrow (OpenCL)

```python
proc = gpuworklib.LchFarrow(ctx)           # ctx: GPUContext
proc.set_delays([0.0, 2.7, 5.0])           # list[float], мкс
proc.set_sample_rate(1e6)                  # float, Гц
proc.set_noise(0.1, norm_val=0.7071, noise_seed=42)
proc.load_matrix("lagrange_matrix_48x5.json")

result = proc.process(input)               # ndarray complex64
# input shape (points,)           → result shape (points,)
# input shape (antennas, points)  → result shape (antennas, points)

proc.sample_rate    # read-only float
proc.delays         # read-only list[float]
repr(proc)          # "<LchFarrow sample_rate=1000000>"
```

### LchFarrowROCm

```python
ctx  = gpuworklib.ROCmGPUContext(0)
proc = gpuworklib.LchFarrowROCm(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
result = proc.process(input)   # идентичный API
```

Внутри вызывает `ProcessFromCPU` (numpy → flat vector → GPU).

---

## Примеры

### C++ OpenCL — production pipeline

```cpp
#include "DrvGPU/backends/opencl/opencl_backend.hpp"
#include "lch_farrow.hpp"

drv_gpu_lib::OpenCLBackend backend;
backend.Initialize(0);

lch_farrow::LchFarrow proc(&backend);
proc.SetSampleRate(12e6f);
proc.SetDelays({0.0f, 0.5f, 1.0f, 1.5f});  // 4 антенны, мкс

// input_buf — cl_mem уже на GPU (от генератора или предыдущего шага)
auto result = proc.Process(input_buf, 4, 4096);
// ... использовать result.data ...
clReleaseMemObject(result.data);
```

### C++ ROCm — production pipeline

```cpp
#include "backends/rocm/rocm_backend.hpp"
#include "lch_farrow_rocm.hpp"

drv_gpu_lib::ROCmBackend backend;
backend.Initialize(0);

lch_farrow::LchFarrowROCm proc(&backend);
proc.SetSampleRate(12e6f);
proc.SetDelays({0.0f, 0.5f, 1.0f, 1.5f});

std::vector<std::complex<float>> flat(4 * 4096);
// ... заполнить сигналом ...
auto result = proc.ProcessFromCPU(flat, 4, 4096);
hipFree(result.data);
```

### C++ OpenCL — benchmark с профилированием

```cpp
lch_farrow::ProfEvents events;
auto result = proc.Process(input_buf, antennas, points, &events);
clReleaseMemObject(result.data);

for (auto& [name, ev] : events) {
    RecordEvent(name, ev);  // GpuBenchmarkBase
}
// events: {"Upload_delay", cl_event}, {"Kernel", cl_event}
```

### Python — multi-channel delay

```python
import numpy as np
import gpuworklib

ctx = gpuworklib.GPUContext(0)

fs = 1e6
N  = 4096
t  = np.arange(N) / fs
cw = np.exp(1j * 2 * np.pi * 50000 * t).astype(np.complex64)
signal_4ch = np.tile(cw, (4, 1))   # (4, 4096)

proc = gpuworklib.LchFarrow(ctx)
proc.set_sample_rate(fs)
proc.set_delays([0.0, 1.5, 3.0, 5.0])   # мкс

delayed = proc.process(signal_4ch)      # (4, 4096)
print(f"sample_rate: {proc.sample_rate:.0f}")
print(f"delays: {list(proc.delays)}")
```

---

## Цепочки вызовов

### OpenCL — минимальный

```
LchFarrow(backend)
  → SetSampleRate(fs)
  → SetDelays(delays_us)
  → Process(input_cl_mem, antennas, points)
  → [caller] clReleaseMemObject(result.data)
```

### ROCm — через CPU

```
LchFarrowROCm(backend)
  → SetSampleRate(fs)
  → SetDelays(delays_us)
  → ProcessFromCPU(flat_vector, antennas, points)
  → [caller] hipFree(result.data)
```

### С профилированием

```
ProfEvents events / ROCmProfEvents events
  → Process(..., &events) / ProcessFromCPU(..., &events)
  → for (name, ev) : events → RecordEvent(name, ev)
  → Report() → PrintReport() + ExportMarkdown() + ExportJSON()
```

---

## Нюансы API

1. **`SetDelays` перед `Process`** — если не вызван, задержки = 0 (заполняются автоматически).
2. **`result.data` — caller owns** — утечка памяти если не освободить.
3. **`input_buf` не трогается** — Process() не вызывает release.
4. **`matrix_buf_` persistent** — создаётся в конструкторе, пересоздаётся только при `LoadMatrix()`.
5. **`delay_buf_` persistent** — увеличивается при росте числа антенн, не уменьшается.
6. **ROCm HSACO кэш** — файлы в `modules/lch_farrow/kernels/`. Удалить при смене GPU архитектуры.
7. **`row` по frac читаемой позиции, не по μ задержки** — частая ошибка в ранних спецификациях.

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4, детальный разбор тестов
- [Quick.md](Quick.md) — краткий справочник
- [Doc/Python/lch_farrow_api.md](../../Python/lch_farrow_api.md) — Python API подробно

---

*Обновлено: 2026-03-09*
