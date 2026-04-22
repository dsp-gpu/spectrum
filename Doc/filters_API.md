# Filters — API Справочник

> Полные сигнатуры всех публичных классов и методов модуля `filters`

**Namespace**: `filters`
**Каталог**: `modules/filters/include/filters/`

---

## Содержание

1. [FirFilter (OpenCL)](#1-firfilter-opencl)
2. [IirFilter (OpenCL)](#2-iirfilter-opencl)
3. [FirFilterROCm](#3-firfilterrocm)
4. [IirFilterROCm](#4-iirfilterrocm)
5. [MovingAverageFilterROCm](#5-movingaveragefilterrocm)
6. [KalmanFilterROCm](#6-kalmanfilterrocm)
7. [KaufmanFilterROCm](#7-kaufmanfilterrocm)
8. [Типы данных](#8-типы-данных)
9. [Python API (OpenCL)](#9-python-api-opencl)
10. [Python API (ROCm)](#10-python-api-rocm)
11. [Цепочки вызовов](#11-цепочки-вызовов)

---

## 1. FirFilter (OpenCL)

**Файл**: `include/filters/fir_filter.hpp`
**Backend**: OpenCL (`cl_mem` I/O)
**Algorithm**: Direct-form convolution, 2D NDRange `(channels, points)`

```cpp
namespace filters {

class FirFilter {
public:
  // ── Конструктор / Деструктор ──────────────────────────────────────────────

  explicit FirFilter(drv_gpu_lib::IBackend* backend);
  // Компилирует OpenCL kernel при создании (eager).
  // Первый запуск: JIT ~50 мс → save binary в kernels/bin/.
  // Повторный:    load binary ~1 мс.
  // Throws: std::runtime_error если backend null или не инициализирован.

  ~FirFilter();
  FirFilter(FirFilter&&) noexcept;
  FirFilter& operator=(FirFilter&&) noexcept;
  FirFilter(const FirFilter&) = delete;
  FirFilter& operator=(const FirFilter&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void LoadConfig(const std::string& json_path);
  // JSON: { "type": "fir", "coefficients": [f1, f2, ...] }
  // Throws: std::runtime_error если файл не найден или невалиден.

  void SetCoefficients(const std::vector<float>& coeffs);
  // Устанавливает тапы h[k] и загружает на GPU.
  // num_taps <= 16000 → __constant memory (быстро)
  // num_taps >  16000 → __global memory  (медленнее)
  // Throws: std::invalid_argument если coeffs пустой.

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<cl_mem> Process(
      cl_mem input_buf,
      uint32_t channels,
      uint32_t points,
      ProfEvents* prof_events = nullptr);
  // input_buf: [channels × points] cl_float2, row-major.
  // Возвращает: cl_mem [channels × points] — CALLER MUST clReleaseMemObject(result.data).
  // prof_events != nullptr → events собираются для GPUProfiler.
  // prof_events == nullptr → events освобождаются.

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points);
  // CPU reference (direct-form). Используется для валидации GPU-результата.

  // ── Геттеры ───────────────────────────────────────────────────────────────

  uint32_t                   GetNumTaps()      const;
  const std::vector<float>&  GetCoefficients() const;
  bool                       IsReady()         const;
};

using ProfEvents = std::vector<std::pair<const char*, cl_event>>;
static constexpr uint32_t kMaxConstantTaps = 16000;

}  // namespace filters
```

---

## 2. IirFilter (OpenCL)

**Файл**: `include/filters/iir_filter.hpp`
**Backend**: OpenCL (`cl_mem` I/O)
**Algorithm**: Biquad cascade Direct Form II Transposed, 1D NDRange `(channels,)`

```cpp
namespace filters {

class IirFilter {
public:
  explicit IirFilter(drv_gpu_lib::IBackend* backend);
  ~IirFilter();
  IirFilter(IirFilter&&) noexcept;
  IirFilter& operator=(IirFilter&&) noexcept;
  IirFilter(const IirFilter&) = delete;
  IirFilter& operator=(const IirFilter&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void LoadConfig(const std::string& json_path);
  // JSON: { "type": "iir", "sections": [{b0,b1,b2,a1,a2}, ...] }

  void SetBiquadSections(const std::vector<BiquadSection>& sections);
  // Загружает SOS-матрицу на GPU: [num_sections × 5] float.
  // GPU-буфер layout: [b0, b1, b2, a1, a2] для каждой секции.
  // Throws: std::invalid_argument если sections пустой.

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<cl_mem> Process(
      cl_mem input_buf,
      uint32_t channels,
      uint32_t points,
      ProfEvents* prof_events = nullptr);
  // 1D NDRange = channels. Каждый work-item → 1 канал, все points последовательно.
  // Каждая секция читает output предыдущей (sec==0: из input).
  // CALLER MUST clReleaseMemObject(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points);

  // ── Геттеры ───────────────────────────────────────────────────────────────

  uint32_t                          GetNumSections() const;
  const std::vector<BiquadSection>& GetSections()    const;
  bool                              IsReady()        const;
};

}  // namespace filters
```

---

## 3. FirFilterROCm

**Файл**: `include/filters/fir_filter_rocm.hpp`
**Backend**: ROCm/HIP (`void*` device pointers)
**Условие**: `ENABLE_ROCM=1` (Linux + AMD GPU). На Windows — compile-only stub (все методы `throw std::runtime_error`).

```cpp
namespace filters {

#if ENABLE_ROCM

class FirFilterROCm {
public:
  explicit FirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~FirFilterROCm();
  FirFilterROCm(FirFilterROCm&&) noexcept;
  FirFilterROCm& operator=(FirFilterROCm&&) noexcept;
  FirFilterROCm(const FirFilterROCm&) = delete;
  FirFilterROCm& operator=(const FirFilterROCm&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void LoadConfig(const std::string& json_path);

  void SetCoefficients(const std::vector<float>& coeffs);
  // hiprtc компиляция при первом SetCoefficients(). ~100–500 мс.
  // num_taps <= 16000 → __constant. Больше → __global.

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr,           // HIP device ptr [channels × points] float2
      uint32_t channels,
      uint32_t points,
      ROCmProfEvents* prof_events = nullptr);
  // input_ptr НЕ освобождается. CALLER MUST hipFree(result.data).

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels,
      uint32_t points,
      ROCmProfEvents* prof_events = nullptr);
  // Upload + Process. Внутренний input buf кешируется.
  // CALLER MUST hipFree(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points);

  // ── Геттеры ───────────────────────────────────────────────────────────────

  uint32_t                   GetNumTaps()      const;
  const std::vector<float>&  GetCoefficients() const;
  bool                       IsReady()         const;

private:
  static constexpr unsigned int kBlockSize = 256;  // threads per block
};

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;

#endif  // ENABLE_ROCM
}  // namespace filters
```

---

## 4. IirFilterROCm

**Файл**: `include/filters/iir_filter_rocm.hpp`
**Backend**: ROCm/HIP
**Условие**: `ENABLE_ROCM=1`. На Windows — compile-only stub.

```cpp
namespace filters {

#if ENABLE_ROCM

class IirFilterROCm {
public:
  explicit IirFilterROCm(drv_gpu_lib::IBackend* backend);
  ~IirFilterROCm();
  IirFilterROCm(IirFilterROCm&&) noexcept;
  IirFilterROCm& operator=(IirFilterROCm&&) noexcept;
  IirFilterROCm(const IirFilterROCm&) = delete;
  IirFilterROCm& operator=(const IirFilterROCm&) = delete;

  void LoadConfig(const std::string& json_path);
  void SetBiquadSections(const std::vector<BiquadSection>& sections);

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr,
      uint32_t channels,
      uint32_t points,
      ROCmProfEvents* prof_events = nullptr);
  // CALLER MUST hipFree(result.data).

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels,
      uint32_t points,
      ROCmProfEvents* prof_events = nullptr);
  // CALLER MUST hipFree(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points);

  uint32_t                          GetNumSections() const;
  const std::vector<BiquadSection>& GetSections()    const;
  bool                              IsReady()        const;

private:
  static constexpr unsigned int kBlockSize = 256;
};

#endif  // ENABLE_ROCM
}  // namespace filters
```

---

## 5. MovingAverageFilterROCm

**Файл**: `include/filters/moving_average_filter_rocm.hpp`
**Backend**: ROCm/HIP, 1D grid (1 thread per channel)
**Условие**: `ENABLE_ROCM=1`. На Windows — compile-only stub.

```cpp
namespace filters {

enum class MAType {
  SMA,   // Simple MA — ring buffer, вес 1/N (max N=128)
  EMA,   // Exponential MA — alpha = 2/(N+1)
  MMA,   // Modified MA (Wilder) — alpha = 1/N
  DEMA,  // Double EMA — 2*EMA1 − EMA2
  TEMA   // Triple EMA — 3*EMA1 − 3*EMA2 + EMA3
};

struct MovingAverageParams {
  MAType   type        = MAType::SMA;
  uint32_t window_size = 10;   // N. SMA max = 128.
};

#if ENABLE_ROCM

class MovingAverageFilterROCm {
public:
  explicit MovingAverageFilterROCm(
      drv_gpu_lib::IBackend* backend,
      unsigned int block_size = 256);
  ~MovingAverageFilterROCm();
  MovingAverageFilterROCm(MovingAverageFilterROCm&&) noexcept;
  MovingAverageFilterROCm& operator=(MovingAverageFilterROCm&&) noexcept;
  MovingAverageFilterROCm(const MovingAverageFilterROCm&) = delete;
  MovingAverageFilterROCm& operator=(const MovingAverageFilterROCm&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void SetParams(const MovingAverageParams& params);
  void SetParams(MAType type, uint32_t window_size);
  // hiprtc компиляция всех 5 kernel-функций при первом вызове.
  // alpha предвычисляется: EMA = 2/(N+1), MMA = 1/N.

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr,
      uint32_t channels,
      uint32_t points);
  // CALLER MUST hipFree(result.data).

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels,
      uint32_t points);
  // Кешированный input buf: hipMalloc/hipFree только при изменении размера.
  // CALLER MUST hipFree(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points) const;

  // ── Геттеры ───────────────────────────────────────────────────────────────

  MAType   GetType()       const;
  uint32_t GetWindowSize() const;
  bool     IsReady()       const;
};

#endif  // ENABLE_ROCM
}  // namespace filters
```

---

## 6. KalmanFilterROCm

**Файл**: `include/filters/kalman_filter_rocm.hpp`
**Backend**: ROCm/HIP, 1D grid (1 thread per channel)
**Algorithm**: 1D scalar Kalman, constant-state model. Re и Im обрабатываются независимо.
**Условие**: `ENABLE_ROCM=1`. На Windows — compile-only stub.

```cpp
namespace filters {

struct KalmanParams {
  float Q  = 0.1f;    // Process noise variance. Q/R << 1 → strong smoothing.
  float R  = 25.0f;   // Measurement noise variance. Старт: R = noise_sigma².
  float x0 = 0.0f;    // Initial state estimate.
  float P0 = 25.0f;   // Initial error covariance (рекомендуется = R).
};

#if ENABLE_ROCM

class KalmanFilterROCm {
public:
  explicit KalmanFilterROCm(
      drv_gpu_lib::IBackend* backend,
      unsigned int block_size = 256);
  ~KalmanFilterROCm();
  KalmanFilterROCm(KalmanFilterROCm&&) noexcept;
  KalmanFilterROCm& operator=(KalmanFilterROCm&&) noexcept;
  KalmanFilterROCm(const KalmanFilterROCm&) = delete;
  KalmanFilterROCm& operator=(const KalmanFilterROCm&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void SetParams(const KalmanParams& params);
  void SetParams(float Q, float R, float x0 = 0.0f, float P0 = 25.0f);
  // hiprtc компиляция при первом вызове.

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr,
      uint32_t channels,
      uint32_t points);
  // CALLER MUST hipFree(result.data).

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels,
      uint32_t points);
  // CALLER MUST hipFree(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points) const;

  // ── Геттеры ───────────────────────────────────────────────────────────────

  const KalmanParams& GetParams() const;
  bool                IsReady()   const;
};

#endif  // ENABLE_ROCM
}  // namespace filters
```

**Параметр-гайд**:

| Q/R | Поведение |
|-----|-----------|
| 0.001 | Максимальное сглаживание, очень медленная реакция |
| 0.01 | Сильное сглаживание |
| 0.1 | Умеренное сглаживание |
| 1.0 | Быстрая реакция, слабое сглаживание |
| Radar beat: Q=0.01, R=σ² | Типичные настройки для LFM dechirp сигнала |

---

## 7. KaufmanFilterROCm

**Файл**: `include/filters/kaufman_filter_rocm.hpp`
**Backend**: ROCm/HIP, 1D grid (1 thread per channel)
**Algorithm**: KAMA (Kaufman Adaptive Moving Average). Re и Im независимо.
**Условие**: `ENABLE_ROCM=1`. На Windows — compile-only stub.

```cpp
namespace filters {

struct KaufmanParams {
  uint32_t er_period   = 10;  // N — Efficiency Ratio period. MAX = 128.
  uint32_t fast_period = 2;   // EMA period при тренде (ER≈1).
  uint32_t slow_period = 30;  // EMA period при шуме (ER≈0).
};

#if ENABLE_ROCM

class KaufmanFilterROCm {
public:
  explicit KaufmanFilterROCm(
      drv_gpu_lib::IBackend* backend,
      unsigned int block_size = 256);
  ~KaufmanFilterROCm();
  KaufmanFilterROCm(KaufmanFilterROCm&&) noexcept;
  KaufmanFilterROCm& operator=(KaufmanFilterROCm&&) noexcept;
  KaufmanFilterROCm(const KaufmanFilterROCm&) = delete;
  KaufmanFilterROCm& operator=(const KaufmanFilterROCm&) = delete;

  // ── Конфигурация ──────────────────────────────────────────────────────────

  void SetParams(const KaufmanParams& params);
  void SetParams(
      uint32_t er_period,
      uint32_t fast_period = 2,
      uint32_t slow_period = 30);
  // fast_sc = 2/(fast_period+1), slow_sc = 2/(slow_period+1) — предвычисляются.
  // er_period MAX = 128 (ring buffer в регистрах GPU).

  // ── Обработка ─────────────────────────────────────────────────────────────

  drv_gpu_lib::InputData<void*> Process(
      void* input_ptr,
      uint32_t channels,
      uint32_t points);
  // CALLER MUST hipFree(result.data).

  drv_gpu_lib::InputData<void*> ProcessFromCPU(
      const std::vector<std::complex<float>>& data,
      uint32_t channels,
      uint32_t points);
  // CALLER MUST hipFree(result.data).

  std::vector<std::complex<float>> ProcessCpu(
      const std::vector<std::complex<float>>& input,
      uint32_t channels,
      uint32_t points) const;

  // ── Геттеры ───────────────────────────────────────────────────────────────

  const KaufmanParams& GetParams() const;
  bool                 IsReady()   const;
};

#endif  // ENABLE_ROCM
}  // namespace filters
```

---

## 8. Типы данных

### BiquadSection

```cpp
// Файл: include/filters/filter_params.hpp (или filter_types.hpp)
namespace filters {

struct BiquadSection {
  float b0 = 1.0f;   // Feedforward
  float b1 = 0.0f;
  float b2 = 0.0f;
  float a1 = 0.0f;   // Feedback (a0 = 1, нормировано)
  float a2 = 0.0f;
};
// H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (1 + a1·z⁻¹ + a2·z⁻²)

}  // namespace filters
```

**scipy SOS → BiquadSection**:
```cpp
// Python scipy: sos = butter(2, 0.1, output='sos')
// sos[i] = [b0, b1, b2, a0, a1, a2], a0 всегда = 1.0
BiquadSection s;
s.b0 = sos[i][0]; s.b1 = sos[i][1]; s.b2 = sos[i][2];
s.a1 = sos[i][4]; s.a2 = sos[i][5];  // a0 = sos[i][3] пропускаем!
```

### FilterConfig (JSON loading)

```cpp
namespace filters {

struct FilterConfig {
  std::string               type;          // "fir" или "iir"
  std::vector<float>        coefficients;  // FIR: h[k]
  std::vector<BiquadSection> sections;     // IIR: biquad sections

  static FilterConfig LoadJson(const std::string& path);
  // Throws std::runtime_error при ошибке.
};

}  // namespace filters
```

**JSON FIR**:
```json
{ "type": "fir", "coefficients": [0.1, 0.2, 0.4, 0.2, 0.1] }
```

**JSON IIR**:
```json
{
  "type": "iir",
  "sections": [
    { "b0": 0.02, "b1": 0.04, "b2": 0.02, "a1": -1.56, "a2": 0.64 }
  ]
}
```

### InputData\<T\>

```cpp
// DrvGPU/interface/input_data.hpp
template <typename T>
struct InputData {
  T      data;   // cl_mem или void* — CALLER OWNS (clReleaseMemObject / hipFree)
  size_t size;   // байт
};
```

### ProfEvents / ROCmProfEvents

```cpp
// OpenCL:
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

// ROCm:
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

---

## 9. Python API (OpenCL)

**Файл биндингов**: `python/py_filters.hpp`
**Статус**: ✅ Зарегистрировано в `python/gpu_worklib_bindings.cpp`

### FirFilter

```python
import gpuworklib as gw
import scipy.signal as sig
import numpy as np

ctx = gw.GPUContext(0)
fir = gw.FirFilter(ctx)

# Конфигурация
fir.set_coefficients(sig.firwin(64, 0.1).tolist())  # list[float]
fir.load_config("path/to/lowpass.json")

# Обработка
# input: numpy.ndarray complex64, shape (points,) или (channels, points)
result = fir.process(input)   # numpy.ndarray complex64, same shape

# Свойства (readonly)
fir.num_taps       # int
fir.coefficients   # list[float]
repr(fir)          # "<FirFilter num_taps=64>"
```

### IirFilter

```python
iir = gw.IirFilter(ctx)

# scipy SOS → sections
sos = sig.butter(2, 0.1, output='sos')
sections = [
    {'b0': float(r[0]), 'b1': float(r[1]), 'b2': float(r[2]),
     'a1': float(r[4]), 'a2': float(r[5])}   # r[3] (a0=1) пропускаем
    for r in sos
]
iir.set_sections(sections)                   # list[dict]
iir.load_config("path/to/butterworth.json")

# Обработка
result = iir.process(signal)   # numpy.ndarray complex64

# Свойства
iir.num_sections   # int
iir.sections       # list[dict] с ключами b0,b1,b2,a1,a2
repr(iir)          # "<IirFilter num_sections=1>"
```

---

## 10. Python API (ROCm)

**Файл биндингов**: `python/py_filters_rocm.hpp`

| Класс | Статус биндингов |
|-------|-----------------|
| `FirFilterROCm` | ✅ Зарегистрирован |
| `IirFilterROCm` | ✅ Зарегистрирован |
| `MovingAverageFilterROCm` | ❌ НЕ зарегистрирован |
| `KalmanFilterROCm` | ❌ НЕ зарегистрирован |
| `KaufmanFilterROCm` | ❌ НЕ зарегистрирован |

### FirFilterROCm (✅)

```python
ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilterROCm(ctx)

fir.set_coefficients(sig.firwin(64, 0.1).tolist())
fir.load_config("path/to/config.json")

result = fir.process(input)   # numpy.ndarray complex64

fir.num_taps       # int
fir.coefficients   # list[float]
repr(fir)          # "<FirFilterROCm num_taps=64>"
```

### IirFilterROCm (✅)

```python
iir = gw.IirFilterROCm(ctx)
iir.set_sections([{'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.56, 'a2': 0.64}])

result = iir.process(signal)   # numpy.ndarray complex64

iir.num_sections   # int
iir.sections       # list[dict]
repr(iir)          # "<IirFilterROCm num_sections=1>"
```

### MovingAverageFilterROCm (❌ — запланировано)

```python
# ВНИМАНИЕ: биндинги не зарегистрированы в gpu_worklib_bindings.cpp!
# Ожидаемый API после регистрации (по test_moving_average_rocm.py):
ma = gw.MovingAverageFilterROCm(ctx)
ma.set_params("EMA", 10)   # type: "SMA"/"EMA"/"MMA"/"DEMA"/"TEMA", window_size
result = ma.process(data)  # numpy.ndarray complex64
ma.is_ready()              # bool
ma.get_window_size()       # int
ma.get_type()              # str
```

### KalmanFilterROCm (❌ — запланировано)

```python
# ВНИМАНИЕ: биндинги не зарегистрированы!
# Ожидаемый API (по test_kalman_rocm.py):
kalman = gw.KalmanFilterROCm(ctx)
kalman.set_params(Q=0.1, R=25.0, x0=0.0, P0=25.0)
result = kalman.process(data)   # numpy.ndarray complex64
```

### KaufmanFilterROCm (❌ — запланировано)

```python
# ВНИМАНИЕ: биндинги не зарегистрированы!
# Ожидаемый API (по test_kaufman_rocm.py):
kauf = gw.KaufmanFilterROCm(ctx)
kauf.set_params(er_period=10, fast_period=2, slow_period=30)
result = kauf.process(data)   # numpy.ndarray complex64
```

---

## 11. Цепочки вызовов

### OpenCL FIR — полная цепочка (C++)

```cpp
// 1. Backend
auto backend = std::make_unique<drv_gpu_lib::OpenCLBackend>();
backend->Initialize(0);

// 2. Фильтр (компилирует kernel или загружает binary cache)
filters::FirFilter fir(backend.get());
fir.SetCoefficients({...});   // или LoadConfig("lowpass.json")

// 3. Входной буфер
cl_context ctx_cl = static_cast<cl_context>(backend->GetNativeContext());
cl_int err;
cl_mem input_buf = clCreateBuffer(ctx_cl,
    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    channels * points * sizeof(std::complex<float>),
    data.data(), &err);

// 4. GPU обработка
auto result = fir.Process(input_buf, channels, points);

// 5. Readback
cl_command_queue queue = static_cast<cl_command_queue>(backend->GetNativeQueue());
std::vector<std::complex<float>> out(channels * points);
clEnqueueReadBuffer(queue, result.data, CL_TRUE, 0,
    out.size() * sizeof(std::complex<float>), out.data(), 0, nullptr, nullptr);

// 6. Освобождение (caller owns)
clReleaseMemObject(input_buf);
clReleaseMemObject(result.data);
```

### ROCm FIR из CPU — минимальный вариант (C++)

```cpp
filters::FirFilterROCm fir(rocm_backend);
fir.SetCoefficients(coeffs);

// Upload + Process (кешированный input buf)
auto res = fir.ProcessFromCPU(cpu_data, channels, points);

// Readback
std::vector<std::complex<float>> out(channels * points);
hipMemcpyDtoH(out.data(), res.data, out.size() * sizeof(std::complex<float>));

// Освобождение
hipFree(res.data);   // CALLER OWNS!
```

### Python scipy → FirFilter → validate (Python)

```python
import gpuworklib as gw
import scipy.signal as sig
import numpy as np

ctx = gw.GPUContext(0)
fir = gw.FirFilter(ctx)

taps = sig.firwin(64, 0.1).astype(np.float32)
fir.set_coefficients(taps.tolist())

# Входной сигнал: 8 каналов × 4096 точек
signal = (np.random.randn(8, 4096) + 1j * np.random.randn(8, 4096)).astype(np.complex64)

gpu_result = fir.process(signal)   # (8, 4096) complex64

# Validate vs scipy
ref = np.array([sig.lfilter(taps, [1.0], signal[ch]) for ch in range(8)])
max_err = np.max(np.abs(gpu_result - ref))
print(f"Max error GPU vs scipy: {max_err:.2e}")  # ожидается < 1e-2
```

### ROCm FIR → Kalman pipeline (C++, только Linux + AMD)

```cpp
// 1. FIR шумоподавление
filters::FirFilterROCm fir(backend);
fir.SetCoefficients(lowpass_coeffs);
auto fir_res = fir.ProcessFromCPU(raw_signal, channels, N);

// 2. Kalman сглаживание beat-сигнала (Q << R → strong smoothing)
filters::KalmanFilterROCm kalman(backend);
kalman.SetParams(0.01f, 0.09f);   // Q/R ≈ 0.1
auto kal_res = kalman.Process(fir_res.data, channels, N);

// fir_res.data больше не нужен → освобождаем
hipFree(fir_res.data);

// Readback kal_res
std::vector<std::complex<float>> out(channels * N);
hipMemcpyDtoH(out.data(), kal_res.data, out.size() * sizeof(std::complex<float>));
hipFree(kal_res.data);
```

---

## Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, детальный rationale тестов
- [Quick.md](Quick.md) — шпаргалка с концепцией и быстрым стартом
- [SciPy firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)
- [SciPy butter SOS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [Digital biquad DFII-T](https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2_transposed)

---

*Обновлено: 2026-03-09*
