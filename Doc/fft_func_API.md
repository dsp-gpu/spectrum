# fft_func — API Reference

> Полный справочник по классам, методам и типам модуля `fft_func`

---

## Содержание

1. [Типы данных — FFTProcessorROCm](#1-типы-данных--fftprocessorrocm)
2. [FFTProcessorROCm](#2-fftprocessorrocm)
3. [Типы данных — ComplexToMagPhaseROCm](#3-типы-данных--complextomag-phaserocm)
4. [ComplexToMagPhaseROCm](#4-complextomag-phaserocm)
5. [Типы данных — SpectrumMaximaFinder](#5-типы-данных--spectrummaxima-finder)
6. [SpectrumMaximaFinder](#6-spectrummaxima-finder)
7. [SpectrumProcessorFactory](#7-spectrumprocessorfactory)
8. [Python API](#8-python-api)
9. [Примеры цепочек вызовов](#9-примеры-цепочек-вызовов)

---

## 1. Типы данных — FFTProcessorROCm

**Namespace**: `fft_processor` | **Заголовки**: `include/types/fft_params.hpp`, `include/types/fft_results.hpp`, `include/types/fft_modes.hpp`

```cpp
// ── Параметры ─────────────────────────────────────────────────────────────

enum class FFTOutputMode {
    COMPLEX,               ///< Комплексный спектр X[k]
    MAGNITUDE_PHASE,       ///< |X[k]| + arg(X[k]) [рад]
    MAGNITUDE_PHASE_FREQ   ///< То же + freq[k] = k×fs/nFFT [Hz]
};

struct FFTProcessorParams {
    uint32_t beam_count   = 1;       ///< Параллельных лучей
    uint32_t n_point      = 0;       ///< Точек на луч (до padding)
    float    sample_rate  = 1000.0f; ///< Частота дискретизации, Гц
    FFTOutputMode output_mode = FFTOutputMode::COMPLEX;
    uint32_t repeat_count = 1;       ///< nFFT = nextPow2(n_point) × repeat_count
    float    memory_limit = 0.80f;   ///< Доля GPU памяти для батча [0..1]
};

// ── Результаты ────────────────────────────────────────────────────────────

struct FFTBeamResult {
    uint32_t beam_id    = 0;    ///< Индекс луча (глобальный, с учётом батчей)
    uint32_t nFFT       = 0;    ///< Размер FFT: nextPow2(n_point) × repeat_count
    float    sample_rate = 0.0f; ///< Копия из params.sample_rate
};

struct FFTComplexResult : FFTBeamResult {
    std::vector<std::complex<float>> spectrum; ///< [nFFT], ненормализован
    // Для физической амплитуды: amplitude = |spectrum[k]| / n_point
};

struct FFTMagPhaseResult : FFTBeamResult {
    std::vector<float> magnitude;  ///< [nFFT], |X[k]| = sqrt(re²+im²), ненормализован
    std::vector<float> phase;      ///< [nFFT], atan2(im, re) в РАДИАНАХ [-π, π]
    std::vector<float> frequency;  ///< [nFFT], Hz — только MAGNITUDE_PHASE_FREQ (иначе пустой)
};

struct FFTProfilingData {
    double upload_time_ms;          ///< CPU→GPU или D2D copy
    double fft_time_ms;             ///< hipfftExecC2C
    double post_processing_time_ms; ///< c2mp kernel (0 при режиме COMPLEX)
    double download_time_ms;        ///< GPU→CPU
    double total_time_ms;           ///< Суммарное
};

// ── ROCm профилирование по стадиям ────────────────────────────────────────

using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
// Стадии: "Upload", "PadData", "FFT", "MagPhase", "Download"
```

---

## 2. FFTProcessorROCm

**Заголовок**: `include/fft_processor_rocm.hpp` | **Namespace**: `fft_processor`
**Требует**: `ENABLE_ROCM=1`

```cpp
class FFTProcessorROCm {
public:
  // ── Конструктор ──────────────────────────────────────────────────────────

  explicit FFTProcessorROCm(drv_gpu_lib::IBackend* backend);
  ~FFTProcessorROCm();

  // Non-copyable, movable
  FFTProcessorROCm(FFTProcessorROCm&& other) noexcept;
  FFTProcessorROCm& operator=(FFTProcessorROCm&& other) noexcept;

  // ── Комплексный спектр ───────────────────────────────────────────────────

  // CPU вход
  std::vector<FFTComplexResult> ProcessComplex(
      const std::vector<std::complex<float>>& data,
      const FFTProcessorParams& params,
      ROCmProfEvents* prof_events = nullptr);

  // GPU вход (void* — без upload, ROCm device pointer)
  std::vector<FFTComplexResult> ProcessComplex(
      void* gpu_data,
      const FFTProcessorParams& params,
      size_t gpu_memory_bytes = 0);

  // ── Амплитуда + Фаза ─────────────────────────────────────────────────────

  // CPU вход
  std::vector<FFTMagPhaseResult> ProcessMagPhase(
      const std::vector<std::complex<float>>& data,
      const FFTProcessorParams& params,
      ROCmProfEvents* prof_events = nullptr);

  // GPU вход
  std::vector<FFTMagPhaseResult> ProcessMagPhase(
      void* gpu_data,
      const FFTProcessorParams& params,
      size_t gpu_memory_bytes = 0);

  // ── Информация ───────────────────────────────────────────────────────────

  FFTProfilingData GetProfilingData() const;
  uint32_t GetNFFT() const;  ///< nFFT после первого вызова; 0 до вызова
};
```

**Параметры ProcessComplex / ProcessMagPhase**:

| Параметр | Тип | Описание |
|----------|-----|----------|
| `data` | `vector<complex<float>>` | Flat: beam 0 points, beam 1 points, ... |
| `gpu_data` | `void*` | ROCm device pointer (hipDevicePtr) |
| `gpu_memory_bytes` | `size_t` | Размер GPU буфера (0 = авто: beam_count×n_point×8) |
| `prof_events` | `ROCmProfEvents*` | Заполняется метками стадий (nullptr = без профилирования) |

---

## 3. Типы данных — ComplexToMagPhaseROCm

**Namespace**: `fft_processor` | **Заголовок**: `include/types/mag_phase_types.hpp`

```cpp
struct MagPhaseParams {
    uint32_t beam_count   = 1;     ///< Параллельных лучей/каналов
    uint32_t n_point      = 0;     ///< Точек на луч
    float    memory_limit = 0.80f; ///< Доля GPU памяти [0..1]
    float    norm_coeff   = 1.0f;  ///< Нормировка:
                                   ///< 0.0f или 1.0f → ×1 (без нормировки)
                                   ///< -1.0f → ÷n_point
                                   ///< >0 (не 1.0) → умножить на значение
};

struct MagPhaseResult {
    uint32_t beam_id = 0;
    uint32_t n_point = 0;
    std::vector<float> magnitude;  ///< |z| = sqrt(re²+im²)
    std::vector<float> phase;      ///< arg(z) = atan2(im, re) в РАДИАНАХ [-π, π]
};

struct MagnitudeResult {
    uint32_t beam_id = 0;
    uint32_t n_point = 0;
    std::vector<float> magnitude;  ///< |z| × inv_n (нормировка применена)
};
```

---

## 4. ComplexToMagPhaseROCm

**Заголовок**: `include/complex_to_mag_phase_rocm.hpp` | **Namespace**: `fft_processor`
**Требует**: `ENABLE_ROCM=1`

```cpp
class ComplexToMagPhaseROCm {
public:
  explicit ComplexToMagPhaseROCm(drv_gpu_lib::IBackend* backend);
  ~ComplexToMagPhaseROCm();
  // Non-copyable, movable

  // ── CPU вход → CPU выход ─────────────────────────────────────────────────

  std::vector<MagPhaseResult> Process(
      const std::vector<std::complex<float>>& data,
      const MagPhaseParams& params);
  // Возвращает: mag (ненормализован) + phase (РАДИАНЫ)

  // ── GPU вход → CPU выход ─────────────────────────────────────────────────

  std::vector<MagPhaseResult> Process(
      void* gpu_data,
      const MagPhaseParams& params,
      size_t gpu_memory_bytes = 0);

  // ── CPU вход → GPU выход (CALLER OWNS!) ──────────────────────────────────

  void* ProcessToGPU(
      const std::vector<std::complex<float>>& data,
      const MagPhaseParams& params);
  // Возвращает: interleaved float2[beam_count × n_point]
  //   out[i].x = magnitude[i], out[i].y = phase[i] (радианы)
  // ОБЯЗАТЕЛЬНО: backend->Free(ptr) или hipFree(ptr) после использования!

  // ── GPU вход → GPU выход (CALLER OWNS!) ──────────────────────────────────

  void* ProcessToGPU(
      void* gpu_data,
      const MagPhaseParams& params,
      size_t gpu_memory_bytes = 0);
  // Zero-copy path. Те же требования по освобождению памяти.

  // ── Только амплитуда (GPU → CPU) ─────────────────────────────────────────

  std::vector<MagnitudeResult> ProcessMagnitude(
      void* gpu_data,
      const MagPhaseParams& params,
      size_t gpu_memory_bytes = 0);
  // params.norm_coeff управляет нормировкой

  // ── Только амплитуда (GPU → GPU, CALLER OWNS!) ───────────────────────────

  void* ProcessMagnitudeToGPU(
      void* gpu_data,
      const MagPhaseParams& params,
      size_t gpu_memory_bytes = 0);
  // Возвращает: float[beam_count × n_point]
  // CALLER OWNS — must hipFree!

  // ── Zero-alloc: GPU → чужой GPU буфер ───────────────────────────────────

  void ProcessMagnitudeToBuffer(
      void* gpu_complex_in,
      void* gpu_magnitude_out,
      const MagPhaseParams& params);
  // Нулевые аллокации. Читает из gpu_complex_in, пишет в gpu_magnitude_out.
  // Используется стратегиями для pipeline: FFT out → statistics in.
};
```

---

## 5. Типы данных — SpectrumMaximaFinder

**Namespace**: `antenna_fft` | **Заголовки**: `include/types/spectrum_*.hpp`, `include/interface/spectrum_input_data.hpp`

```cpp
// ── Режимы поиска ─────────────────────────────────────────────────────────

enum class PeakSearchMode {
    ONE_PEAK,    ///< Один пик + 3 соседних → 4 MaxValue
    TWO_PEAKS,   ///< Два пика + 3 соседних каждый → 8 MaxValue
    ALL_MAXIMA   ///< Все локальные максимумы (stream compaction)
};

using DriverType = drv_gpu_lib::BackendType;  ///< OPENCL, ROCm, AUTO

// ── Входные данные ────────────────────────────────────────────────────────

template<typename T>
struct InputData {
    uint32_t antenna_count = 0;
    uint32_t n_point = 0;          ///< Для AllMaxima: n_point = nFFT!
    T data{};                      ///< vector<complex<float>> или void* (ROCm)
    size_t gpu_memory_bytes = 0;
    uint32_t repeat_count = 2;     ///< nFFT = nextPow2(n_point) × repeat_count
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;     ///< 0 = auto = nFFT/4
    float memory_limit = 0.80f;
    size_t max_maxima_per_beam = 1000;
};

// ── Один максимум ─────────────────────────────────────────────────────────

struct MaxValue {                  // 32 байта (с pad)
    uint32_t index;                ///< Бин FFT [0, nFFT)
    float real, imag;              ///< Re/Im FFT[index]
    float magnitude;               ///< |FFT[index]| = sqrt(re²+im²)
    float phase;                   ///< arg(FFT[index]) в ГРАДУСАХ = atan2×180/π
                                   ///< ВНИМАНИЕ: ГРАДУСЫ, не радианы!
    float freq_offset;             ///< δ параболической интерп. ∈ [-0.5, +0.5]
    float refined_frequency;       ///< (index + δ) × sample_rate / nFFT [Hz]
                                   ///< В compact_maxima δ=0 (нет интерп.)
    uint32_t pad;
};

// ── Результат ONE_PEAK / TWO_PEAKS ────────────────────────────────────────

struct SpectrumResult {
    uint32_t antenna_id;
    MaxValue interpolated;   ///< peak[0] с параболической интерполяцией
    MaxValue left_point;     ///< FFT[peak_bin - 1]
    MaxValue center_point;   ///< FFT[peak_bin]
    MaxValue right_point;    ///< FFT[peak_bin + 1]
};

// ── Результат AllMaxima ───────────────────────────────────────────────────

struct AllMaximaBeamResult {
    uint32_t antenna_id;
    uint32_t num_maxima;
    std::vector<MaxValue> maxima;  ///< Пусто при Dest=GPU
};

struct AllMaximaResult {
    std::vector<AllMaximaBeamResult> beams;  ///< Заполнено при Dest=CPU/ALL
    OutputDestination destination;
    void* gpu_maxima = nullptr;  ///< [beam_count × max_per_beam × sizeof(MaxValue)]
                                 ///< CALLER OWNS при Dest=GPU или ALL!
    void* gpu_counts = nullptr;  ///< [beam_count × sizeof(uint32_t)]
                                 ///< CALLER OWNS при Dest=GPU или ALL!
    size_t total_maxima = 0;
    size_t gpu_bytes = 0;
    size_t TotalMaxima() const { return total_maxima; }
};

// ── Параметры SpectrumParams ──────────────────────────────────────────────
// (заполняется автоматически из InputData через PrepareParams)

struct SpectrumParams {
    uint32_t antenna_count = 5;
    uint32_t n_point = 1000;
    uint32_t repeat_count = 2;         ///< nFFT = nextPow2(n_point) × repeat_count
    float sample_rate = 1000.0f;
    uint32_t search_range = 0;         ///< 0 = auto = nFFT/4
    PeakSearchMode peak_mode = PeakSearchMode::ONE_PEAK;
    float memory_limit = 0.80f;
    size_t max_maxima_per_beam = 1000;
    uint32_t nFFT = 0;    ///< ВЫЧИСЛЯЕМОЕ
    uint32_t base_fft = 0; ///< ВЫЧИСЛЯЕМОЕ = nextPow2(n_point)
};
```

---

## 6. SpectrumMaximaFinder

**Заголовок**: `include/interface/spectrum_maxima_types.h` (агрегатор) | **Namespace**: `antenna_fft`

```cpp
class SpectrumMaximaFinder {
public:
  // ── Конструктор ──────────────────────────────────────────────────────────

  explicit SpectrumMaximaFinder(drv_gpu_lib::IBackend* backend);
  // Initialize() вызывается автоматически при первом Process

  // DEPRECATED:
  [[deprecated]] SpectrumMaximaFinder(SpectrumParams params,
                                       drv_gpu_lib::IBackend* backend);

  bool IsInitialized() const;

  // ── Process — один или два пика ───────────────────────────────────────

  template<typename T>
  std::vector<SpectrumResult> Process(
      const InputData<T>& input,
      PeakSearchMode mode,          ///< ONE_PEAK или TWO_PEAKS
      DriverType driver = DriverType::AUTO);
  // T = vector<complex<float>> или void*
  // Возвращает: один SpectrumResult на антенну

  // ── FindAllMaxima — все пики из сырого сигнала ────────────────────────

  template<typename T>
  AllMaximaResult FindAllMaxima(
      const InputData<T>& input,
      OutputDestination dest = OutputDestination::CPU,
      DriverType driver = DriverType::AUTO,
      uint32_t search_start = 0,    ///< 0 = auto = 1 (пропуск DC)
      uint32_t search_end = 0);     ///< 0 = auto = nFFT/2
  // T = vector<complex<float>> или void*

  // Перегрузка для готового GPU FFT буфера (cl_mem или void*)
  AllMaximaResult FindAllMaxima(
      void* gpu_fft_buffer,
      uint32_t beam_count,
      uint32_t nFFT,
      float sample_rate,
      OutputDestination dest = OutputDestination::CPU,
      uint32_t search_start = 0,
      uint32_t search_end = 0);

  // ── AllMaxima — только detect+scan+compact (без FFT) ─────────────────

  template<typename T>
  AllMaximaResult AllMaxima(
      const InputData<T>& input,    ///< n_point ДОЛЖЕН быть = nFFT!
      OutputDestination dest = OutputDestination::CPU,
      DriverType driver = DriverType::AUTO,
      uint32_t search_start = 0,
      uint32_t search_end = 0);

  // ── Информация ───────────────────────────────────────────────────────

  ProfilingData GetProfilingData() const;
  SpectrumParams GetParams() const;
};
```

---

## 7. SpectrumProcessorFactory

**Заголовок**: `include/factory/spectrum_processor_factory.hpp` | **Namespace**: `antenna_fft`

```cpp
class SpectrumProcessorFactory {
public:
    static std::unique_ptr<ISpectrumProcessor> Create(
        drv_gpu_lib::BackendType backend_type,
        drv_gpu_lib::IBackend* backend);
    // backend_type: OPENCL, ROCm, AUTO (→ OPENCL)
};
```

---

## 8. Python API

### 8.1 FFTProcessorROCm

**Модуль**: `gpuworklib` | **Биндинг**: `python/py_fft_processor_rocm.hpp`

```python
class FFTProcessorROCm:
    def __init__(self, ctx: ROCmGPUContext): ...

    def process_complex(
        self,
        data: np.ndarray,        # complex64, flat 1D или 2D [B, N]
        sample_rate: float,      # позиционный аргумент!
        beam_count: int = 0,     # 0 → из формы массива
        n_point: int = 0         # 0 → из формы массива
    ) -> np.ndarray:             # complex64, 1D (1 луч) или 2D [B, nFFT]
        ...

    def process_mag_phase(
        self,
        data: np.ndarray,
        sample_rate: float,
        beam_count: int = 0,
        n_point: int = 0,
        include_freq: bool = True
    ) -> dict:  # ключи: 'magnitude', 'phase' (радианы!),
                # 'frequency' (если include_freq), 'nFFT', 'sample_rate'
        ...

    @property
    def nfft(self) -> int: ...   # read-only, 0 до первого вызова
```

**Логика автоопределения beam_count/n_point**:
- 2D ndarray `[B, N]`: `beam_count=B`, `n_point=N`
- 1D с явным `beam_count`: `n_point = len(data) / beam_count`
- 1D без `beam_count`: `beam_count=1`, `n_point=len(data)`

### 8.2 SpectrumMaximaFinder

**Модуль**: `gpuworklib` | **Биндинг**: `python/gpu_worklib_bindings.cpp`

```python
class SpectrumMaximaFinder:
    def __init__(self, ctx):  # GPUContext или ROCmGPUContext
        ...

    def find_all_maxima(
        self,
        fft_data: np.ndarray,    # complex64, FFT-СПЕКТР (не сырой сигнал!)
        sample_rate: float,
        beam_count: int = 0,     # 0 = auto
        nFFT: int = 0,           # 0 = auto
        search_start: int = 0,   # 0 = auto = 1 (пропуск DC)
        search_end: int = 0      # 0 = auto = nFFT/2
    ) -> dict | list[dict]:      # dict если 1 луч, list[dict] если несколько
        ...

# Формат dict одного луча:
{
    'num_maxima':  int,             # количество найденных максимумов
    'positions':   np.array(uint32), # бины [num_maxima]
    'magnitudes':  np.array(float32),# амплитуды [num_maxima]
    'frequencies': np.array(float32) # частоты Hz [num_maxima]
}
```

### 8.3 ComplexToMagROCm

**Модуль**: `gpuworklib` (если доступен в сборке)

```python
class ComplexToMagROCm:
    def __init__(self, ctx: ROCmGPUContext): ...

    def process_magnitude(
        self,
        gpu_data,            # GPU buffer (из ROCm)
        beam_count: int,
        n_point: int,
        norm_coeff: float = 1.0   # 0=без норм, -1=÷n, >0=умножить
    ) -> np.ndarray:              # float32 [beam_count × n_point]
        ...
```

---

## 9. Примеры цепочек вызовов

### 9.1 Полный pipeline: IQ → пики

```cpp
// C++: сигнал → FFT → поиск пика

// 1. FFT
fft_processor::FFTProcessorROCm fft(backend);
fft_processor::FFTProcessorParams fft_params;
fft_params.beam_count = 5;
fft_params.n_point    = 1000;
fft_params.sample_rate = 10000.0f;

auto fft_results = fft.ProcessComplex(raw_signal, fft_params);

// 2. Поиск всех пиков (без повторного FFT — AllMaxima)
antenna_fft::InputData<void*> fft_input;
fft_input.antenna_count = 5;
fft_input.n_point = fft.GetNFFT();  // ВАЖНО: n_point = nFFT!
fft_input.data = fft_results_gpu;   // буфер на GPU
fft_input.sample_rate = 10000.0f;

antenna_fft::SpectrumMaximaFinder finder(backend);
auto peaks = finder.AllMaxima(fft_input, antenna_fft::OutputDestination::CPU);
```

### 9.2 Python: FFT → find_all_maxima

```python
ctx = gpuworklib.ROCmGPUContext(0)
fft    = gpuworklib.FFTProcessorROCm(ctx)
finder = gpuworklib.SpectrumMaximaFinder(ctx)

# Многолучевой сигнал: 5 лучей × 1024 точки
signals = np.zeros((5, 1024), dtype=np.complex64)
for i in range(5):
    t = np.arange(1024) / 1000.0
    signals[i] = np.exp(2j * np.pi * (50 + i * 50) * t).astype(np.complex64)

spectra = fft.process_complex(signals, 1000.0)  # → shape [5, nFFT]
results = finder.find_all_maxima(spectra, sample_rate=1000.0)

for i, beam in enumerate(results):
    print(f"Beam {i}: {beam['num_maxima']} peaks at {beam['frequencies']} Hz")
```

### 9.3 ComplexToMagPhaseROCm в pipeline статистики

```cpp
// C++: ComplexToMagPhaseROCm.ProcessMagnitudeToBuffer → statistics

fft_processor::ComplexToMagPhaseROCm c2m(backend);
fft_processor::MagPhaseParams params;
params.beam_count = beam_count;
params.n_point    = n_fft;
params.norm_coeff = -1.0f;  // нормировать на n_point

// Результат FFT уже на GPU в fft_output_gpu
// Целевой буфер для статистики уже выделен в mag_buffer_gpu
c2m.ProcessMagnitudeToBuffer(fft_output_gpu, mag_buffer_gpu, params);
// Теперь mag_buffer_gpu содержит |FFT[k]| / n_point
// → передать в statistics::compute_statistics()
```

### 9.4 ROCm профилирование FFT по стадиям

```cpp
fft_processor::ROCmProfEvents events;
auto results = fft.ProcessComplex(data, params, &events);

for (const auto& [stage, ev] : events) {
    double ms = (ev.end_ns - ev.start_ns) / 1e6;
    printf("  %s: %.3f ms\n", stage, ms);
    // stages: Upload, PadData, FFT, MagPhase, Download
}
```

### 9.5 OutputDestination::GPU и освобождение памяти

```cpp
// AllMaxima с оставлением данных на GPU
auto result = finder.FindAllMaxima(input, antenna_fft::OutputDestination::GPU);

// result.beams — пусто (данные на GPU)
// result.gpu_maxima, result.gpu_counts — CALLER OWNS!

// ... использовать gpu_maxima ...

// Обязательно освободить!
if (result.gpu_maxima) hipFree(result.gpu_maxima);
if (result.gpu_counts) hipFree(result.gpu_counts);
```

---

## 7.5 Op-классы — Ref03 Layer 5

Op-классы не являются публичным API — используются внутри процессорных классов. Документируются для понимания архитектуры.

**Заголовки**: `include/operations/*.hpp` | **Требует**: `ENABLE_ROCM=1`

Все Op-классы наследуют `drv_gpu_lib::GpuKernelOp`:
- `Initialize(GpuContext*)` — привязка к контексту
- `kernel(name)` → `hipFunction_t` — получить функцию по имени
- `stream()` → `hipStream_t` — текущий поток из GpuContext

| Op-класс | Namespace | Kernel | Сигнатура Execute |
|----------|-----------|--------|-------------------|
| `PadDataOp` | `fft_processor` | `pad_data` | `Execute(in, out, beam_count, n_point, nFFT)` |
| `MagPhaseOp` | `fft_processor` | `complex_to_mag_phase` | `Execute(in, out, total)` |
| `MagnitudeOp` | `fft_processor` | `complex_to_magnitude` | `Execute(in, out, total, inv_n)` |
| `SpectrumPadOp` | `antenna_fft` | `pad_data` | `Execute(in, out, beam_count, n_point, nFFT, beam_offset=0)` |
| `ComputeMagnitudesOp` | `antenna_fft` | `compute_magnitudes` | `Execute(fft_out, mag_buf, total)` |
| `SpectrumPostOp` | `antenna_fft` | `post_kernel` | `Execute(mag, fft, maxima, beam_count, nFFT, search_range)` |

---

*Обновлено: 2026-03-28*

*См. также: [Full.md](Full.md) | [Quick.md](Quick.md)*
