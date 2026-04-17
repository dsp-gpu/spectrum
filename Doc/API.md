# Spectrum — API (объединённый: FFT + Filters + LCH Farrow)

> Репо `spectrum` объединяет три компонента из DSP-GPU: **fft_func** (БПФ-пайплайн), **filters** (FIR/IIR/адаптивные) и **lch_farrow** (LCH + Farrow дробная задержка).

---

## Компонент: FFT Pipeline


> Полный справочник по классам, методам и типам модуля `fft_func`

---

#### Содержание

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

#### 1. Типы данных — FFTProcessorROCm

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

#### 2. FFTProcessorROCm

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

#### 3. Типы данных — ComplexToMagPhaseROCm

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

#### 4. ComplexToMagPhaseROCm

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

#### 5. Типы данных — SpectrumMaximaFinder

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

#### 6. SpectrumMaximaFinder

**Заголовок**: `include/interface/spectrum_maxima_types.h` (агрегатор) | **Namespace**: `antenna_fft`

```cpp
class SpectrumMaximaFinder {
public:
  // ── Конструктор ──────────────────────────────────────────────────────────

  explicit SpectrumMaximaFinderROCm(drv_gpu_lib::IBackend* backend);
  // Initialize() вызывается автоматически при первом Process

  // DEPRECATED:
  [[deprecated]] SpectrumMaximaFinderROCm(SpectrumParams params,
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

#### 7. SpectrumProcessorFactory

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

#### 8. Python API

##### 8.1 FFTProcessorROCm

**Модуль**: `dsp_spectrum` | **Биндинг**: `python/py_fft_processor_rocm.hpp`

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

##### 8.2 SpectrumMaximaFinder

**Модуль**: `dsp_spectrum` | **Биндинг**: `python/gpu_worklib_bindings.cpp`

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

### Формат dict одного луча:
{
    'num_maxima':  int,             # количество найденных максимумов
    'positions':   np.array(uint32), # бины [num_maxima]
    'magnitudes':  np.array(float32),# амплитуды [num_maxima]
    'frequencies': np.array(float32) # частоты Hz [num_maxima]
}
```

##### 8.3 ComplexToMagROCm

**Модуль**: `dsp_spectrum` (если доступен в сборке)

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

#### 9. Примеры цепочек вызовов

##### 9.1 Полный pipeline: IQ → пики

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

##### 9.2 Python: FFT → find_all_maxima

```python
ctx = dsp_spectrum.ROCmGPUContext(0)
fft    = dsp_spectrum.FFTProcessorROCm(ctx)
finder = dsp_spectrum.SpectrumMaximaFinderROCm(ctx)

### Многолучевой сигнал: 5 лучей × 1024 точки
signals = np.zeros((5, 1024), dtype=np.complex64)
for i in range(5):
    t = np.arange(1024) / 1000.0
    signals[i] = np.exp(2j * np.pi * (50 + i * 50) * t).astype(np.complex64)

spectra = fft.process_complex(signals, 1000.0)  # → shape [5, nFFT]
results = finder.find_all_maxima(spectra, sample_rate=1000.0)

for i, beam in enumerate(results):
    print(f"Beam {i}: {beam['num_maxima']} peaks at {beam['frequencies']} Hz")
```

##### 9.3 ComplexToMagPhaseROCm в pipeline статистики

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

##### 9.4 ROCm профилирование FFT по стадиям

```cpp
fft_processor::ROCmProfEvents events;
auto results = fft.ProcessComplex(data, params, &events);

for (const auto& [stage, ev] : events) {
    double ms = (ev.end_ns - ev.start_ns) / 1e6;
    printf("  %s: %.3f ms\n", stage, ms);
    // stages: Upload, PadData, FFT, MagPhase, Download
}
```

##### 9.5 OutputDestination::GPU и освобождение памяти

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

#### 7.5 Op-классы — Ref03 Layer 5

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

---

## Компонент: Filters


> Полные сигнатуры всех публичных классов и методов модуля `filters`

**Namespace**: `filters`
**Каталог**: `spectrum/include/filters/`

---

#### Содержание

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

#### 1. FirFilter (OpenCL)

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

#### 2. IirFilter (OpenCL)

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

#### 3. FirFilterROCm

**Файл**: `include/filters/fir_filter_rocm.hpp`
**Backend**: ROCm/HIP (`void*` device pointers)
**Условие**: `ENABLE_ROCM=1` (Linux + AMD GPU). На Windows — compile-only stub (все методы `throw std::runtime_error`).

```cpp
namespace filters {

###if ENABLE_ROCM

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

###endif  // ENABLE_ROCM
}  // namespace filters
```

---

#### 4. IirFilterROCm

**Файл**: `include/filters/iir_filter_rocm.hpp`
**Backend**: ROCm/HIP
**Условие**: `ENABLE_ROCM=1`. На Windows — compile-only stub.

```cpp
namespace filters {

###if ENABLE_ROCM

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

###endif  // ENABLE_ROCM
}  // namespace filters
```

---

#### 5. MovingAverageFilterROCm

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

###if ENABLE_ROCM

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

###endif  // ENABLE_ROCM
}  // namespace filters
```

---

#### 6. KalmanFilterROCm

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

###if ENABLE_ROCM

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

###endif  // ENABLE_ROCM
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

#### 7. KaufmanFilterROCm

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

###if ENABLE_ROCM

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

###endif  // ENABLE_ROCM
}  // namespace filters
```

---

#### 8. Типы данных

##### BiquadSection

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

##### FilterConfig (JSON loading)

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

##### InputData\<T\>

```cpp
// core/interface/input_data.hpp
template <typename T>
struct InputData {
  T      data;   // cl_mem или void* — CALLER OWNS (clReleaseMemObject / hipFree)
  size_t size;   // байт
};
```

##### ProfEvents / ROCmProfEvents

```cpp
// OpenCL:
using ProfEvents = std::vector<std::pair<const char*, cl_event>>;

// ROCm:
using ROCmProfEvents = std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

---

#### 9. Python API (OpenCL)

**Файл биндингов**: `python/py_filters.hpp`
**Статус**: ✅ Зарегистрировано в `python/gpu_worklib_bindings.cpp`

##### FirFilter

```python
import dsp_spectrum as gw
import scipy.signal as sig
import numpy as np

ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilter(ctx)

### Конфигурация
fir.set_coefficients(sig.firwin(64, 0.1).tolist())  # list[float]
fir.load_config("path/to/lowpass.json")

### Обработка
### input: numpy.ndarray complex64, shape (points,) или (channels, points)
result = fir.process(input)   # numpy.ndarray complex64, same shape

### Свойства (readonly)
fir.num_taps       # int
fir.coefficients   # list[float]
repr(fir)          # "<FirFilter num_taps=64>"
```

##### IirFilter

```python
iir = gw.IirFilter(ctx)

### scipy SOS → sections
sos = sig.butter(2, 0.1, output='sos')
sections = [
    {'b0': float(r[0]), 'b1': float(r[1]), 'b2': float(r[2]),
     'a1': float(r[4]), 'a2': float(r[5])}   # r[3] (a0=1) пропускаем
    for r in sos
]
iir.set_sections(sections)                   # list[dict]
iir.load_config("path/to/butterworth.json")

### Обработка
result = iir.process(signal)   # numpy.ndarray complex64

### Свойства
iir.num_sections   # int
iir.sections       # list[dict] с ключами b0,b1,b2,a1,a2
repr(iir)          # "<IirFilter num_sections=1>"
```

---

#### 10. Python API (ROCm)

**Файл биндингов**: `python/py_filters_rocm.hpp`

| Класс | Статус биндингов |
|-------|-----------------|
| `FirFilterROCm` | ✅ Зарегистрирован |
| `IirFilterROCm` | ✅ Зарегистрирован |
| `MovingAverageFilterROCm` | ❌ НЕ зарегистрирован |
| `KalmanFilterROCm` | ❌ НЕ зарегистрирован |
| `KaufmanFilterROCm` | ❌ НЕ зарегистрирован |

##### FirFilterROCm (✅)

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

##### IirFilterROCm (✅)

```python
iir = gw.IirFilterROCm(ctx)
iir.set_sections([{'b0': 0.02, 'b1': 0.04, 'b2': 0.02, 'a1': -1.56, 'a2': 0.64}])

result = iir.process(signal)   # numpy.ndarray complex64

iir.num_sections   # int
iir.sections       # list[dict]
repr(iir)          # "<IirFilterROCm num_sections=1>"
```

##### MovingAverageFilterROCm (❌ — запланировано)

```python
### ВНИМАНИЕ: биндинги не зарегистрированы в gpu_worklib_bindings.cpp!
### Ожидаемый API после регистрации (по test_moving_average_rocm.py):
ma = gw.MovingAverageFilterROCm(ctx)
ma.set_params("EMA", 10)   # type: "SMA"/"EMA"/"MMA"/"DEMA"/"TEMA", window_size
result = ma.process(data)  # numpy.ndarray complex64
ma.is_ready()              # bool
ma.get_window_size()       # int
ma.get_type()              # str
```

##### KalmanFilterROCm (❌ — запланировано)

```python
### ВНИМАНИЕ: биндинги не зарегистрированы!
### Ожидаемый API (по test_kalman_rocm.py):
kalman = gw.KalmanFilterROCm(ctx)
kalman.set_params(Q=0.1, R=25.0, x0=0.0, P0=25.0)
result = kalman.process(data)   # numpy.ndarray complex64
```

##### KaufmanFilterROCm (❌ — запланировано)

```python
### ВНИМАНИЕ: биндинги не зарегистрированы!
### Ожидаемый API (по test_kaufman_rocm.py):
kauf = gw.KaufmanFilterROCm(ctx)
kauf.set_params(er_period=10, fast_period=2, slow_period=30)
result = kauf.process(data)   # numpy.ndarray complex64
```

---

#### 11. Цепочки вызовов

##### OpenCL FIR — полная цепочка (C++)

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

##### ROCm FIR из CPU — минимальный вариант (C++)

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

##### Python scipy → FirFilter → validate (Python)

```python
import dsp_spectrum as gw
import scipy.signal as sig
import numpy as np

ctx = gw.ROCmGPUContext(0)
fir = gw.FirFilter(ctx)

taps = sig.firwin(64, 0.1).astype(np.float32)
fir.set_coefficients(taps.tolist())

### Входной сигнал: 8 каналов × 4096 точек
signal = (np.random.randn(8, 4096) + 1j * np.random.randn(8, 4096)).astype(np.complex64)

gpu_result = fir.process(signal)   # (8, 4096) complex64

### Validate vs scipy
ref = np.array([sig.lfilter(taps, [1.0], signal[ch]) for ch in range(8)])
max_err = np.max(np.abs(gpu_result - ref))
print(f"Max error GPU vs scipy: {max_err:.2e}")  # ожидается < 1e-2
```

##### ROCm FIR → Kalman pipeline (C++, только Linux + AMD)

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

#### Ссылки

- [Full.md](Full.md) — математика, pipeline, C4 диаграммы, детальный rationale тестов
- [Quick.md](Quick.md) — шпаргалка с концепцией и быстрым стартом
- [SciPy firwin](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html)
- [SciPy butter SOS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)
- [Digital biquad DFII-T](https://en.wikipedia.org/wiki/Digital_biquad_filter#Direct_form_2_transposed)

---

*Обновлено: 2026-03-09*

---

## Компонент: LCH Farrow


> Дробная задержка Lagrange 48×5 на GPU. OpenCL + ROCm.

**Namespace**: `lch_farrow`
**Каталог**: `spectrum/`

---

#### Подключение

```cpp
// C++ — OpenCL
#include <spectrum/lch_farrow.hpp>

// C++ — ROCm (только ENABLE_ROCM=1, Linux + AMD GPU)
#include <spectrum/lch_farrow_rocm.hpp>
```

```python
import dsp_spectrum

proc_ocl  = dsp_spectrum.LchFarrow(ctx)       # ctx: GPUContext
proc_rocm = dsp_spectrum.LchFarrowROCm(ctx)   # ctx: ROCmGPUContext
```

---

#### Типы данных

##### ProfEvents (OpenCL)

```cpp
using lch_farrow::ProfEvents =
    std::vector<std::pair<const char*, cl_event>>;
```

Передать в `Process()` для сбора `cl_event`. `nullptr` = production (нулевой overhead).
Стейджи: `"Upload_delay"`, `"Kernel"`.

##### ROCmProfEvents

```cpp
using lch_farrow::ROCmProfEvents =
    std::vector<std::pair<const char*, drv_gpu_lib::ROCmProfilingData>>;
```

Стейджи `ProcessFromCPU`: `"Upload_input"`, `"Upload_delay"`, `"Kernel"`.

##### InputData

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

#### Класс LchFarrow (OpenCL)

##### Конструктор

```cpp
explicit LchFarrow(drv_gpu_lib::IBackend* backend);
```

Инициализирует OpenCL context/queue/device, загружает встроенную матрицу, компилирует kernel (`-cl-fast-relaxed-math`), загружает матрицу в GPU. Бросает `std::runtime_error` если backend null.

##### Move-семантика

```cpp
LchFarrow(LchFarrow&&) noexcept;
LchFarrow& operator=(LchFarrow&&) noexcept;
// Copy удалён
```

##### Setters

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

##### Process (GPU→GPU)

```cpp
drv_gpu_lib::InputData<cl_mem> Process(
    cl_mem input_buf,
    uint32_t antennas,
    uint32_t points,
    ProfEvents* prof_events = nullptr);
```

`input_buf` — не освобождается. Вызывает `clFinish`. Возвращает `InputData<cl_mem>` — **caller обязан** `clReleaseMemObject(result.data)`.

Бросает `std::runtime_error` при ошибке OpenCL; `std::invalid_argument` если `delay_us_.size() != antennas`.

##### ProcessCpu (CPU reference)

```cpp
std::vector<std::vector<std::complex<float>>> ProcessCpu(
    const std::vector<std::vector<std::complex<float>>>& input,
    uint32_t antennas, uint32_t points);
```

CPU-реализация того же алгоритма для верификации. `input[antenna][sample]`.

##### Getters

```cpp
const std::vector<float>& GetDelays() const;
float GetSampleRate() const;
```

---

#### Класс LchFarrowROCm (ENABLE_ROCM=1)

На Windows без `ENABLE_ROCM` — stub, все методы бросают `std::runtime_error("LchFarrowROCm: ROCm not enabled")`.

##### Конструктор

```cpp
explicit LchFarrowROCm(drv_gpu_lib::IBackend* backend);
```

Получает `hipStream_t`, компилирует через hiprtc, использует HSACO disk cache. Холодный старт ~100-200 мс, горячий ~1-5 мс.

##### Setters

Идентичны `LchFarrow` (SetDelays, SetSampleRate, SetNoise, LoadMatrix).

##### Process (GPU→GPU)

```cpp
drv_gpu_lib::InputData<void*> Process(
    void* input_ptr, uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events = nullptr);
```

Принимает HIP device pointer. Стейджи: `Upload_delay`, `Kernel`. **Caller** — `hipFree(result.data)`.

##### ProcessFromCPU (CPU→GPU)

```cpp
drv_gpu_lib::InputData<void*> ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events = nullptr);
```

Плоский вектор `data[antennas * points]`. Загружает на GPU (временный буфер, освобождается внутри), обрабатывает. Стейджи: `Upload_input`, `Upload_delay`, `Kernel`. **Caller** — `hipFree(result.data)`.

##### ProcessCpu, Getters

Идентичны `LchFarrow`.

---

#### Сравнение OpenCL vs ROCm

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

#### Python API

##### LchFarrow (OpenCL)

```python
proc = dsp_spectrum.LchFarrow(ctx)           # ctx: GPUContext
proc.set_delays([0.0, 2.7, 5.0])           # list[float], мкс
proc.set_sample_rate(1e6)                  # float, Гц
proc.set_noise(0.1, norm_val=0.7071, noise_seed=42)
proc.load_matrix("lagrange_matrix_48x5.json")

result = proc.process(input)               # ndarray complex64
### input shape (points,)           → result shape (points,)
### input shape (antennas, points)  → result shape (antennas, points)

proc.sample_rate    # read-only float
proc.delays         # read-only list[float]
repr(proc)          # "<LchFarrow sample_rate=1000000>"
```

##### LchFarrowROCm

```python
ctx  = dsp_spectrum.ROCmGPUContext(0)
proc = dsp_spectrum.LchFarrowROCm(ctx)
proc.set_sample_rate(1e6)
proc.set_delays([0.0, 2.7, 5.0])
result = proc.process(input)   # идентичный API
```

Внутри вызывает `ProcessFromCPU` (numpy → flat vector → GPU).

---

#### Примеры

##### C++ OpenCL — production pipeline

```cpp
#include <core/backends/opencl/opencl_backend.hpp>
#include <spectrum/lch_farrow.hpp>

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

##### C++ ROCm — production pipeline

```cpp
#include <core/backends/rocm/rocm_backend.hpp>
#include <spectrum/lch_farrow_rocm.hpp>

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

##### C++ OpenCL — benchmark с профилированием

```cpp
lch_farrow::ProfEvents events;
auto result = proc.Process(input_buf, antennas, points, &events);
clReleaseMemObject(result.data);

for (auto& [name, ev] : events) {
    RecordEvent(name, ev);  // GpuBenchmarkBase
}
// events: {"Upload_delay", cl_event}, {"Kernel", cl_event}
```

##### Python — multi-channel delay

```python
import numpy as np
import dsp_spectrum

ctx = dsp_spectrum.ROCmGPUContext(0)

fs = 1e6
N  = 4096
t  = np.arange(N) / fs
cw = np.exp(1j * 2 * np.pi * 50000 * t).astype(np.complex64)
signal_4ch = np.tile(cw, (4, 1))   # (4, 4096)

proc = dsp_spectrum.LchFarrow(ctx)
proc.set_sample_rate(fs)
proc.set_delays([0.0, 1.5, 3.0, 5.0])   # мкс

delayed = proc.process(signal_4ch)      # (4, 4096)
print(f"sample_rate: {proc.sample_rate:.0f}")
print(f"delays: {list(proc.delays)}")
```

---

#### Цепочки вызовов

##### OpenCL — минимальный

```
LchFarrow(backend)
  → SetSampleRate(fs)
  → SetDelays(delays_us)
  → Process(input_cl_mem, antennas, points)
  → [caller] clReleaseMemObject(result.data)
```

##### ROCm — через CPU

```
LchFarrowROCm(backend)
  → SetSampleRate(fs)
  → SetDelays(delays_us)
  → ProcessFromCPU(flat_vector, antennas, points)
  → [caller] hipFree(result.data)
```

##### С профилированием

```
ProfEvents events / ROCmProfEvents events
  → Process(..., &events) / ProcessFromCPU(..., &events)
  → for (name, ev) : events → RecordEvent(name, ev)
  → Report() → PrintReport() + ExportMarkdown() + ExportJSON()
```

---

#### Нюансы API

1. **`SetDelays` перед `Process`** — если не вызван, задержки = 0 (заполняются автоматически).
2. **`result.data` — caller owns** — утечка памяти если не освободить.
3. **`input_buf` не трогается** — Process() не вызывает release.
4. **`matrix_buf_` persistent** — создаётся в конструкторе, пересоздаётся только при `LoadMatrix()`.
5. **`delay_buf_` persistent** — увеличивается при росте числа антенн, не уменьшается.
6. **ROCm HSACO кэш** — файлы в `spectrum/kernels/`. Удалить при смене GPU архитектуры.
7. **`row` по frac читаемой позиции, не по μ задержки** — частая ошибка в ранних спецификациях.

---

#### Ссылки

- [Full.md](Full.md) — математика, pipeline, C4, детальный разбор тестов
- [Quick.md](Quick.md) — краткий справочник
- [Doc/Python/lch_farrow_api.md](../../Python/lch_farrow_api.md) — Python API подробно

---

*Обновлено: 2026-03-09*

---

