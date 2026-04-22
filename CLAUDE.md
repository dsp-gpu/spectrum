# 🤖 CLAUDE — `spectrum`

> FFT / IFFT, оконные функции, поиск максимума, фильтры, lch_farrow.
> Зависит от: `core` + `hipFFT`. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

| Класс | Что делает |
|-------|-----------|
| `FFTProcessor` | FFT/IFFT прямое/обратное через hipFFT, режимы MagPhase / Complex |
| `SpectrumMaximaFinder` | Поиск пиков спектра (top-K, threshold) |
| `FirFilter` / `IirFilter` | Фильтрация во временной области |
| `LchFarrowResampler` | Polyphase + Farrow для fractional resampling |
| `WindowFunctions` | Hann, Hamming, Blackman, Kaiser, ... |

**Каноничное имя репо** — `spectrum` (не `fft_func`, не `fft_processor` — spectrum более общее).

## 📁 Структура

```
spectrum/
├── include/dsp/spectrum/
│   ├── spectrum_processor.hpp    # facade
│   ├── gpu_context.hpp
│   ├── i_gpu_operation.hpp       # (шарится через core или локально)
│   ├── operations/               # FFTOp, WindowOp, MaximaOp, ...
│   └── strategies/
├── src/
├── kernels/rocm/                 # window.hip, find_maxima.hip, ...
├── tests/
└── python/dsp_spectrum_module.cpp
```

## ⚠️ Специфика

- **hipFFT** — единственный путь для FFT (не clFFT, не cuFFT). Включать через `find_package(hipfft REQUIRED)`.
- **Batch FFT**: предпочтительно для многих пучков — один план hipFFT, много сигналов.
- **Workspace управляется самим hipFFT** — не пытаться выделять вручную.
- **IFFT нормализация**: hipFFT не делит на N — делать это в kernel scale или документировать.

## 🚫 Запреты

- Не использовать `clFFT` / `cuFFT` — только `hipFFT`.
- Не импортировать классы `radar` или `strategies` — иерархия: `spectrum` ниже.
- Не создавать новый `FFTProcessorV2` — расширять существующий через OCP.

## 🔗 Правила (path-scoped автоматически)

- `09-rocm-only.md` — hipFFT обязателен
- `05-architecture-ref03.md` — 6-слойная модель
- `14-cpp-style.md` + `15-cpp-testing.md` — стиль / тесты
- `11-python-bindings.md` — pybind11
