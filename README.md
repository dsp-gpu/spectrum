# DspSpectrum

> Part of [dsp-gpu](https://github.com/dsp-gpu) organization.

Spectral DSP library: FFT/IFFT processor, FIR/IIR filters, LCH-Farrow interpolation.

## Dependencies

- DspCore (core)
- ROCm: hipFFT

## Build

```bash
cmake -S . -B build --preset local-dev
cmake --build build
```

## Contents

- `fft_func/` — FFT/IFFT processor (hipFFT backend)
- `filters/`  — FIR, IIR GPU filters
- `lch_farrow/` — LCH-Farrow fractional delay interpolation
