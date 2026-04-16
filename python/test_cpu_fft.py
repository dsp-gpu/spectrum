#!/usr/bin/env python3
"""
test_cpu_fft.py — standalone тест CpuFFT Python API (pocketfft)

Запуск:
  cd spectrum/build
  python3 ../python/test_cpu_fft.py

НЕ использует pytest (запрещён в CLAUDE.md).
"""

import sys
import os
import numpy as np

# Добавить путь к собранному .so
build_dir = os.path.dirname(os.path.abspath(__file__))
for candidate in ['python', '.', '../build/python', 'build/python']:
    path = os.path.join(build_dir, candidate)
    if os.path.isdir(path):
        sys.path.insert(0, path)
# Также проверяем CWD/python
sys.path.insert(0, os.path.join(os.getcwd(), 'python'))

import dsp_spectrum


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed += 1
            self.errors.append(name)
            print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))

    def summary(self):
        total = self.passed + self.failed
        status = "ALL PASSED" if self.failed == 0 else "FAILED"
        print(f"\nResults: {self.passed}/{total} passed  [{status}]")
        if self.errors:
            print(f"Failed: {', '.join(self.errors)}")
        return self.failed == 0


def test_c2c_forward(tr: TestRunner):
    """C2C forward: complex sinusoid → peak at expected bin"""
    N, fs, freq = 1024, 1000.0, 100.0
    t = np.arange(N) / fs
    signal = np.exp(2j * np.pi * freq * t).astype(np.complex64)

    spectrum = dsp_spectrum.cpu_fft_c2c(signal)
    mag = np.abs(spectrum)
    peak_bin = int(np.argmax(mag[:N // 2]))
    peak_freq = peak_bin * fs / N

    tr.check("c2c_forward_shape", spectrum.shape == (N,), f"shape={spectrum.shape}")
    tr.check("c2c_forward_peak", abs(peak_freq - freq) < fs / N,
             f"peak={peak_freq:.1f} Hz, expected={freq:.1f} Hz")


def test_c2c_roundtrip(tr: TestRunner):
    """C2C forward → inverse = original (within float32 precision)"""
    N = 512
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)

    spectrum = dsp_spectrum.cpu_fft_c2c(signal)
    recovered = dsp_spectrum.cpu_ifft_c2c(spectrum)
    err = float(np.max(np.abs(signal - recovered)))

    tr.check("c2c_roundtrip", err < 1e-5, f"max_err={err:.2e}")


def test_r2c_half(tr: TestRunner):
    """R2C: real signal → half spectrum (N/2+1 bins)"""
    N, fs, freq = 2048, 4000.0, 500.0
    t = np.arange(N) / fs
    signal = np.sin(2 * np.pi * freq * t).astype(np.float32)

    half = dsp_spectrum.cpu_fft_r2c(signal)
    expected_size = N // 2 + 1

    tr.check("r2c_half_shape", half.shape == (expected_size,),
             f"shape={half.shape}, expected=({expected_size},)")

    peak_bin = int(np.argmax(np.abs(half)))
    peak_freq = peak_bin * fs / N
    tr.check("r2c_half_peak", abs(peak_freq - freq) < fs / N,
             f"peak={peak_freq:.1f} Hz, expected={freq:.1f} Hz")


def test_r2c_full(tr: TestRunner):
    """R2C full: real signal → full spectrum (N bins, hermitian mirror)"""
    N = 1024
    signal = np.sin(2 * np.pi * 100 * np.arange(N) / 1000).astype(np.float32)

    full = dsp_spectrum.cpu_fft_r2c_full(signal)
    tr.check("r2c_full_shape", full.shape == (N,), f"shape={full.shape}")

    # Hermitian: X[k] = conj(X[N-k])
    max_err = 0.0
    for k in range(1, N // 2):
        err = abs(full[k] - np.conj(full[N - k]))
        max_err = max(max_err, err)
    tr.check("r2c_full_hermitian", max_err < 1e-5, f"max_err={max_err:.2e}")


def test_magnitude(tr: TestRunner):
    """magnitude: |X| and |X|² from complex spectrum"""
    spectrum = np.array([3 + 4j, 1 + 0j, 0 + 1j], dtype=np.complex64)

    mag = dsp_spectrum.magnitude(spectrum)
    mag2 = dsp_spectrum.magnitude(spectrum, squared=True)

    tr.check("magnitude_abs", abs(mag[0] - 5.0) < 1e-5, f"|3+4j|={mag[0]:.4f}")
    tr.check("magnitude_sq", abs(mag2[0] - 25.0) < 1e-3, f"|3+4j|²={mag2[0]:.4f}")
    tr.check("magnitude_unit", abs(mag[1] - 1.0) < 1e-5, f"|1+0j|={mag[1]:.4f}")


def test_r2c_vs_numpy(tr: TestRunner):
    """R2C vs numpy.fft.rfft — cross-validation"""
    N = 4096
    signal = np.random.randn(N).astype(np.float32)

    our_half = dsp_spectrum.cpu_fft_r2c(signal)
    np_half = np.fft.rfft(signal).astype(np.complex64)

    err = float(np.max(np.abs(our_half - np_half)))
    tr.check("r2c_vs_numpy", err < 1e-2, f"max_err={err:.2e}")


def test_c2c_vs_numpy(tr: TestRunner):
    """C2C vs numpy.fft.fft — cross-validation"""
    N = 2048
    signal = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)

    our = dsp_spectrum.cpu_fft_c2c(signal)
    np_ref = np.fft.fft(signal).astype(np.complex64)

    err = float(np.max(np.abs(our - np_ref)))
    peak = float(np.max(np.abs(np_ref)))
    rel_err = err / peak if peak > 0 else err
    tr.check("c2c_vs_numpy", rel_err < 1e-5, f"rel_err={rel_err:.2e}")


def main():
    print("=" * 60)
    print("  dsp_spectrum — CpuFFT Python API Tests")
    print("=" * 60)
    print(f"  Module: {dsp_spectrum.__file__}")
    print(f"  API: {[x for x in dir(dsp_spectrum) if not x.startswith('_')]}")
    print()

    tr = TestRunner()

    test_c2c_forward(tr)
    test_c2c_roundtrip(tr)
    test_r2c_half(tr)
    test_r2c_full(tr)
    test_magnitude(tr)
    test_r2c_vs_numpy(tr)
    test_c2c_vs_numpy(tr)

    ok = tr.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
