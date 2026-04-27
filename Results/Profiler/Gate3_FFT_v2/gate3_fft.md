# GPU Profiling Report — 2026-04-21 10:13:28

## GPU 0: AMD Radeon RX 9070

### Pipeline Breakdown — spectrum/fft

| Event | Kind | Avg ms | % | Count |
|-------|------|-------:|--:|------:|
| Download | copy | 0.523 | 84.3% | 20 |
| Upload | copy | 0.073 | 11.7% | 20 |
| Pad | kernel | 0.014 | 2.2% | 20 |
| FFT | kernel | 0.011 | 1.7% | 20 |
| **TOTAL** | | **0.620** | **100.0%** |  kernel: 3.9%, copy: 96.1%, barrier: 0.0% |

### Statistical Summary — spectrum/fft

| Event | N | Avg | Med | p95 | StdDev | Min | Max |
|-------|--:|----:|----:|----:|-------:|----:|----:|
| Download | 20 | 0.523 | 0.518 | 0.548 | 0.014 | 0.499 | 0.559 |
| FFT | 20 | 0.011 | 0.011 | 0.011 | 0.000 | 0.010 | 0.011 |
| Pad | 20 | 0.014 | 0.013 | 0.014 | 0.001 | 0.013 | 0.018 |
| Upload | 20 | 0.073 | 0.072 | 0.078 | 0.009 | 0.056 | 0.106 |

