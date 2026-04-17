# 📚 Research: GPU FIR/IIR Filter Implementation (OpenCL)

> **Источник**: AI Research Agent (2026-02-18)
> **Статус**: ✅ Завершено, используется в TASK-007

---

## 1. FIR Filter — Выбор алгоритма

| Метод | Когда использовать | GPU-пригодность |
|-------|-------------------|--------------------|
| Direct-form | Короткие фильтры (N ≤ 128 отводов) | Отличная — полностью параллельно по семплам |
| Overlap-Save (OLS) | Длинные фильтры (N > 256), блочная обработка | Отличная — FFT-based, **переиспользует FFTProcessor** |
| Overlap-Add (OLA) | Аналогично OLS, чуть сложнее | Отличная — тоже FFT-based |

**Ключевое**: для длинных фильтров — переиспользуем существующий `FFTProcessor` из DSP-GPU!

### Лимиты `__constant` памяти
- Типично 64 KB → `float`: **16 384 коэф. максимум**
- Если > 64 KB: использовать `__global const __restrict` (кэшируется аппаратно)

---

## 2. FIR Kernels

### 2.1 Direct-Form FIR (float2 complex) — основной

```opencl
__kernel void fir_direct_float2(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  coeffs,
    const int num_taps,
    const int signal_length,
    const int channel_offset)
{
    int idx = get_global_id(0);
    if (idx >= signal_length) return;

    float2 acc = (float2)(0.0f, 0.0f);
    int base = idx + channel_offset;

    for (int k = 0; k < num_taps; k++) {
        int src_idx = base - k;
        if (src_idx >= 0) {
            float2 x = input[src_idx];
            float  h = coeffs[k];
            acc.x += x.x * h;
            acc.y += x.y * h;
        }
    }
    output[idx + channel_offset] = acc;
}
```

### 2.2 Tiled FIR с Local Memory (уменьшает bandwidth)

```opencl
#define WG_SIZE 256

__kernel __attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void fir_tiled_float2(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  coeffs,
    const int num_taps,
    const int signal_length)
{
    __local float2 lmem[WG_SIZE + 512];  // WG_SIZE + max_taps - 1

    int gid  = get_global_id(0);
    int lid  = get_local_id(0);
    int wg   = get_group_id(0);
    int base = wg * WG_SIZE;

    int load_idx = base + lid - (num_taps - 1);
    lmem[lid] = (load_idx >= 0) ? input[load_idx] : (float2)(0.0f, 0.0f);

    if (lid < num_taps - 1) {
        int extra = base + WG_SIZE + lid - (num_taps - 1);
        lmem[WG_SIZE + lid] = (extra >= 0 && extra < signal_length)
                              ? input[extra] : (float2)(0.0f, 0.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < signal_length) {
        float2 acc = (float2)(0.0f, 0.0f);
        int lbase = lid + (num_taps - 1);
        for (int k = 0; k < num_taps; k++) {
            float2 x = lmem[lbase - k];
            acc.x += x.x * coeffs[k];
            acc.y += x.y * coeffs[k];
        }
        output[gid] = acc;
    }
}
```

### 2.3 Multi-Channel FIR с 2D NDRange

```opencl
// NDRange: (signal_length, num_channels) — WorkGroup: (256, 1)
__kernel void fir_multichannel_float2(
    __global const float2* restrict input,
    __global       float2* restrict output,
    __constant     float*  coeffs,
    const int num_taps,
    const int N,
    const int num_channels)
{
    int sample  = get_global_id(0);
    int channel = get_global_id(1);
    if (sample >= N || channel >= num_channels) return;

    int base = channel * N;
    float2 acc = (float2)(0.0f, 0.0f);

    for (int k = 0; k < num_taps; k++) {
        int src = sample - k + base;
        float2 x = (src >= base) ? input[src] : (float2)(0.0f, 0.0f);
        acc.x += x.x * coeffs[k];
        acc.y += x.y * coeffs[k];
    }
    output[base + sample] = acc;
}
```

### 2.4 Overlap-Save (FFT-based) — для длинных фильтров

```opencl
// Поэлементное умножение спектров X*H
__kernel void freq_multiply_float2(
    __global const float2* X,
    __global const float2* H,
    __global       float2* Y,
    const int N)
{
    int i = get_global_id(0);
    if (i >= N) return;
    float a = X[i].x, b = X[i].y;
    float c = H[i].x, d = H[i].y;
    Y[i] = (float2)(a*c - b*d, a*d + b*c);
}
```

**Параметры**: `L = 2^N`, `M = num_taps`, `S = L - M + 1` (новых семплов на блок)

---

## 3. IIR Filter — Проблемы и Решения

### 3.1 Почему IIR сложно параллелить

Biquad: `y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]`

`y[n]` зависит от `y[n-1]` и `y[n-2]` — **жёсткая последовательная зависимость**.
GPU-параллелизм ТОЛЬКО поперёк каналов, не по времени.

### 3.2 Cascade Biquad SOS (рекомендуемый подход)

**1 work-item = 1 канал** (все семплы + все секции последовательно):

```opencl
__kernel void iir_cascade_biquad(
    __global const float2* input,        // [num_channels * N]
    __global       float2* output,
    __constant     float*  sos_matrix,   // [num_sections * 5] = {b0,b1,b2,a1,a2}
    __global       float2* state,        // [ch * num_sections * 2]
    const int num_sections,
    const int N)
{
    int ch   = get_global_id(0);
    int base = ch * N;

    for (int sec = 0; sec < num_sections; sec++) {
        float b0 = sos_matrix[sec*5+0], b1 = sos_matrix[sec*5+1];
        float b2 = sos_matrix[sec*5+2];
        float a1 = sos_matrix[sec*5+3], a2 = sos_matrix[sec*5+4];

        int si = ch * num_sections * 2 + sec * 2;
        float2 y1 = state[si+0], y2 = state[si+1];
        float2 x1 = (float2)(0.0f, 0.0f), x2 = (float2)(0.0f, 0.0f);

        for (int n = 0; n < N; n++) {
            float2 x = (sec == 0) ? input[base+n] : output[base+n];
            float2 y;
            y.x = b0*x.x + b1*x1.x + b2*x2.x - a1*y1.x - a2*y2.x;
            y.y = b0*x.y + b1*x1.y + b2*x2.y - a1*y1.y - a2*y2.y;
            output[base+n] = y;
            x2=x1; x1=x; y2=y1; y1=y;
        }
        state[si+0] = y1; state[si+1] = y2;
    }
}
```

**Форма**: Transposed Direct Form II — лучшая численная устойчивость.

### 3.3 Look-Ahead (продвинутый, для будущего)

Двухпроходный алгоритм:
1. Каждый workgroup вычисляет блок при `y[-1] = 0`
2. Prefix-scan для распространения граничных условий

---

## 4. Сводная Таблица NDRange-стратегий

| Тип | NDRange | Work-item делает | Параллелизм |
|-----|---------|------------------|-------------|
| FIR direct | `(N, channels)` | 1 семпл × 1 канал | По семплам И каналам |
| FIR tiled | `(N,)` | 1 семпл (local mem) | По семплам |
| IIR cascade | `(channels,)` | Весь канал (N семплов, все секции) | Только по каналам |

---

## 5. Раскладка памяти

**Channel-Sequential (рекомендуется)** — coalesced access:
```
Buffer: [ch0_s0, ch0_s1, ..., ch0_sN-1,  ch1_s0, ch1_s1, ...]
Access: input[channel * N + sample]
```

**Channel-Interleaved (плохо для GPU)** — non-coalesced:
```
Buffer: [ch0_s0, ch1_s0, ..., chK_s0,  ch0_s1, ch1_s1, ...]
```

---

## 6. Оптимальные размеры Workgroup

| GPU | Рекомендуемый WG | Причина |
|-----|----------------:|---------|
| AMD RDNA/GCN | 64 или 256 | Wavefront = 64 потока |
| NVIDIA | 128 или 256 | Warp = 32 потока |
| Intel iGPU | 16 или 32 | EU = 8 потоков |

**Практика**: 256 как дефолт.

---

## 7. Производительность (ориентиры)

| Тип | Каналов | Отводов/порядок | Ускорение GPU vs CPU |
|-----|---------|-----------------|---------------------|
| FIR direct | 1 | 64 | ~1x (нет выигрыша!) |
| FIR direct | 64 | 64 | ~40–60x |
| FIR OLS (FFT) | 1 | 1024 | ~8–15x |
| FIR OLS (FFT) | 64 | 1024 | ~100x+ |
| IIR cascade | 1 | 8 секций | ~0.5x (медленнее CPU!) |
| IIR cascade | 64 | 8 секций | ~50–80x |
| IIR cascade | 256 | 8 секций | ~200x+ |

**Вывод**: GPU-фильтры эффективны ТОЛЬКО при multi-channel. Single-channel IIR — лучше на CPU.

---

## 8. JSON Schema

### FIR
```json
{
  "filter_type": "FIR",
  "design_params": {
    "sample_rate": 1000000.0, "cutoff_hz": 100000.0,
    "num_taps": 127, "window": "kaiser", "beta": 8.6
  },
  "coefficients": {
    "format": "float32", "symmetric": true, "num_taps": 127,
    "h": [0.000123, 0.000456, ..., 0.000456, 0.000123]
  }
}
```

### IIR (SOS format)
```json
{
  "filter_type": "IIR",
  "design_params": {
    "sample_rate": 1000000.0, "cutoff_hz": 100000.0,
    "order": 4, "filter_family": "butterworth"
  },
  "coefficients": {
    "format": "sos", "num_sections": 2, "gain": 1.0,
    "sections": [
      {"section": 0, "b": [0.00482, 0.00965, 0.00482], "a": [1.0, -1.8187, 0.8275]},
      {"section": 1, "b": [0.00482, 0.00965, 0.00482], "a": [1.0, -1.9116, 0.9149]}
    ]
  }
}
```

`a[0]` всегда `1.0` (нормированная форма). Scipy: `sos = butter(4, 0.1, output='sos')`

---

## 9. Decision Tree для выбора ядра

```
FIR?
  num_taps ≤ 64    → fir_direct_float2,   __constant coeffs
  num_taps ≤ 512   → fir_tiled_float2,    local memory
  num_taps > 512   → Overlap-Save,        переиспользуем FFTProcessor

IIR?
  num_channels ≥ 8 → iir_cascade_biquad (1 work-item per channel)
  num_channels < 8 → рассмотреть CPU
  нужна точность   → Transposed Direct Form II
```

---

## 10. Управление состоянием IIR (для стриминга)

```
state buffer: [num_channels * num_sections * 2] float2
state[ch * num_sections * 2 + sec * 2 + 0] = y[n-1]
state[ch * num_sections * 2 + sec * 2 + 1] = y[n-2]
```

Это важно для streaming — персистировать между блоками данных.

---

*Исследование: 2026-02-18*
*Автор: AI Research Agent*
*Используется в: TASK-007 (spectrum)*
