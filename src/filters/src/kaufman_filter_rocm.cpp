/**
 * @file kaufman_filter_rocm.cpp
 * @brief KaufmanFilterROCm implementation (KAMA)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include "filters/kaufman_filter_rocm.hpp"
#include "kernels/kaufman_kernels_rocm.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cmath>
#include <cstring>

namespace filters {

static const std::vector<std::string> kKaufmanKernelNames = { "kaufman_kernel" };

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

KaufmanFilterROCm::KaufmanFilterROCm(
    drv_gpu_lib::IBackend* backend, unsigned int block_size)
    : ctx_(backend, "KAMA", "modules/filters/kernels")
    , block_size_(block_size) {
  // Kernel compiled lazily in EnsureKernel() — requires er_period from SetParams()
}

KaufmanFilterROCm::~KaufmanFilterROCm() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

KaufmanFilterROCm::KaufmanFilterROCm(KaufmanFilterROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , compiled_(other.compiled_)
    , params_(other.params_)
    , fast_sc_(other.fast_sc_), slow_sc_(other.slow_sc_)
    , cached_input_buf_(other.cached_input_buf_)
    , cached_input_size_(other.cached_input_size_)
    , block_size_(other.block_size_)
    , compiled_window_size_(other.compiled_window_size_) {
  other.compiled_ = false;
  other.cached_input_buf_ = nullptr;
  other.cached_input_size_ = 0;
  other.compiled_window_size_ = 0;
}

KaufmanFilterROCm& KaufmanFilterROCm::operator=(
    KaufmanFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    ctx_ = std::move(other.ctx_);
    compiled_ = other.compiled_;
    params_ = other.params_;
    fast_sc_ = other.fast_sc_;
    slow_sc_ = other.slow_sc_;
    cached_input_buf_ = other.cached_input_buf_;
    cached_input_size_ = other.cached_input_size_;
    block_size_ = other.block_size_;
    compiled_window_size_ = other.compiled_window_size_;

    other.compiled_ = false;
    other.cached_input_buf_ = nullptr;
    other.cached_input_size_ = 0;
    other.compiled_window_size_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::SetParams(const KaufmanParams& params) {
  SetParams(params.er_period, params.fast_period, params.slow_period);
}

void KaufmanFilterROCm::SetParams(uint32_t er_period,
                                   uint32_t fast_period,
                                   uint32_t slow_period) {
  if (er_period == 0)
    throw std::invalid_argument("KaufmanFilterROCm: er_period must be > 0");
  if (fast_period == 0 || slow_period == 0)
    throw std::invalid_argument("KaufmanFilterROCm: fast/slow periods must be > 0");

  params_.er_period = er_period;
  params_.fast_period = fast_period;
  params_.slow_period = slow_period;

  // fast_sc_ и slow_sc_ — EMA-сглаживающие константы для предельных случаев KAMA:
  // При ER=1 (чистый тренд) используем fast_sc = 2/(fast+1) -> быстрое следование
  // При ER=0 (шум) используем slow_sc = 2/(slow+1) -> почти нет изменений
  // SC (smoothing constant) интерполируется: SC = (ER*(fast-slow)+slow)^2
  // Формула 2/(N+1) — стандартная EMA, возводим в квадрат для нелинейного отклика
  fast_sc_ = 2.0f / static_cast<float>(fast_period + 1);
  slow_sc_ = 2.0f / static_cast<float>(slow_period + 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::EnsureKernel() {
  uint32_t er = params_.er_period;
  if (er == 0)
    throw std::runtime_error("KaufmanFilterROCm: SetParams() must be called before Process()");
  if (compiled_ && compiled_window_size_ == er) return;

  if (compiled_) {
    // N_WINDOW changed — need to recompile. Reconstruct GpuContext to reset module.
    auto* backend = ctx_.backend();
    ctx_ = drv_gpu_lib::GpuContext(backend, "KAMA", "modules/filters/kernels");
    compiled_ = false;
  }

  std::string block_size_def  = "-DBLOCK_SIZE="  + std::to_string(block_size_);
  std::string n_window_def    = "-DN_WINDOW="    + std::to_string(er);
  ctx_.CompileModule(kernels::GetKaufmanSource_rocm(), kKaufmanKernelNames,
                     {block_size_def, n_window_def});
  compiled_ = true;
  compiled_window_size_ = er;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
KaufmanFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr || channels == 0 || points == 0)
    throw std::runtime_error("KaufmanFilterROCm::Process: invalid arguments");
  EnsureKernel();

  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("KaufmanFilterROCm: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));

  unsigned int ch = channels;
  unsigned int pts = points;
  unsigned int N = params_.er_period;
  float fast = fast_sc_;
  float slow = slow_sc_;

  void* args[] = {
    &input_ptr, &output_ptr,
    &ch, &pts,
    &N, &fast, &slow
  };

  unsigned int grid_x = (channels + block_size_ - 1) / block_size_;

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("kaufman_kernel"),
      grid_x, 1, 1,
      block_size_, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error("KaufmanFilterROCm: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  hipStreamSynchronize(ctx_.stream());

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = channels;
  result.n_point = points;
  result.data = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

drv_gpu_lib::InputData<void*>
KaufmanFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {
  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  if (data.size() < total)
    throw std::runtime_error("KaufmanFilterROCm::ProcessFromCPU: data too small");

  // Кешируем input-буфер — hipMalloc дорогой (~0.5 мс).
  // При неизменном размере только перезаписываем данные без повторной аллокации.
  if (buffer_size != cached_input_size_) {
    if (cached_input_buf_) hipFree(cached_input_buf_);
    hipError_t err = hipMalloc(&cached_input_buf_, buffer_size);
    if (err != hipSuccess)
      throw std::runtime_error("KaufmanFilterROCm: hipMalloc(input) failed");
    cached_input_size_ = buffer_size;
  }

  hipMemcpyHtoDAsync(cached_input_buf_, data.data(), buffer_size, ctx_.stream());
  return Process(cached_input_buf_, channels, points);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
KaufmanFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) const {
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> output(total);

  const uint32_t N = params_.er_period;
  const float eps = 1e-8f;
  const float sc_diff = fast_sc_ - slow_sc_;

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    // Passthrough first N samples
    for (uint32_t i = 0; i < N && i < points; ++i) {
      output[base + i] = input[base + i];
    }
    if (points <= N) continue;

    // KAMA initial state = last sample of warmup period
    auto kama = input[base + N - 1];

    for (uint32_t n = N; n < points; ++n) {
      auto x = input[base + n];

      // Direction: |x[n] - x[n-N]|
      float dir_re = std::abs(x.real() - input[base + n - N].real());
      float dir_im = std::abs(x.imag() - input[base + n - N].imag());

      // Volatility: sum |x[i] - x[i-1]| for i = n-N+1 to n (N terms)
      float vol_re = 0.0f, vol_im = 0.0f;
      for (uint32_t i = n - N + 1; i <= n; ++i) {
        vol_re += std::abs(input[base + i].real() - input[base + i - 1].real());
        vol_im += std::abs(input[base + i].imag() - input[base + i - 1].imag());
      }

      // ER
      float er_re = (vol_re > eps) ? dir_re / vol_re : 0.0f;
      float er_im = (vol_im > eps) ? dir_im / vol_im : 0.0f;

      // SC = (ER*(fast-slow)+slow)^2
      float sc_re = er_re * sc_diff + slow_sc_;
      float sc_im = er_im * sc_diff + slow_sc_;
      sc_re *= sc_re;
      sc_im *= sc_im;

      // Update KAMA
      float kama_re = kama.real() + sc_re * (x.real() - kama.real());
      float kama_im = kama.imag() + sc_im * (x.imag() - kama.imag());
      kama = {kama_re, kama_im};

      output[base + n] = kama;
    }
  }
  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup
// ════════════════════════════════════════════════════════════════════════════

void KaufmanFilterROCm::ReleaseGpuResources() {
  if (cached_input_buf_) { hipFree(cached_input_buf_); cached_input_buf_ = nullptr; }
  cached_input_size_ = 0;
  // GpuContext manages kernel module
}

}  // namespace filters

#endif  // ENABLE_ROCM
