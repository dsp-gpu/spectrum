/**
 * @file moving_average_filter_rocm.cpp
 * @brief MovingAverageFilterROCm implementation (SMA, EMA, MMA, DEMA, TEMA)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01
 */

#if ENABLE_ROCM

#include <spectrum/filters/moving_average_filter_rocm.hpp>
#include <spectrum/kernels/moving_average_kernels_rocm.hpp>
#include <core/services/console_output.hpp>

#include <stdexcept>
#include <cstring>

namespace filters {

static const std::vector<std::string> kSmaKernelNames = {
  "sma_kernel", "ema_kernel", "mma_kernel", "dema_kernel", "tema_kernel"
};

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor
// ════════════════════════════════════════════════════════════════════════════

MovingAverageFilterROCm::MovingAverageFilterROCm(
    drv_gpu_lib::IBackend* backend, unsigned int block_size)
    : ctx_(backend, "SMA", "modules/filters/kernels")
    , block_size_(block_size) {
  // Kernels compiled lazily in EnsureKernels() — SMA requires window_size from SetParams()
}

MovingAverageFilterROCm::~MovingAverageFilterROCm() {
  ReleaseGpuResources();
}

// ════════════════════════════════════════════════════════════════════════════
// Move semantics
// ════════════════════════════════════════════════════════════════════════════

MovingAverageFilterROCm::MovingAverageFilterROCm(
    MovingAverageFilterROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , compiled_(other.compiled_)
    , ma_type_(other.ma_type_), window_size_(other.window_size_)
    , alpha_(other.alpha_)
    , cached_input_buf_(other.cached_input_buf_)
    , cached_input_size_(other.cached_input_size_)
    , block_size_(other.block_size_)
    , compiled_sma_window_(other.compiled_sma_window_) {
  other.compiled_ = false;
  other.cached_input_buf_ = nullptr;
  other.cached_input_size_ = 0;
  other.compiled_sma_window_ = 0;
}

MovingAverageFilterROCm& MovingAverageFilterROCm::operator=(
    MovingAverageFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    ctx_ = std::move(other.ctx_);
    compiled_ = other.compiled_;
    ma_type_ = other.ma_type_;
    window_size_ = other.window_size_;
    alpha_ = other.alpha_;
    cached_input_buf_ = other.cached_input_buf_;
    cached_input_size_ = other.cached_input_size_;
    block_size_ = other.block_size_;
    compiled_sma_window_ = other.compiled_sma_window_;

    other.compiled_ = false;
    other.cached_input_buf_ = nullptr;
    other.cached_input_size_ = 0;
    other.compiled_sma_window_ = 0;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::SetParams(const MovingAverageParams& params) {
  SetParams(params.type, params.window_size);
}

/**
 * @brief Устанавливает тип и размер окна скользящей средней
 *
 * Вычисляет alpha (сглаживающий коэффициент) для передачи в kernel:
 * - EMA/DEMA/TEMA: alpha = 2/(N+1) — классическая формула: при N=10 → alpha=0.182
 * - MMA (Wilder): alpha = 1/N — более медленная реакция чем EMA при том же N
 * - SMA: alpha не используется ядром (работает ring buffer + inv_N)
 *
 * SMA ring buffer размером N хранится в thread-local регистрах. Размер задаётся через
 * hiprtc define -DN_WINDOW=<window_size> при компиляции — нет ограничения 128.
 *
 * @param type Тип скользящей средней (SMA/EMA/MMA/DEMA/TEMA)
 * @param window_size N — размер окна; SMA: max 128
 */
void MovingAverageFilterROCm::SetParams(MAType type, uint32_t window_size) {
  if (window_size == 0)
    throw std::invalid_argument("MovingAverageFilterROCm: window_size must be > 0");

  ma_type_ = type;
  window_size_ = window_size;

  switch (type) {
    case MAType::EMA:
    case MAType::DEMA:
    case MAType::TEMA:
      alpha_ = 2.0f / (static_cast<float>(window_size) + 1.0f);
      break;
    case MAType::MMA:
      alpha_ = 1.0f / static_cast<float>(window_size);
      break;
    case MAType::SMA:
      alpha_ = 1.0f / static_cast<float>(window_size);  // not used by kernel
      break;
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel compilation
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::EnsureKernels() {
  // For SMA: recompile when window_size changes (N_WINDOW define changes).
  // For EMA/MMA/DEMA/TEMA: compile once with any N_WINDOW (not used by those kernels).
  bool sma_window_changed = (ma_type_ == MAType::SMA &&
                              compiled_sma_window_ != window_size_);
  if (compiled_ && !sma_window_changed) return;

  if (compiled_) {
    // N_WINDOW changed — need to recompile. Reconstruct GpuContext to reset module.
    auto* backend = ctx_.backend();
    ctx_ = drv_gpu_lib::GpuContext(backend, "SMA", "modules/filters/kernels");
    compiled_ = false;
  }

  std::string block_size_def = "-DBLOCK_SIZE=" + std::to_string(block_size_);
  std::string n_window_def   = "-DN_WINDOW="   + std::to_string(window_size_);
  ctx_.CompileModule(kernels::GetMovingAverageSource_rocm(), kSmaKernelNames,
                     {block_size_def, n_window_def});
  compiled_ = true;
  compiled_sma_window_ = window_size_;
}

// ════════════════════════════════════════════════════════════════════════════
// GPU Processing
// ════════════════════════════════════════════════════════════════════════════

drv_gpu_lib::InputData<void*>
MovingAverageFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points) {
  if (!input_ptr || channels == 0 || points == 0)
    throw std::runtime_error("MovingAverageFilterROCm::Process: invalid arguments");
  EnsureKernels();

  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  // Allocate output (caller must hipFree)
  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));

  unsigned int ch = channels;
  unsigned int pts = points;
  unsigned int N = window_size_;
  float inv_N = 1.0f / static_cast<float>(window_size_);
  float alpha = alpha_;

  // Выбор kernel и набора аргументов по типу MA:
  // SMA: signature (in, out, ch, pts, N, inv_N) — 6 аргументов
  //      N передаём явно для ring buffer, inv_N = 1/N — избегаем деления в kernel
  // EMA/MMA/DEMA/TEMA: signature (in, out, ch, pts, alpha) — 5 аргументов
  //      Alpha уже вычислен в SetParams() и не зависит от N во время работы
  hipFunction_t kernel = nullptr;
  void* args[7];

  if (ma_type_ == MAType::SMA) {
    kernel = ctx_.GetKernel("sma_kernel");
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &ch;
    args[3] = &pts;
    args[4] = &N;
    args[5] = &inv_N;  // precomputed 1/N: передаём чтобы избежать деления в kernel
  } else {
    // EMA, MMA, DEMA, TEMA — единая сигнатура с alpha
    switch (ma_type_) {
      case MAType::EMA:  kernel = ctx_.GetKernel("ema_kernel");  break;
      case MAType::MMA:  kernel = ctx_.GetKernel("mma_kernel");  break;
      case MAType::DEMA: kernel = ctx_.GetKernel("dema_kernel"); break;
      case MAType::TEMA: kernel = ctx_.GetKernel("tema_kernel"); break;
      default: break;
    }
    args[0] = &input_ptr;
    args[1] = &output_ptr;
    args[2] = &ch;
    args[3] = &pts;
    args[4] = &alpha;
  }

  // 1D grid: one thread per channel
  unsigned int grid_x = (channels + block_size_ - 1) / block_size_;

  err = hipModuleLaunchKernel(
      kernel,
      grid_x, 1, 1,
      block_size_, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (err != hipSuccess) {
    hipFree(output_ptr);
    throw std::runtime_error("MovingAverageFilterROCm: hipModuleLaunchKernel failed: " +
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
MovingAverageFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points) {
  size_t total = static_cast<size_t>(channels) * points;
  size_t buffer_size = total * sizeof(std::complex<float>);

  if (data.size() < total)
    throw std::runtime_error("MovingAverageFilterROCm::ProcessFromCPU: data too small");

  // Кешируем input-буфер на GPU: hipMalloc/hipFree дорогие операции (~0.5 мс).
  // Если размер совпадает — просто перезаписываем данные без переаллокации.
  // Буфер принадлежит объекту (освобождается в ReleaseGpuResources).
  if (buffer_size != cached_input_size_) {
    if (cached_input_buf_) hipFree(cached_input_buf_);
    hipError_t err = hipMalloc(&cached_input_buf_, buffer_size);
    if (err != hipSuccess)
      throw std::runtime_error("MovingAverageFilterROCm: hipMalloc(input) failed");
    cached_input_size_ = buffer_size;
  }

  hipError_t merr = hipMemcpyHtoDAsync(cached_input_buf_, data.data(), buffer_size, ctx_.stream());
  if (merr != hipSuccess)
    throw std::runtime_error("MovingAverageFilterROCm: hipMemcpyHtoDAsync failed: " +
                              std::string(hipGetErrorString(merr)));
  return Process(cached_input_buf_, channels, points);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Reference (for testing)
// ════════════════════════════════════════════════════════════════════════════

std::vector<std::complex<float>>
MovingAverageFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) const {
  size_t total = static_cast<size_t>(channels) * points;
  std::vector<std::complex<float>> output(total, {0.0f, 0.0f});

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    switch (ma_type_) {
      case MAType::SMA: {
        std::vector<std::complex<float>> ring(window_size_);
        float sum_re = 0.0f, sum_im = 0.0f;
        uint32_t head = 0;
        for (uint32_t n = 0; n < points; ++n) {
          auto x = input[base + n];
          if (n < window_size_) {
            ring[n] = x;
            sum_re += x.real();
            sum_im += x.imag();
            float inv = 1.0f / static_cast<float>(n + 1);
            output[base + n] = {sum_re * inv, sum_im * inv};
          } else {
            auto old_val = ring[head];
            ring[head] = x;
            head = (head + 1) % window_size_;
            sum_re += x.real() - old_val.real();
            sum_im += x.imag() - old_val.imag();
            float inv_N = 1.0f / static_cast<float>(window_size_);
            output[base + n] = {sum_re * inv_N, sum_im * inv_N};
          }
        }
        break;
      }

      case MAType::EMA:
      case MAType::MMA: {
        auto state = input[base];
        output[base] = state;
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          state = {a * x.real() + b * state.real(),
                   a * x.imag() + b * state.imag()};
          output[base + n] = state;
        }
        break;
      }

      case MAType::DEMA: {
        auto ema1 = input[base];
        auto ema2 = input[base];
        output[base] = {2.0f * ema1.real() - ema2.real(),
                        2.0f * ema1.imag() - ema2.imag()};
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          ema1 = {a * x.real()      + b * ema1.real(),
                  a * x.imag()      + b * ema1.imag()};
          ema2 = {a * ema1.real()   + b * ema2.real(),
                  a * ema1.imag()   + b * ema2.imag()};
          output[base + n] = {2.0f * ema1.real() - ema2.real(),
                              2.0f * ema1.imag() - ema2.imag()};
        }
        break;
      }

      case MAType::TEMA: {
        auto ema1 = input[base];
        auto ema2 = input[base];
        auto ema3 = input[base];
        output[base] = {3.0f * ema1.real() - 3.0f * ema2.real() + ema3.real(),
                        3.0f * ema1.imag() - 3.0f * ema2.imag() + ema3.imag()};
        float a = alpha_;
        float b = 1.0f - a;
        for (uint32_t n = 1; n < points; ++n) {
          auto x = input[base + n];
          ema1 = {a * x.real()      + b * ema1.real(),
                  a * x.imag()      + b * ema1.imag()};
          ema2 = {a * ema1.real()   + b * ema2.real(),
                  a * ema1.imag()   + b * ema2.imag()};
          ema3 = {a * ema2.real()   + b * ema3.real(),
                  a * ema2.imag()   + b * ema3.imag()};
          output[base + n] = {3.0f * ema1.real() - 3.0f * ema2.real() + ema3.real(),
                              3.0f * ema1.imag() - 3.0f * ema2.imag() + ema3.imag()};
        }
        break;
      }
    }
  }
  return output;
}

// ════════════════════════════════════════════════════════════════════════════
// Cleanup
// ════════════════════════════════════════════════════════════════════════════

void MovingAverageFilterROCm::ReleaseGpuResources() {
  if (cached_input_buf_) { hipFree(cached_input_buf_); cached_input_buf_ = nullptr; }
  cached_input_size_ = 0;
  // GpuContext manages kernel module
}

}  // namespace filters

#endif  // ENABLE_ROCM
