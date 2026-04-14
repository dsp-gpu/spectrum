/**
 * @file fir_filter_rocm.cpp
 * @brief FirFilterROCm implementation - GPU FIR convolution (ROCm/HIP)
 *
 * Port of fir_filter.cpp (OpenCL) to HIP/ROCm.
 * Uses hiprtc for runtime kernel compilation, hipModuleLaunchKernel for dispatch.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <spectrum/filters/fir_filter_rocm.hpp>
#include "kernels/fir_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"
#include <core/services/console_output.hpp>

#include <stdexcept>
#include <cstring>
#include <algorithm>

using fft_func_utils::MakeROCmDataFromEvents;

namespace filters {

static const std::vector<std::string> kFirKernelNames = { "fir_filter_cf32" };

// ========================================================================
// Constructor / Destructor
// ========================================================================

FirFilterROCm::FirFilterROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "FIR", "modules/filters/kernels") {
  EnsureCompiled();
}

FirFilterROCm::~FirFilterROCm() {
  ReleaseGpuResources();
}

FirFilterROCm::FirFilterROCm(FirFilterROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , coefficients_(std::move(other.coefficients_))
    , compiled_(other.compiled_)
    , coeff_buf_(other.coeff_buf_) {
  other.compiled_ = false;
  other.coeff_buf_ = nullptr;
}

FirFilterROCm& FirFilterROCm::operator=(FirFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    ctx_ = std::move(other.ctx_);
    coefficients_ = std::move(other.coefficients_);
    compiled_ = other.compiled_;
    coeff_buf_ = other.coeff_buf_;
    other.compiled_ = false;
    other.coeff_buf_ = nullptr;
  }
  return *this;
}

// ========================================================================
// Configuration
// ========================================================================

void FirFilterROCm::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "fir") {
    throw std::runtime_error(
        "FirFilterROCm::LoadConfig: expected type 'fir', got '" + cfg.type + "'");
  }
  SetCoefficients(cfg.coefficients);
}

void FirFilterROCm::SetCoefficients(const std::vector<float>& coeffs) {
  if (coeffs.empty()) {
    throw std::invalid_argument("FirFilterROCm::SetCoefficients: empty coefficients");
  }

  coefficients_ = coeffs;
  UploadCoefficients();
}

// ========================================================================
// GPU Processing
// ========================================================================

drv_gpu_lib::InputData<void*>
FirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points,
                       ROCmProfEvents* prof_events) {
  if (!input_ptr) {
    throw std::invalid_argument("FirFilterROCm::Process: input_ptr is null");
  }
  if (channels == 0 || points == 0) {
    throw std::runtime_error("FirFilterROCm::Process: channels or points is 0");
  }
  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilterROCm::Process: no coefficients set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);
  hipError_t err;

  // Allocate output buffer
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::Process: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Kernel arguments
  unsigned int num_taps = static_cast<unsigned int>(coefficients_.size());
  unsigned int ch = channels;
  unsigned int pts = points;

  void* args[] = {
    &input_ptr,
    &output_ptr,
    &coeff_buf_,
    &num_taps,
    &ch,
    &pts
  };

  // 2D grid: X = samples, Y = channels (eliminates div/mod in kernel)
  unsigned int grid_x = static_cast<unsigned int>(
      (points + kBlockSize - 1) / kBlockSize);
  unsigned int grid_y = channels;

  hipEvent_t ev_k_s = nullptr, ev_k_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_s);
    hipEventCreate(&ev_k_e);
    hipEventRecord(ev_k_s, ctx_.stream());
  }

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("fir_filter_cf32"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (prof_events) {
    hipEventRecord(ev_k_e, ctx_.stream());
  }

  if (err != hipSuccess) {
    if (ev_k_s) { hipEventDestroy(ev_k_s); hipEventDestroy(ev_k_e); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "FirFilterROCm::Process: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_s, ev_k_e, 0, "fir_filter")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = channels;
  result.n_point       = points;
  result.data          = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

drv_gpu_lib::InputData<void*>
FirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points,
    ROCmProfEvents* prof_events)
{
  size_t expected = static_cast<size_t>(channels) * points;
  if (data.size() < expected) {
    throw std::invalid_argument(
        "FirFilterROCm::ProcessFromCPU: input size " +
        std::to_string(data.size()) + " < expected " +
        std::to_string(expected));
  }

  size_t data_size = expected * sizeof(std::complex<float>);
  void* input_ptr = nullptr;
  hipError_t err = hipMalloc(&input_ptr, data_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::ProcessFromCPU: hipMalloc(input) failed");
  }

  // H2D Upload timing
  hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_up_s);
    hipEventCreate(&ev_up_e);
    hipEventRecord(ev_up_s, ctx_.stream());
  }

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, ctx_.stream());

  if (prof_events) {
    hipEventRecord(ev_up_e, ctx_.stream());
  }

  if (err != hipSuccess) {
    if (ev_up_s) { hipEventDestroy(ev_up_s); hipEventDestroy(ev_up_e); }
    (void)hipFree(input_ptr);
    throw std::runtime_error(
        "FirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
  }
  // No hipStreamSynchronize here: kernel in same stream will wait for H2D

  if (prof_events) {
    prof_events->push_back({"Upload",
        MakeROCmDataFromEvents(ev_up_s, ev_up_e, 0, "H2D")});
  }

  auto result = Process(input_ptr, channels, points, prof_events);

  (void)hipFree(input_ptr);
  return result;
}

// ========================================================================
// CPU Reference Implementation
// ========================================================================

std::vector<std::complex<float>>
FirFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (coefficients_.empty()) {
    throw std::runtime_error("FirFilterROCm::ProcessCpu: no coefficients set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("FirFilterROCm::ProcessCpu: input too small");
  }

  std::vector<std::complex<float>> output(total, {0.0f, 0.0f});
  uint32_t num_taps = static_cast<uint32_t>(coefficients_.size());

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;
    for (uint32_t n = 0; n < points; ++n) {
      std::complex<float> acc(0.0f, 0.0f);
      for (uint32_t k = 0; k < num_taps; ++k) {
        int idx = static_cast<int>(n) - static_cast<int>(k);
        if (idx >= 0) {
          acc += coefficients_[k] * input[base + idx];
        }
      }
      output[base + n] = acc;
    }
  }

  return output;
}

// ========================================================================
// GPU Internals
// ========================================================================

void FirFilterROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetFirDirectSource_rocm(), kFirKernelNames,
                     {"-DBLOCK_SIZE=256"});
  compiled_ = true;
}

void FirFilterROCm::UploadCoefficients() {
  if (coeff_buf_) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
  }

  size_t coeff_size = coefficients_.size() * sizeof(float);
  hipError_t err = hipMalloc(&coeff_buf_, coeff_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "FirFilterROCm::UploadCoefficients: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  err = hipMemcpyHtoDAsync(coeff_buf_, coefficients_.data(),
                            coeff_size, ctx_.stream());
  if (err != hipSuccess) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
    throw std::runtime_error(
        "FirFilterROCm::UploadCoefficients: hipMemcpyHtoDAsync failed");
  }
  (void)hipStreamSynchronize(ctx_.stream());
}

void FirFilterROCm::ReleaseGpuResources() {
  // GpuContext manages kernel module
  if (coeff_buf_) {
    (void)hipFree(coeff_buf_);
    coeff_buf_ = nullptr;
  }
}

}  // namespace filters

#endif  // ENABLE_ROCM
