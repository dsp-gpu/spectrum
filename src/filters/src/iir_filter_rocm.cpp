/**
 * @file iir_filter_rocm.cpp
 * @brief IirFilterROCm implementation - GPU IIR biquad cascade (ROCm/HIP)
 *
 * Port of iir_filter.cpp (OpenCL) to HIP/ROCm.
 * Uses GpuContext for kernel compilation and management.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include <spectrum/filters/iir_filter_rocm.hpp>
#include <spectrum/kernels/iir_kernels_rocm.hpp>
#include <spectrum/utils/rocm_profiling_helpers.hpp>
#include <core/services/scoped_hip_event.hpp>
#include <core/services/console_output.hpp>

#include <stdexcept>
#include <cstring>
#include <algorithm>

using fft_func_utils::MakeROCmDataFromEvents;
using drv_gpu_lib::ScopedHipEvent;

namespace filters {

static const std::vector<std::string> kIirKernelNames = { "iir_biquad_cascade_cf32" };

// ========================================================================
// Constructor / Destructor
// ========================================================================

IirFilterROCm::IirFilterROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "IIR", "modules/filters/kernels") {
  EnsureCompiled();
}

IirFilterROCm::~IirFilterROCm() {
  ReleaseGpuResources();
}

IirFilterROCm::IirFilterROCm(IirFilterROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , sections_(std::move(other.sections_))
    , compiled_(other.compiled_)
    , sos_buf_(other.sos_buf_) {
  other.compiled_ = false;
  other.sos_buf_ = nullptr;
}

IirFilterROCm& IirFilterROCm::operator=(IirFilterROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    ctx_ = std::move(other.ctx_);
    sections_ = std::move(other.sections_);
    compiled_ = other.compiled_;
    sos_buf_ = other.sos_buf_;
    other.compiled_ = false;
    other.sos_buf_ = nullptr;
  }
  return *this;
}

// ========================================================================
// Configuration
// ========================================================================

void IirFilterROCm::LoadConfig(const std::string& json_path) {
  auto cfg = FilterConfig::LoadJson(json_path);
  if (cfg.type != "iir") {
    throw std::runtime_error(
        "IirFilterROCm::LoadConfig: expected type 'iir', got '" + cfg.type + "'");
  }
  SetBiquadSections(cfg.sections);
}

void IirFilterROCm::SetBiquadSections(const std::vector<BiquadSection>& sections) {
  if (sections.empty()) {
    throw std::invalid_argument("IirFilterROCm::SetBiquadSections: empty sections");
  }

  sections_ = sections;
  UploadSosMatrix();
}

// ========================================================================
// GPU Processing
// ========================================================================

drv_gpu_lib::InputData<void*>
IirFilterROCm::Process(void* input_ptr, uint32_t channels, uint32_t points,
                       ROCmProfEvents* prof_events) {
  if (!input_ptr) {
    throw std::invalid_argument("IirFilterROCm::Process: input_ptr is null");
  }
  if (channels == 0 || points == 0) {
    throw std::runtime_error("IirFilterROCm::Process: channels or points is 0");
  }
  if (sections_.empty()) {
    throw std::runtime_error("IirFilterROCm::Process: no biquad sections set");
  }

  size_t total_points = static_cast<size_t>(channels) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);
  hipError_t err;

  // Allocate output buffer
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::Process: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Kernel arguments
  unsigned int num_sec = static_cast<unsigned int>(sections_.size());
  unsigned int ch = channels;
  unsigned int pts = points;

  void* args[] = {
    &input_ptr,
    &output_ptr,
    &sos_buf_,
    &num_sec,
    &ch,
    &pts
  };

  // Launch: 1D grid, one thread per channel
  unsigned int grid_size = static_cast<unsigned int>(
      (channels + kBlockSize - 1) / kBlockSize);

  // RAII-события: hipEventDestroy вызывается автоматически
  // в ~ScopedHipEvent при выходе из scope (в т.ч. при throw).
  ScopedHipEvent ev_k_s, ev_k_e;
  if (prof_events) {
    ev_k_s.Create();
    ev_k_e.Create();
    hipEventRecord(ev_k_s.get(), ctx_.stream());
  }

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("iir_biquad_cascade_cf32"),
      grid_size, 1, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (prof_events) {
    hipEventRecord(ev_k_e.get(), ctx_.stream());
  }

  if (err != hipSuccess) {
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "IirFilterROCm::Process: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  (void)hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_s.get(), ev_k_e.get(), 0, "iir_filter")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = channels;
  result.n_point       = points;
  result.data          = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  return result;
}

drv_gpu_lib::InputData<void*>
IirFilterROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t channels, uint32_t points,
    ROCmProfEvents* prof_events)
{
  size_t expected = static_cast<size_t>(channels) * points;
  if (data.size() < expected) {
    throw std::invalid_argument(
        "IirFilterROCm::ProcessFromCPU: input size " +
        std::to_string(data.size()) + " < expected " +
        std::to_string(expected));
  }

  size_t data_size = expected * sizeof(std::complex<float>);
  void* input_ptr = nullptr;
  hipError_t err = hipMalloc(&input_ptr, data_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::ProcessFromCPU: hipMalloc(input) failed");
  }

  // H2D Upload timing (RAII — события освобождаются в ~ScopedHipEvent)
  ScopedHipEvent ev_up_s, ev_up_e;
  if (prof_events) {
    ev_up_s.Create();
    ev_up_e.Create();
    hipEventRecord(ev_up_s.get(), ctx_.stream());
  }

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, ctx_.stream());

  if (prof_events) {
    hipEventRecord(ev_up_e.get(), ctx_.stream());
  }

  if (err != hipSuccess) {
    (void)hipFree(input_ptr);
    throw std::runtime_error(
        "IirFilterROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
  }
  // No hipStreamSynchronize here: kernel in same stream will wait for H2D

  if (prof_events) {
    prof_events->push_back({"Upload",
        MakeROCmDataFromEvents(ev_up_s.get(), ev_up_e.get(), 0, "H2D")});
  }

  auto result = Process(input_ptr, channels, points, prof_events);

  (void)hipFree(input_ptr);
  return result;
}

// ========================================================================
// CPU Reference Implementation
// ========================================================================

std::vector<std::complex<float>>
IirFilterROCm::ProcessCpu(
    const std::vector<std::complex<float>>& input,
    uint32_t channels, uint32_t points) {

  if (sections_.empty()) {
    throw std::runtime_error("IirFilterROCm::ProcessCpu: no biquad sections set");
  }

  size_t total = static_cast<size_t>(channels) * points;
  if (input.size() < total) {
    throw std::invalid_argument("IirFilterROCm::ProcessCpu: input too small");
  }

  // Copy input as working buffer
  std::vector<std::complex<float>> output(input.begin(), input.begin() + total);

  for (uint32_t ch = 0; ch < channels; ++ch) {
    size_t base = static_cast<size_t>(ch) * points;

    for (const auto& sec : sections_) {
      // Direct Form II Transposed state
      std::complex<float> w1(0.0f, 0.0f);
      std::complex<float> w2(0.0f, 0.0f);

      for (uint32_t n = 0; n < points; ++n) {
        std::complex<float> x = output[base + n];
        std::complex<float> y = sec.b0 * x + w1;
        w1 = sec.b1 * x - sec.a1 * y + w2;
        w2 = sec.b2 * x - sec.a2 * y;
        output[base + n] = y;
      }
    }
  }

  return output;
}

// ========================================================================
// GPU Internals
// ========================================================================

void IirFilterROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetIirBiquadCascadeSource_rocm(), kIirKernelNames,
                     {"-DBLOCK_SIZE=256"});
  compiled_ = true;
}

void IirFilterROCm::UploadSosMatrix() {
  if (sos_buf_) {
    (void)hipFree(sos_buf_);
    sos_buf_ = nullptr;
  }

  // Pack sections into flat float array: [sec*5 + {b0,b1,b2,a1,a2}]
  std::vector<float> sos_flat;
  sos_flat.reserve(sections_.size() * 5);
  for (const auto& sec : sections_) {
    sos_flat.push_back(sec.b0);
    sos_flat.push_back(sec.b1);
    sos_flat.push_back(sec.b2);
    sos_flat.push_back(sec.a1);
    sos_flat.push_back(sec.a2);
  }

  size_t sos_size = sos_flat.size() * sizeof(float);
  hipError_t err = hipMalloc(&sos_buf_, sos_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "IirFilterROCm::UploadSosMatrix: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  err = hipMemcpyHtoDAsync(sos_buf_, sos_flat.data(), sos_size, ctx_.stream());
  if (err != hipSuccess) {
    (void)hipFree(sos_buf_);
    sos_buf_ = nullptr;
    throw std::runtime_error(
        "IirFilterROCm::UploadSosMatrix: hipMemcpyHtoDAsync failed");
  }
  (void)hipStreamSynchronize(ctx_.stream());
}

void IirFilterROCm::ReleaseGpuResources() {
  // GpuContext manages kernel module
  if (sos_buf_) {
    (void)hipFree(sos_buf_);
    sos_buf_ = nullptr;
  }
}

}  // namespace filters

#endif  // ENABLE_ROCM
