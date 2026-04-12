/**
 * @file complex_to_mag_phase_rocm.cpp
 * @brief ComplexToMagPhaseROCm — Ref03 implementation (GpuContext + BufferSet + Ops)
 *
 * Refactored from legacy manual hiprtc/hipMalloc to Ref03 Unified Architecture.
 * GpuContext handles: kernel compilation, disk cache, arch detection, warp_size.
 * BufferSet handles: GPU buffer lifecycle (lazy alloc, reuse, RAII cleanup).
 * MagPhaseOp / MagnitudeOp handle: kernel launch with correct args.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-01 (v1 legacy), 2026-03-22 (v2 Ref03)
 */

#if ENABLE_ROCM

#include "complex_to_mag_phase_rocm.hpp"
#include "kernels/complex_to_mag_phase_kernels_rocm.hpp"
#include "services/console_output.hpp"
#include "logger/logger.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>

namespace fft_processor {

// Both kernels compiled in one GpuContext::CompileModule() call
static const std::vector<std::string> kC2MPKernelNames = {
    "complex_to_mag_phase",
    "complex_to_magnitude"
};

// =========================================================================
// PART 1: Constructor / Destructor / Move
// =========================================================================

ComplexToMagPhaseROCm::ComplexToMagPhaseROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "C2MP", "modules/fft_func/kernels") {
}

ComplexToMagPhaseROCm::~ComplexToMagPhaseROCm() {
  mag_phase_op_.Release();
  magnitude_op_.Release();
}

ComplexToMagPhaseROCm::ComplexToMagPhaseROCm(ComplexToMagPhaseROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , bufs_(std::move(other.bufs_))
    , mag_phase_op_(std::move(other.mag_phase_op_))
    , magnitude_op_(std::move(other.magnitude_op_))
    , compiled_(other.compiled_)
    , n_point_(other.n_point_) {
  other.compiled_ = false;
  other.n_point_ = 0;
}

ComplexToMagPhaseROCm& ComplexToMagPhaseROCm::operator=(ComplexToMagPhaseROCm&& other) noexcept {
  if (this != &other) {
    mag_phase_op_.Release();
    magnitude_op_.Release();

    ctx_ = std::move(other.ctx_);
    bufs_ = std::move(other.bufs_);
    mag_phase_op_ = std::move(other.mag_phase_op_);
    magnitude_op_ = std::move(other.magnitude_op_);
    compiled_ = other.compiled_;
    n_point_ = other.n_point_;

    other.compiled_ = false;
    other.n_point_ = 0;
  }
  return *this;
}

// =========================================================================
// Lazy compilation (GpuContext — one call for all kernels)
// =========================================================================

void ComplexToMagPhaseROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(
      kernels::GetCombinedC2MPKernelSource(),
      kC2MPKernelNames,
      {"-DBLOCK_SIZE=256"});
  mag_phase_op_.Initialize(ctx_);
  magnitude_op_.Initialize(ctx_);
  compiled_ = true;
}

// =========================================================================
// Buffer management (BufferSet — lazy alloc, reuse)
// =========================================================================

void ComplexToMagPhaseROCm::AllocateBuffers(size_t batch_beam_count) {
  size_t input_size = batch_beam_count * n_point_ * sizeof(std::complex<float>);
  size_t output_size = batch_beam_count * n_point_ * 2 * sizeof(float);
  size_t mag_size = batch_beam_count * n_point_ * sizeof(float);

  bufs_.Require(kInput, input_size);
  bufs_.Require(kOutput, output_size);
  bufs_.Require(kMagOnly, mag_size);
}

// =========================================================================
// Data transfer
// =========================================================================

void ComplexToMagPhaseROCm::UploadData(const std::complex<float>* data, size_t count) {
  hipError_t err = hipMemcpyHtoDAsync(
      bufs_.Get(kInput),
      const_cast<std::complex<float>*>(data),
      count * sizeof(std::complex<float>), ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("C2MP UploadData: " + std::string(hipGetErrorString(err)));
  }
}

void ComplexToMagPhaseROCm::CopyGpuData(void* src, size_t offset_bytes, size_t count) {
  char* src_ptr = static_cast<char*>(src) + offset_bytes;
  hipError_t err = hipMemcpyDtoDAsync(
      bufs_.Get(kInput), src_ptr,
      count * sizeof(std::complex<float>), ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("C2MP CopyGpuData: " + std::string(hipGetErrorString(err)));
  }
}

// =========================================================================
// Read results
// =========================================================================

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::ReadResults(
    size_t beam_count, size_t start_beam) {
  size_t total = beam_count * n_point_;
  std::vector<float> raw(total * 2);

  hipError_t err = hipMemcpyDtoH(raw.data(), bufs_.Get(kOutput),
                                   total * 2 * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("C2MP ReadResults: " + std::string(hipGetErrorString(err)));
  }

  std::vector<MagPhaseResult> results;
  results.reserve(beam_count);
  for (size_t i = 0; i < beam_count; ++i) {
    MagPhaseResult r;
    r.beam_id = static_cast<uint32_t>(start_beam + i);
    r.n_point = n_point_;
    r.magnitude.resize(n_point_);
    r.phase.resize(n_point_);
    for (uint32_t k = 0; k < n_point_; ++k) {
      size_t idx = (i * n_point_ + k) * 2;
      r.magnitude[k] = raw[idx];
      r.phase[k] = raw[idx + 1];
    }
    results.push_back(std::move(r));
  }
  return results;
}

// =========================================================================
// Utilities
// =========================================================================

size_t ComplexToMagPhaseROCm::CalculateBytesPerBeam() const {
  return n_point_ * sizeof(std::complex<float>) +
         n_point_ * 2 * sizeof(float);
}

float ComplexToMagPhaseROCm::ComputeInvN(const MagPhaseParams& params) {
  if (params.norm_coeff < 0.0f)
    return (params.n_point > 0) ? 1.0f / static_cast<float>(params.n_point) : 1.0f;
  if (params.norm_coeff > 0.0f)
    return params.norm_coeff;
  return 1.0f;
}

// =========================================================================
// PART 2: Public API — Process (CPU → CPU)
// =========================================================================

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::Process(
    const std::vector<std::complex<float>>& data,
    const MagPhaseParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("C2MP::Process: input size " + std::to_string(data.size()) +
                                 " != expected " + std::to_string(expected));
  }

  n_point_ = params.n_point;
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam();
  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit);
  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  std::vector<MagPhaseResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count);
    UploadData(data.data() + batch.start * params.n_point, batch.count * params.n_point);
    mag_phase_op_.Execute(bufs_.Get(kInput), bufs_.Get(kOutput), batch.count * n_point_);
    hipStreamSynchronize(ctx_.stream());
    auto br = ReadResults(batch.count, batch.start);
    for (auto& r : br) all_results.push_back(std::move(r));
  }
  return all_results;
}

// =========================================================================
// Public API — Process (GPU → CPU)
// =========================================================================

std::vector<MagPhaseResult> ComplexToMagPhaseROCm::Process(
    void* gpu_data, const MagPhaseParams& params, size_t gpu_memory_bytes) {
  if (!gpu_data) throw std::invalid_argument("C2MP::Process: gpu_data is null");

  n_point_ = params.n_point;
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam();
  size_t external_memory = (gpu_memory_bytes > 0)
      ? gpu_memory_bytes
      : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);
  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit, external_memory);
  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  std::vector<MagPhaseResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count);
    size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
    CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);
    mag_phase_op_.Execute(bufs_.Get(kInput), bufs_.Get(kOutput), batch.count * n_point_);
    hipStreamSynchronize(ctx_.stream());
    auto br = ReadResults(batch.count, batch.start);
    for (auto& r : br) all_results.push_back(std::move(r));
  }
  return all_results;
}

// =========================================================================
// Public API — ProcessToGPU (CPU → GPU, caller owns output)
// =========================================================================

void* ComplexToMagPhaseROCm::ProcessToGPU(
    const std::vector<std::complex<float>>& data,
    const MagPhaseParams& params) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected)
    throw std::invalid_argument("C2MP::ProcessToGPU: input size mismatch");

  n_point_ = params.n_point;
  EnsureCompiled();

  size_t total = expected;

  // Temporary input buffer (freed after kernel)
  void* tmp_input = nullptr;
  hipError_t err = hipMalloc(&tmp_input, total * sizeof(std::complex<float>));
  if (err != hipSuccess)
    throw std::runtime_error("C2MP ProcessToGPU: hipMalloc input failed");

  err = hipMemcpyHtoDAsync(tmp_input, const_cast<std::complex<float>*>(data.data()),
                             total * sizeof(std::complex<float>), ctx_.stream());
  if (err != hipSuccess) {
    (void)hipFree(tmp_input);
    throw std::runtime_error("C2MP ProcessToGPU: H2D failed");
  }

  // Output buffer — CALLER OWNS
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, total * 2 * sizeof(float));
  if (err != hipSuccess) {
    (void)hipFree(tmp_input);
    throw std::runtime_error("C2MP ProcessToGPU: hipMalloc output failed");
  }

  mag_phase_op_.Execute(tmp_input, output_ptr, total);
  hipStreamSynchronize(ctx_.stream());
  (void)hipFree(tmp_input);

  return output_ptr;  // CALLER OWNS
}

// =========================================================================
// Public API — ProcessToGPU (GPU → GPU, caller owns output)
// =========================================================================

void* ComplexToMagPhaseROCm::ProcessToGPU(
    void* gpu_data, const MagPhaseParams& params, size_t /*gpu_memory_bytes*/) {
  if (!gpu_data) throw std::invalid_argument("C2MP::ProcessToGPU: gpu_data is null");

  n_point_ = params.n_point;
  EnsureCompiled();

  size_t total = static_cast<size_t>(params.beam_count) * params.n_point;

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, total * 2 * sizeof(float));
  if (err != hipSuccess)
    throw std::runtime_error("C2MP ProcessToGPU: hipMalloc output failed");

  // Zero-copy: kernel reads directly from gpu_data
  mag_phase_op_.Execute(gpu_data, output_ptr, total);
  hipStreamSynchronize(ctx_.stream());

  return output_ptr;  // CALLER OWNS
}

// =========================================================================
// Public API — ProcessMagnitude (GPU → CPU, magnitude only)
// =========================================================================

std::vector<MagnitudeResult> ComplexToMagPhaseROCm::ProcessMagnitude(
    void* gpu_data, const MagPhaseParams& params, size_t gpu_memory_bytes) {
  if (!gpu_data) throw std::invalid_argument("C2MP::ProcessMagnitude: gpu_data is null");

  n_point_ = params.n_point;
  EnsureCompiled();
  float inv_n = ComputeInvN(params);

  size_t bytes_per_beam = CalculateBytesPerBeam();
  size_t external_memory = (gpu_memory_bytes > 0)
      ? gpu_memory_bytes
      : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);
  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit, external_memory);
  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  std::vector<MagnitudeResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count);
    size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
    CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

    size_t total = batch.count * n_point_;
    magnitude_op_.Execute(bufs_.Get(kInput), bufs_.Get(kMagOnly), total, inv_n);
    hipStreamSynchronize(ctx_.stream());

    std::vector<float> raw(total);
    hipError_t err = hipMemcpyDtoH(raw.data(), bufs_.Get(kMagOnly), total * sizeof(float));
    if (err != hipSuccess)
      throw std::runtime_error("C2MP ProcessMagnitude: D2H failed");

    for (size_t i = 0; i < batch.count; ++i) {
      MagnitudeResult result;
      result.beam_id = static_cast<uint32_t>(batch.start + i);
      result.n_point = n_point_;
      result.magnitude.assign(raw.data() + i * n_point_,
                               raw.data() + (i + 1) * n_point_);
      all_results.push_back(std::move(result));
    }
  }
  return all_results;
}

// =========================================================================
// Public API — ProcessMagnitudeToGPU (GPU → GPU, caller owns output)
// =========================================================================

void* ComplexToMagPhaseROCm::ProcessMagnitudeToGPU(
    void* gpu_data, const MagPhaseParams& params, size_t /*gpu_memory_bytes*/) {
  if (!gpu_data) throw std::invalid_argument("C2MP::ProcessMagnitudeToGPU: gpu_data is null");

  n_point_ = params.n_point;
  EnsureCompiled();
  float inv_n = ComputeInvN(params);

  size_t total = static_cast<size_t>(params.beam_count) * params.n_point;

  void* output_ptr = nullptr;
  hipError_t err = hipMalloc(&output_ptr, total * sizeof(float));
  if (err != hipSuccess)
    throw std::runtime_error("C2MP ProcessMagnitudeToGPU: hipMalloc failed");

  magnitude_op_.Execute(gpu_data, output_ptr, total, inv_n);
  hipStreamSynchronize(ctx_.stream());

  return output_ptr;  // CALLER OWNS
}

// =========================================================================
// Public API — ProcessMagnitudeToBuffer (GPU → caller's GPU buffer)
// =========================================================================

void ComplexToMagPhaseROCm::ProcessMagnitudeToBuffer(
    void* gpu_complex_in, void* gpu_magnitude_out,
    const MagPhaseParams& params) {
  if (!gpu_complex_in || !gpu_magnitude_out)
    throw std::invalid_argument("C2MP::ProcessMagnitudeToBuffer: null pointer");

  n_point_ = params.n_point;
  EnsureCompiled();
  float inv_n = ComputeInvN(params);

  size_t total = static_cast<size_t>(params.beam_count) * params.n_point;
  magnitude_op_.Execute(gpu_complex_in, gpu_magnitude_out, total, inv_n);
  hipStreamSynchronize(ctx_.stream());
}

}  // namespace fft_processor

#endif  // ENABLE_ROCM
