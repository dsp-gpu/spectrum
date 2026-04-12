/**
 * @file fft_processor_rocm.cpp
 * @brief FFTProcessorROCm — thin Facade implementation (Ref03)
 *
 * Ref03 Unified Architecture: Layer 6 (Facade).
 * Kernel compilation delegated to GpuContext.
 * Kernel launches delegated to PadDataOp + MagPhaseOp.
 * Buffer management via BufferSet<4>.
 * hipFFT plan management via LRU-2 cache (stays in facade).
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23 (v1), 2026-03-14 (v2 Ref03 Facade)
 */

#if ENABLE_ROCM

#include "fft_processor_rocm.hpp"
#include "kernels/fft_processor_kernels_rocm.hpp"
#include "rocm_profiling_helpers.hpp"
#include "services/gpu_profiler.hpp"
#include "config/gpu_config.hpp"
#include "logger/logger.hpp"
#include "services/console_output.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>

using fft_func_utils::MakeROCmDataFromEvents;
using fft_func_utils::MakeROCmDataFromClock;

namespace fft_processor {

// All kernel names used by FFT module
// SNR_02/02b/04: added pad_data_windowed + magnitude kernels for ProcessMagnitudesToGPU.
static const std::vector<std::string> kKernelNames = {
  "pad_data",
  "pad_data_windowed",           // SNR_02b: Hann/Hamming/Blackman window
  "complex_to_mag_phase",
  "complex_to_magnitude",        // SNR_04: |X| output for ProcessMagnitudesToGPU
  "complex_to_magnitude_squared" // SNR_02:  |X|² output (square-law, no sqrt)
};

// =========================================================================
// Constructor / Destructor / Move
// =========================================================================

FFTProcessorROCm::FFTProcessorROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "FFTProc", "modules/fft_func/kernels") {
}

FFTProcessorROCm::~FFTProcessorROCm() {
  pad_op_.Release();
  mag_phase_op_.Release();
  mag_op_.Release();
  ReleasePlans();
}

FFTProcessorROCm::FFTProcessorROCm(FFTProcessorROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , bufs_(std::move(other.bufs_))
    , pad_op_(std::move(other.pad_op_))
    , mag_phase_op_(std::move(other.mag_phase_op_))
    , mag_op_(std::move(other.mag_op_))
    , compiled_(other.compiled_)
    , plan_(other.plan_)
    , plan_created_(other.plan_created_)
    , plan_last_(other.plan_last_)
    , plan_last_batch_(other.plan_last_batch_)
    , nFFT_(other.nFFT_)
    , n_point_(other.n_point_)
    , plan_batch_size_(other.plan_batch_size_)
    , has_mag_phase_buffers_(other.has_mag_phase_buffers_) {
  other.compiled_ = false;
  other.plan_ = 0;
  other.plan_created_ = false;
  other.plan_last_ = 0;
  other.plan_last_batch_ = 0;
  other.plan_batch_size_ = 0;
  other.has_mag_phase_buffers_ = false;
}

FFTProcessorROCm& FFTProcessorROCm::operator=(FFTProcessorROCm&& other) noexcept {
  if (this != &other) {
    pad_op_.Release();
    mag_phase_op_.Release();
    mag_op_.Release();
    ReleasePlans();

    ctx_ = std::move(other.ctx_);
    bufs_ = std::move(other.bufs_);
    pad_op_ = std::move(other.pad_op_);
    mag_phase_op_ = std::move(other.mag_phase_op_);
    mag_op_ = std::move(other.mag_op_);
    compiled_ = other.compiled_;
    plan_ = other.plan_;
    plan_created_ = other.plan_created_;
    plan_last_ = other.plan_last_;
    plan_last_batch_ = other.plan_last_batch_;
    nFFT_ = other.nFFT_;
    n_point_ = other.n_point_;
    plan_batch_size_ = other.plan_batch_size_;
    has_mag_phase_buffers_ = other.has_mag_phase_buffers_;

    other.compiled_ = false;
    other.plan_ = 0;
    other.plan_created_ = false;
    other.plan_last_ = 0;
    other.plan_last_batch_ = 0;
    other.plan_batch_size_ = 0;
    other.has_mag_phase_buffers_ = false;
  }
  return *this;
}

// =========================================================================
// Lazy compilation
// =========================================================================

void FFTProcessorROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetHIPKernelSource(), kKernelNames);
  pad_op_.Initialize(ctx_);
  mag_phase_op_.Initialize(ctx_);
  mag_op_.Initialize(ctx_);  // SNR_04
  compiled_ = true;
}

// =========================================================================
// Buffer management (BufferSet)
// =========================================================================

void FFTProcessorROCm::AllocateBuffers(size_t batch_beam_count, FFTOutputMode mode) {
  size_t input_size = batch_beam_count * n_point_ * sizeof(std::complex<float>);
  size_t fft_size = batch_beam_count * nFFT_ * sizeof(std::complex<float>);

  bufs_.Require(kInputBuf, input_size);
  bufs_.Require(kFftBuf, fft_size);  // in-place: pad + FFT in same buffer

  if (mode != FFTOutputMode::COMPLEX) {
    size_t mp_size = batch_beam_count * nFFT_ * 2 * sizeof(float);
    bufs_.Require(kMagPhaseInterleaved, mp_size);
    has_mag_phase_buffers_ = true;
  }
}

// =========================================================================
// hipFFT plan management (LRU-2 cache)
// =========================================================================

void FFTProcessorROCm::CreateFFTPlan(size_t batch_beam_count) {
  if (plan_created_ && plan_batch_size_ == batch_beam_count) return;

  // LRU-2: swap with secondary cache if it matches
  if (plan_last_ != 0 && plan_last_batch_ == batch_beam_count) {
    std::swap(plan_, plan_last_);
    std::swap(plan_batch_size_, plan_last_batch_);
    plan_created_ = true;
    return;
  }

  // Evict secondary, move current to secondary
  if (plan_last_ != 0) {
    hipfftDestroy(plan_last_);
    plan_last_ = 0;
    plan_last_batch_ = 0;
  }
  if (plan_created_) {
    plan_last_ = plan_;
    plan_last_batch_ = plan_batch_size_;
    plan_ = 0;
    plan_created_ = false;
  }

  hipfftResult result = hipfftPlan1d(&plan_, static_cast<int>(nFFT_),
                                      HIPFFT_C2C,
                                      static_cast<int>(batch_beam_count));
  if (result != HIPFFT_SUCCESS) {
    throw std::runtime_error("CreateFFTPlan: hipfftPlan1d failed: " +
                              std::to_string(static_cast<int>(result)));
  }

  result = hipfftSetStream(plan_, ctx_.stream());
  if (result != HIPFFT_SUCCESS) {
    hipfftDestroy(plan_);
    plan_ = 0;
    throw std::runtime_error("CreateFFTPlan: hipfftSetStream failed");
  }

  plan_batch_size_ = batch_beam_count;
  plan_created_ = true;
}

void FFTProcessorROCm::ReleasePlans() {
  if (plan_created_) { hipfftDestroy(plan_); plan_ = 0; plan_created_ = false; }
  if (plan_last_ != 0) { hipfftDestroy(plan_last_); plan_last_ = 0; plan_last_batch_ = 0; }
}

// =========================================================================
// Data transfer
// =========================================================================

void FFTProcessorROCm::UploadData(const std::complex<float>* data, size_t count) {
  hipError_t err = hipMemcpyHtoDAsync(
      bufs_.Get(kInputBuf),
      const_cast<std::complex<float>*>(data),
      count * sizeof(std::complex<float>), ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("UploadData: " + std::string(hipGetErrorString(err)));
  }
}

void FFTProcessorROCm::CopyGpuData(void* src, size_t src_offset_bytes, size_t count) {
  char* src_ptr = static_cast<char*>(src) + src_offset_bytes;
  hipError_t err = hipMemcpyDtoDAsync(
      bufs_.Get(kInputBuf), src_ptr,
      count * sizeof(std::complex<float>), ctx_.stream());
  if (err != hipSuccess) {
    throw std::runtime_error("CopyGpuData: " + std::string(hipGetErrorString(err)));
  }
}

// =========================================================================
// Read results
// =========================================================================

std::vector<FFTComplexResult> FFTProcessorROCm::ReadComplexResults(
    size_t beam_count, size_t start_beam, float sample_rate) {
  size_t total = beam_count * nFFT_;
  std::vector<std::complex<float>> raw(total);

  hipError_t err = hipMemcpyDtoH(raw.data(), bufs_.Get(kFftBuf),
                                   total * sizeof(std::complex<float>));
  if (err != hipSuccess) {
    throw std::runtime_error("ReadComplexResults: " + std::string(hipGetErrorString(err)));
  }

  std::vector<FFTComplexResult> results;
  results.reserve(beam_count);
  for (size_t i = 0; i < beam_count; ++i) {
    FFTComplexResult r;
    r.beam_id = static_cast<uint32_t>(start_beam + i);
    r.nFFT = nFFT_;
    r.sample_rate = sample_rate;
    r.spectrum.assign(raw.begin() + i * nFFT_, raw.begin() + (i + 1) * nFFT_);
    results.push_back(std::move(r));
  }
  return results;
}

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ReadMagPhaseResults(
    size_t beam_count, size_t start_beam, float sample_rate, bool include_freq) {
  size_t total = beam_count * nFFT_;
  std::vector<float> raw(total * 2);

  hipError_t err = hipMemcpyDtoH(raw.data(), bufs_.Get(kMagPhaseInterleaved),
                                   total * 2 * sizeof(float));
  if (err != hipSuccess) {
    throw std::runtime_error("ReadMagPhaseResults: " + std::string(hipGetErrorString(err)));
  }

  std::vector<FFTMagPhaseResult> results;
  results.reserve(beam_count);
  for (size_t i = 0; i < beam_count; ++i) {
    FFTMagPhaseResult r;
    r.beam_id = static_cast<uint32_t>(start_beam + i);
    r.nFFT = nFFT_;
    r.sample_rate = sample_rate;
    r.magnitude.resize(nFFT_);
    r.phase.resize(nFFT_);

    for (uint32_t k = 0; k < nFFT_; ++k) {
      size_t idx = (i * nFFT_ + k) * 2;
      r.magnitude[k] = raw[idx];
      r.phase[k] = raw[idx + 1];
    }

    if (include_freq) {
      r.frequency.resize(nFFT_);
      float freq_step = sample_rate / static_cast<float>(nFFT_);
      for (uint32_t k = 0; k < nFFT_; ++k) {
        r.frequency[k] = static_cast<float>(k) * freq_step;
      }
    }

    results.push_back(std::move(r));
  }
  return results;
}

// =========================================================================
// Public API — ProcessComplex
// =========================================================================

std::vector<FFTComplexResult> FFTProcessorROCm::ProcessComplex(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ProcessComplex: input size " + std::to_string(data.size()) +
                                 " != expected " + std::to_string(expected));
  }

  CalculateNFFT(params);
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam(FFTOutputMode::COMPLEX);
  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit);

  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  std::vector<FFTComplexResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count, FFTOutputMode::COMPLEX);
    CreateFFTPlan(batch.count);

    hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
    hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
    hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
    if (prof_events) {
      hipEventCreate(&ev_up_s); hipEventCreate(&ev_up_e);
      hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
      hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
    }

    const auto* batch_data = data.data() + batch.start * params.n_point;
    if (prof_events) hipEventRecord(ev_up_s, ctx_.stream());
    UploadData(batch_data, batch.count * params.n_point);
    if (prof_events) hipEventRecord(ev_up_e, ctx_.stream());

    if (prof_events) hipEventRecord(ev_pad_s, ctx_.stream());
    pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                    batch.count, n_point_, nFFT_);
    if (prof_events) hipEventRecord(ev_pad_e, ctx_.stream());

    if (prof_events) hipEventRecord(ev_fft_s, ctx_.stream());
    hipfftResult fft_result = hipfftExecC2C(
        plan_,
        static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
        static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
        HIPFFT_FORWARD);
    if (fft_result != HIPFFT_SUCCESS) {
      throw std::runtime_error("hipfftExecC2C failed: " +
                                std::to_string(static_cast<int>(fft_result)));
    }
    if (prof_events) hipEventRecord(ev_fft_e, ctx_.stream());

    hipStreamSynchronize(ctx_.stream());

    if (prof_events) {
      prof_events->push_back({"Upload", MakeROCmDataFromEvents(ev_up_s, ev_up_e, 1, "H2D copy")});
      prof_events->push_back({"Pad", MakeROCmDataFromEvents(ev_pad_s, ev_pad_e, 0, "pad kernel")});
      prof_events->push_back({"FFT", MakeROCmDataFromEvents(ev_fft_s, ev_fft_e, 0, "hipfftExecC2C")});
    }

    auto t_dl_start = std::chrono::high_resolution_clock::now();
    auto batch_results = ReadComplexResults(batch.count, batch.start, params.sample_rate);
    auto t_dl_end = std::chrono::high_resolution_clock::now();
    if (prof_events) {
      prof_events->push_back({"Download", MakeROCmDataFromClock(t_dl_start, t_dl_end, 1, "D2H copy")});
    }

    for (auto& r : batch_results) all_results.push_back(std::move(r));
  }
  return all_results;
}

std::vector<FFTComplexResult> FFTProcessorROCm::ProcessComplex(
    void* gpu_data, const FFTProcessorParams& params, size_t gpu_memory_bytes) {
  if (!gpu_data) throw std::invalid_argument("ProcessComplex: gpu_data is null");

  CalculateNFFT(params);
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam(FFTOutputMode::COMPLEX);
  size_t external_memory = (gpu_memory_bytes > 0)
      ? gpu_memory_bytes
      : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  std::vector<FFTComplexResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count, FFTOutputMode::COMPLEX);
    CreateFFTPlan(batch.count);

    size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
    CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

    pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                    batch.count, n_point_, nFFT_);

    hipfftResult fft_res = hipfftExecC2C(plan_,
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  HIPFFT_FORWARD);
    if (fft_res != HIPFFT_SUCCESS) {
      throw std::runtime_error("ProcessComplex(GPU): hipfftExecC2C failed: " +
                                std::to_string(static_cast<int>(fft_res)));
    }
    hipStreamSynchronize(ctx_.stream());

    auto batch_results = ReadComplexResults(batch.count, batch.start, params.sample_rate);
    for (auto& r : batch_results) all_results.push_back(std::move(r));
  }
  return all_results;
}

// =========================================================================
// Public API — ProcessMagPhase
// =========================================================================

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ProcessMagPhase(
    const std::vector<std::complex<float>>& data,
    const FFTProcessorParams& params,
    ROCmProfEvents* prof_events) {
  size_t expected = static_cast<size_t>(params.beam_count) * params.n_point;
  if (data.size() != expected) {
    throw std::invalid_argument("ProcessMagPhase: input size mismatch");
  }

  CalculateNFFT(params);
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam(params.output_mode);
  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit);

  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  bool include_freq = (params.output_mode == FFTOutputMode::MAGNITUDE_PHASE_FREQ);

  std::vector<FFTMagPhaseResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count, params.output_mode);
    CreateFFTPlan(batch.count);

    hipEvent_t ev_up_s = nullptr, ev_up_e = nullptr;
    hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
    hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
    hipEvent_t ev_mag_s = nullptr, ev_mag_e = nullptr;
    if (prof_events) {
      hipEventCreate(&ev_up_s); hipEventCreate(&ev_up_e);
      hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
      hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
      hipEventCreate(&ev_mag_s); hipEventCreate(&ev_mag_e);
    }

    const auto* batch_data = data.data() + batch.start * params.n_point;
    if (prof_events) hipEventRecord(ev_up_s, ctx_.stream());
    UploadData(batch_data, batch.count * params.n_point);
    if (prof_events) hipEventRecord(ev_up_e, ctx_.stream());

    if (prof_events) hipEventRecord(ev_pad_s, ctx_.stream());
    pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                    batch.count, n_point_, nFFT_);
    if (prof_events) hipEventRecord(ev_pad_e, ctx_.stream());

    if (prof_events) hipEventRecord(ev_fft_s, ctx_.stream());
    hipfftResult fft_res_mp = hipfftExecC2C(plan_,
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  HIPFFT_FORWARD);
    if (fft_res_mp != HIPFFT_SUCCESS) {
      throw std::runtime_error("ProcessMagPhase: hipfftExecC2C failed: " +
                                std::to_string(static_cast<int>(fft_res_mp)));
    }
    if (prof_events) hipEventRecord(ev_fft_e, ctx_.stream());

    if (prof_events) hipEventRecord(ev_mag_s, ctx_.stream());
    mag_phase_op_.Execute(bufs_.Get(kFftBuf), bufs_.Get(kMagPhaseInterleaved),
                          batch.count * nFFT_);
    if (prof_events) hipEventRecord(ev_mag_e, ctx_.stream());

    hipStreamSynchronize(ctx_.stream());

    if (prof_events) {
      prof_events->push_back({"Upload", MakeROCmDataFromEvents(ev_up_s, ev_up_e, 1, "H2D copy")});
      prof_events->push_back({"Pad", MakeROCmDataFromEvents(ev_pad_s, ev_pad_e, 0, "pad kernel")});
      prof_events->push_back({"FFT", MakeROCmDataFromEvents(ev_fft_s, ev_fft_e, 0, "hipfftExecC2C")});
      prof_events->push_back({"MagPhase", MakeROCmDataFromEvents(ev_mag_s, ev_mag_e, 0, "mag_phase kernel")});
    }

    auto t_dl_start = std::chrono::high_resolution_clock::now();
    auto batch_results = ReadMagPhaseResults(batch.count, batch.start,
                                              params.sample_rate, include_freq);
    auto t_dl_end = std::chrono::high_resolution_clock::now();
    if (prof_events) {
      prof_events->push_back({"Download", MakeROCmDataFromClock(t_dl_start, t_dl_end, 1, "D2H copy")});
    }

    for (auto& r : batch_results) all_results.push_back(std::move(r));
  }
  return all_results;
}

std::vector<FFTMagPhaseResult> FFTProcessorROCm::ProcessMagPhase(
    void* gpu_data, const FFTProcessorParams& params, size_t gpu_memory_bytes) {
  if (!gpu_data) throw std::invalid_argument("ProcessMagPhase: gpu_data is null");

  CalculateNFFT(params);
  EnsureCompiled();

  size_t bytes_per_beam = CalculateBytesPerBeam(params.output_mode);
  size_t external_memory = (gpu_memory_bytes > 0)
      ? gpu_memory_bytes
      : static_cast<size_t>(params.beam_count) * params.n_point * sizeof(std::complex<float>);

  size_t optimal_batch = drv_gpu_lib::BatchManager::CalculateOptimalBatchSize(
      ctx_.backend(), params.beam_count, bytes_per_beam, params.memory_limit, external_memory);

  auto batches = drv_gpu_lib::BatchManager::CreateBatches(
      params.beam_count, optimal_batch, 3, true);

  bool include_freq = (params.output_mode == FFTOutputMode::MAGNITUDE_PHASE_FREQ);

  std::vector<FFTMagPhaseResult> all_results;
  all_results.reserve(params.beam_count);

  for (const auto& batch : batches) {
    AllocateBuffers(batch.count, params.output_mode);
    CreateFFTPlan(batch.count);

    size_t src_offset = batch.start * params.n_point * sizeof(std::complex<float>);
    CopyGpuData(gpu_data, src_offset, batch.count * params.n_point);

    pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                    batch.count, n_point_, nFFT_);
    hipfftResult fft_res = hipfftExecC2C(plan_,
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
                  HIPFFT_FORWARD);
    if (fft_res != HIPFFT_SUCCESS) {
      throw std::runtime_error("ProcessMagPhase(GPU): hipfftExecC2C failed: " +
                                std::to_string(static_cast<int>(fft_res)));
    }
    mag_phase_op_.Execute(bufs_.Get(kFftBuf), bufs_.Get(kMagPhaseInterleaved),
                          batch.count * nFFT_);
    hipStreamSynchronize(ctx_.stream());

    auto batch_results = ReadMagPhaseResults(batch.count, batch.start,
                                              params.sample_rate, include_freq);
    for (auto& r : batch_results) all_results.push_back(std::move(r));
  }
  return all_results;
}

// =========================================================================
// Public API — ProcessMagnitudesToGPU (SNR_04)
// =========================================================================
//
// Записывает |X| или |X|² прямо в caller-provided GPU буфер, без D2H.
// Используется SnrEstimatorOp — полный pipeline gather → FFT → CFAR на GPU.

void FFTProcessorROCm::ProcessMagnitudesToGPU(
    void* gpu_data,
    void* gpu_out_magnitudes,
    const FFTProcessorParams& params,
    bool squared,
    WindowType window,
    ROCmProfEvents* prof_events)
{
  if (!gpu_data) {
    throw std::invalid_argument("ProcessMagnitudesToGPU: gpu_data is null");
  }
  if (!gpu_out_magnitudes) {
    throw std::invalid_argument("ProcessMagnitudesToGPU: gpu_out_magnitudes is null");
  }

  // 1. nFFT + kernels compilation
  CalculateNFFT(params);
  EnsureCompiled();

  // 2. Allocate pipeline buffers. Используем COMPLEX mode — kMagPhaseInterleaved
  //    не выделяется, выход идёт напрямую в caller-provided gpu_out_magnitudes.
  AllocateBuffers(params.beam_count, FFTOutputMode::COMPLEX);

  // 3. hipFFT plan (LRU-2 cache)
  CreateFFTPlan(params.beam_count);

  // 4. Copy input → kInputBuf (D2D)
  CopyGpuData(gpu_data, 0,
              static_cast<size_t>(params.beam_count) * params.n_point);

  // Optional profiling events
  hipEvent_t ev_pad_s = nullptr, ev_pad_e = nullptr;
  hipEvent_t ev_fft_s = nullptr, ev_fft_e = nullptr;
  hipEvent_t ev_mag_s = nullptr, ev_mag_e = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_pad_s); hipEventCreate(&ev_pad_e);
    hipEventCreate(&ev_fft_s); hipEventCreate(&ev_fft_e);
    hipEventCreate(&ev_mag_s); hipEventCreate(&ev_mag_e);
  }

  // 5. Pad (+ optional window) → kFftBuf
  if (prof_events) hipEventRecord(ev_pad_s, ctx_.stream());
  pad_op_.Execute(bufs_.Get(kInputBuf), bufs_.Get(kFftBuf),
                  params.beam_count, n_point_, nFFT_,
                  window);  // SNR_02b: window function (default None)
  if (prof_events) hipEventRecord(ev_pad_e, ctx_.stream());

  // 6. FFT in-place in kFftBuf
  if (prof_events) hipEventRecord(ev_fft_s, ctx_.stream());
  hipfftResult fft_err = hipfftExecC2C(
      plan_,
      static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
      static_cast<hipfftComplex*>(bufs_.Get(kFftBuf)),
      HIPFFT_FORWARD);
  if (fft_err != HIPFFT_SUCCESS) {
    throw std::runtime_error("ProcessMagnitudesToGPU: hipfftExecC2C failed: " +
                              std::to_string(static_cast<int>(fft_err)));
  }
  if (prof_events) hipEventRecord(ev_fft_e, ctx_.stream());

  // 7. Magnitude (|X| или |X|²) → caller-provided gpu_out_magnitudes
  //    inv_n = 1.0f: raw output, caller сам нормирует если нужно.
  //    Для CFAR SNR-estimator нормировка не нужна — ratio сокращает её.
  size_t total = static_cast<size_t>(params.beam_count) * nFFT_;
  if (prof_events) hipEventRecord(ev_mag_s, ctx_.stream());
  mag_op_.Execute(bufs_.Get(kFftBuf), gpu_out_magnitudes,
                  total, 1.0f, squared);
  if (prof_events) hipEventRecord(ev_mag_e, ctx_.stream());

  hipStreamSynchronize(ctx_.stream());

  if (prof_events) {
    prof_events->push_back({"Pad",
        MakeROCmDataFromEvents(ev_pad_s, ev_pad_e, 0, "pad (+window)")});
    prof_events->push_back({"FFT",
        MakeROCmDataFromEvents(ev_fft_s, ev_fft_e, 0, "hipfftExecC2C")});
    prof_events->push_back({"Magnitude",
        MakeROCmDataFromEvents(ev_mag_s, ev_mag_e, 0,
                               squared ? "complex_to_magnitude_squared"
                                       : "complex_to_magnitude")});
  }
}

// =========================================================================
// Utilities
// =========================================================================

uint32_t FFTProcessorROCm::NextPowerOf2(uint32_t n) {
  if (n == 0) return 1;
  n--;
  n |= n >> 1;  n |= n >> 2;  n |= n >> 4;
  n |= n >> 8;  n |= n >> 16;
  return n + 1;
}

void FFTProcessorROCm::CalculateNFFT(const FFTProcessorParams& params) {
  n_point_ = params.n_point;
  uint32_t base_fft = NextPowerOf2(params.n_point);
  nFFT_ = base_fft * params.repeat_count;
}

size_t FFTProcessorROCm::CalculateBytesPerBeam(FFTOutputMode mode) const {
  size_t input_bytes = n_point_ * sizeof(std::complex<float>);
  // In-place FFT: single buffer for pad + FFT result (was 2 buffers before)
  size_t fft_bytes = nFFT_ * sizeof(std::complex<float>);
  size_t post_bytes = (mode != FFTOutputMode::COMPLEX)
      ? 2 * nFFT_ * sizeof(float)
      : 0;
  return input_bytes + fft_bytes + post_bytes;
}

FFTProfilingData FFTProcessorROCm::GetProfilingData() const {
  FFTProfilingData out{};
  const int gpu_id = ctx_.backend() ? ctx_.backend()->GetDeviceIndex() : 0;
  auto stats = drv_gpu_lib::GPUProfiler::GetInstance().GetStats(gpu_id);
  auto it = stats.find("FFTProcessorROCm");
  if (it == stats.end()) return out;

  const auto& mod = it->second;
  auto ev = [&mod](const char* name) -> double {
    auto e = mod.events.find(name);
    return (e != mod.events.end()) ? e->second.GetAvgTimeMs() : 0.0;
  };
  out.upload_time_ms = ev("Upload");
  out.fft_time_ms = ev("FFT");
  out.post_processing_time_ms = ev("PostProcessing");
  out.download_time_ms = ev("Download");
  out.total_time_ms = mod.GetTotalTimeMs();
  return out;
}

}  // namespace fft_processor

#endif  // ENABLE_ROCM
