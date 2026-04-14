/**
 * @file lch_farrow_rocm.cpp
 * @brief LchFarrowROCm - fractional delay processor (Lagrange 48x5) on ROCm/HIP
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * CONTENTS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PART 1: Built-in Lagrange matrix 48x5
 * PART 2: Constructor, Destructor, Move Semantics
 * PART 3: Setters (SetDelays, SetSampleRate, SetNoise, LoadMatrix)
 * PART 4: GPU Processing (Process, ProcessFromCPU)
 * PART 5: CPU Processing (ProcessCpu — reference)
 * PART 6: GPU Internals (CompileKernel, UploadMatrix, ReleaseGpuResources)
 *
 * Key differences from OpenCL version:
 * - hiprtc for runtime kernel compilation
 * - void* device pointers instead of cl_mem
 * - hipModuleLaunchKernel for dispatch
 * - hipStream_t from IBackend::GetNativeQueue()
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-23
 */

#if ENABLE_ROCM

#include "lch_farrow_rocm.hpp"
#include "kernels/lch_farrow_kernels_rocm.hpp"
#include <spectrum/utils/rocm_profiling_helpers.hpp>
#include <core/services/console_output.hpp>
#include <core/services/profiling_types.hpp>

#include <stdexcept>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lch_farrow {

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: Built-in Lagrange matrix 48x5
// ═══════════════════════════════════════════════════════════════════════════

static const float kBuiltinLagrangeMatrix[48 * 5] = {
  // Row 0  (mu = 0/48)
   0.0f,     1.0f,     0.0f,     0.0f,     0.0f,
  // Row 1  (mu = 1/48)
  -0.0052f,  1.0417f, -0.0417f,  0.0052f,  0.0f,
  // Row 2
  -0.01f,    1.08f,   -0.08f,    0.01f,    0.0f,
  // Row 3
  -0.0143f,  1.1143f, -0.1143f,  0.0143f,  0.0f,
  // Row 4
  -0.018f,   1.144f,  -0.144f,   0.018f,   0.0f,
  // Row 5
  -0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f,
  // Row 6
  -0.0228f,  1.1827f, -0.1827f,  0.0228f,  0.0f,
  // Row 7
  -0.0239f,  1.1914f, -0.1914f,  0.0239f,  0.0f,
  // Row 8
  -0.024f,   1.2f,    -0.2f,     0.024f,   0.0f,
  // Row 9
  -0.0231f,  1.1914f, -0.1914f,  0.0231f,  0.0f,
  // Row 10
  -0.0208f,  1.1667f, -0.1667f,  0.0208f,  0.0f,
  // Row 11
  -0.0169f,  1.1198f, -0.1198f,  0.0169f,  0.0f,
  // Row 12
  -0.0111f,  1.0432f, -0.0432f,  0.0111f,  0.0f,
  // Row 13
   0.0026f,  0.9323f,  0.0677f, -0.0026f,  0.0f,
  // Row 14
   0.0229f,  0.7812f,  0.2188f, -0.0229f,  0.0f,
  // Row 15
   0.0507f,  0.5823f,  0.4177f, -0.0507f,  0.0f,
  // Row 16
   0.0859f,  0.3281f,  0.6719f, -0.0859f,  0.0f,
  // Row 17
   0.1276f,  0.0104f,  0.9896f, -0.1276f,  0.0f,
  // Row 18
   0.175f,  -0.3802f,  1.3802f, -0.175f,   0.0f,
  // Row 19
   0.2274f, -0.8385f,  1.8385f, -0.2274f,  0.0f,
  // Row 20
   0.2839f, -1.3567f,  2.3567f, -0.2839f,  0.0f,
  // Row 21
   0.3438f, -1.9375f,  2.9375f, -0.3438f,  0.0f,
  // Row 22
   0.4063f, -2.5846f,  3.5846f, -0.4063f,  0.0f,
  // Row 23
   0.4705f, -3.2917f,  4.2917f, -0.4705f,  0.0f,
  // Row 24
   0.5355f, -4.0521f,  5.0521f, -0.5355f,  0.0f,
  // Row 25
   0.6f,    -4.8594f,  5.8594f, -0.6f,     0.0f,
  // Row 26
   0.6628f, -5.7083f,  6.7083f, -0.6628f,  0.0f,
  // Row 27
   0.7227f, -6.592f,   7.592f,  -0.7227f,  0.0f,
  // Row 28
   0.7786f, -7.5052f,  8.5052f, -0.7786f,  0.0f,
  // Row 29
   0.8293f, -8.4411f,  9.4411f, -0.8293f,  0.0f,
  // Row 30
   0.8734f, -9.3937f, 10.3937f, -0.8734f,  0.0f,
  // Row 31
   0.9102f,-10.3567f, 11.3567f, -0.9102f,  0.0f,
  // Row 32
   0.9384f,-11.3229f, 12.3229f, -0.9384f,  0.0f,
  // Row 33
   0.957f, -12.2857f, 13.2857f, -0.957f,   0.0f,
  // Row 34
   0.9648f,-13.2386f, 14.2386f, -0.9648f,  0.0f,
  // Row 35
   0.9609f,-14.1748f, 15.1748f, -0.9609f,  0.0f,
  // Row 36
   0.9446f,-14.9877f, 16.0f,    -0.9446f,  0.0f,
  // Row 37
   0.9141f,-15.6684f, 16.75f,   -0.9141f,  0.0f,
  // Row 38
   0.8684f,-16.2096f, 17.4219f, -0.8684f,  0.0f,
  // Row 39
   0.8066f,-16.6039f, 18.0078f, -0.8066f,  0.0f,
  // Row 40
   0.7275f,-16.8438f, 18.5f,    -0.7275f,  0.0f,
  // Row 41
   0.6299f,-16.9219f, 19.0f,    -0.6299f,  0.0f,
  // Row 42
   0.5126f,-16.8301f, 19.3984f, -0.5126f,  0.0f,
  // Row 43
   0.3745f,-16.5605f, 19.6875f, -0.3745f,  0.0f,
  // Row 44
   0.2143f,-16.0955f, 19.875f,  -0.2143f,  0.0f,
  // Row 45
   0.0307f,-15.4175f, 20.0f,    -0.0307f,  0.0f,
  // Row 46
  -0.1816f,-14.5086f, 20.0234f,  0.1816f,  0.0f,
  // Row 47
  -0.4347f,-13.3521f, 20.0547f,  0.4347f,  0.0f
};

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: Constructor, Destructor, Move Semantics
// ═══════════════════════════════════════════════════════════════════════════

static const std::vector<std::string> kKernelNames = {
  "lch_farrow_delay"
};

LchFarrowROCm::LchFarrowROCm(drv_gpu_lib::IBackend* backend)
    : ctx_(backend, "LchFarrow", "modules/lch_farrow/kernels") {
  // Load built-in matrix
  lagrange_matrix_.assign(kBuiltinLagrangeMatrix,
                          kBuiltinLagrangeMatrix + 48 * 5);
  matrix_loaded_ = true;

  EnsureCompiled();
  UploadMatrix();
}

LchFarrowROCm::~LchFarrowROCm() {
  ReleaseGpuResources();
}

LchFarrowROCm::LchFarrowROCm(LchFarrowROCm&& other) noexcept
    : ctx_(std::move(other.ctx_))
    , delay_us_(std::move(other.delay_us_))
    , sample_rate_(other.sample_rate_)
    , noise_amplitude_(other.noise_amplitude_)
    , norm_val_(other.norm_val_)
    , noise_seed_(other.noise_seed_)
    , lagrange_matrix_(std::move(other.lagrange_matrix_))
    , matrix_loaded_(other.matrix_loaded_)
    , compiled_(other.compiled_)
    , matrix_buf_(other.matrix_buf_)
    , delay_buf_(other.delay_buf_)
    , delay_buf_size_(other.delay_buf_size_) {
  other.compiled_ = false;
  other.matrix_buf_ = nullptr;
  other.delay_buf_ = nullptr;
  other.delay_buf_size_ = 0;
}

LchFarrowROCm& LchFarrowROCm::operator=(LchFarrowROCm&& other) noexcept {
  if (this != &other) {
    ReleaseGpuResources();
    ctx_ = std::move(other.ctx_);
    delay_us_ = std::move(other.delay_us_);
    sample_rate_ = other.sample_rate_;
    noise_amplitude_ = other.noise_amplitude_;
    norm_val_ = other.norm_val_;
    noise_seed_ = other.noise_seed_;
    lagrange_matrix_ = std::move(other.lagrange_matrix_);
    matrix_loaded_ = other.matrix_loaded_;
    compiled_ = other.compiled_;
    matrix_buf_ = other.matrix_buf_;
    delay_buf_ = other.delay_buf_;
    delay_buf_size_ = other.delay_buf_size_;
    other.compiled_ = false;
    other.matrix_buf_ = nullptr;
    other.delay_buf_ = nullptr;
    other.delay_buf_size_ = 0;
  }
  return *this;
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 3: Setters
// ═══════════════════════════════════════════════════════════════════════════

void LchFarrowROCm::SetDelays(const std::vector<float>& delay_us) {
  delay_us_ = delay_us;
}

void LchFarrowROCm::SetSampleRate(float sample_rate) {
  sample_rate_ = sample_rate;
}

void LchFarrowROCm::SetNoise(float noise_amplitude, float norm_val,
                              uint32_t noise_seed) {
  noise_amplitude_ = noise_amplitude;
  norm_val_ = norm_val;
  noise_seed_ = noise_seed;
}

void LchFarrowROCm::LoadMatrix(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "LchFarrowROCm::LoadMatrix: cannot open " + json_path);
  }

  std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

  lagrange_matrix_.clear();

  auto data_pos = content.find("\"data\"");
  if (data_pos == std::string::npos) {
    throw std::runtime_error(
        "LchFarrowROCm::LoadMatrix: no 'data' key in JSON");
  }

  auto arr_start = content.find('[', data_pos);
  if (arr_start == std::string::npos) {
    throw std::runtime_error("LchFarrowROCm::LoadMatrix: malformed JSON");
  }

  std::string nums;
  for (size_t i = arr_start; i < content.size(); ++i) {
    char c = content[i];
    if (c == '-' || c == '.' || (c >= '0' && c <= '9') ||
        c == 'e' || c == 'E' || c == '+') {
      nums += c;
    } else if (!nums.empty()) {
      lagrange_matrix_.push_back(std::stof(nums));
      nums.clear();
    }
  }
  if (!nums.empty()) {
    lagrange_matrix_.push_back(std::stof(nums));
  }

  if (lagrange_matrix_.size() != 48 * 5) {
    throw std::runtime_error(
        "LchFarrowROCm::LoadMatrix: expected 240 values, got "
        + std::to_string(lagrange_matrix_.size()));
  }

  matrix_loaded_ = true;
  UploadMatrix();
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 4: GPU Processing
// ═══════════════════════════════════════════════════════════════════════════

using fft_func_utils::MakeROCmDataFromEvents;
using fft_func_utils::MakeROCmDataFromClock;

drv_gpu_lib::InputData<void*>
LchFarrowROCm::Process(void* input_ptr, uint32_t antennas, uint32_t points,
                        ROCmProfEvents* prof_events) {
  if (!input_ptr) {
    throw std::invalid_argument("LchFarrowROCm::Process: input_ptr is null");
  }
  if (antennas == 0 || points == 0) {
    throw std::runtime_error("LchFarrowROCm::Process: antennas or points is 0");
  }

  if (delay_us_.empty()) {
    delay_us_.assign(antennas, 0.0f);
  }

  if (delay_us_.size() != antennas) {
    throw std::invalid_argument(
        "LchFarrowROCm: delay_us.size()=" + std::to_string(delay_us_.size())
        + " != antennas=" + std::to_string(antennas));
  }

  size_t total_points = static_cast<size_t>(antennas) * points;
  size_t buffer_size = total_points * sizeof(std::complex<float>);
  hipError_t err;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "LchFarrow[ROCm]", "Process: " + std::to_string(antennas) +
      " antennas, " + std::to_string(points) + " points");

  // Allocate output buffer
  void* output_ptr = nullptr;
  err = hipMalloc(&output_ptr, buffer_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "LchFarrowROCm::Process: hipMalloc(output) failed: " +
        std::string(hipGetErrorString(err)));
  }

  // Persistent delay_buf: resize only when needed (avoid hipMalloc/Free per call)
  size_t delay_size = delay_us_.size() * sizeof(float);
  if (delay_buf_size_ < delay_size) {
    if (delay_buf_) (void)hipFree(delay_buf_);
    err = hipMalloc(&delay_buf_, delay_size);
    if (err != hipSuccess) {
      delay_buf_ = nullptr;
      delay_buf_size_ = 0;
      (void)hipFree(output_ptr);
      throw std::runtime_error(
          "LchFarrowROCm::Process: hipMalloc(delay) failed");
    }
    delay_buf_size_ = delay_size;
  }

  // ── Upload_delay (async) ─────────────────────────────────────────
  hipEvent_t ev_up_start = nullptr, ev_up_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_up_start);
    hipEventCreate(&ev_up_end);
    hipEventRecord(ev_up_start, ctx_.stream());
  }

  err = hipMemcpyHtoDAsync(delay_buf_, delay_us_.data(), delay_size, ctx_.stream());
  if (err != hipSuccess) {
    if (ev_up_start) { hipEventDestroy(ev_up_start); hipEventDestroy(ev_up_end); }
    (void)hipFree(output_ptr);
    throw std::runtime_error(
        "LchFarrowROCm::Process: hipMemcpyHtoDAsync(delay) failed");
  }

  if (prof_events) {
    hipEventRecord(ev_up_end, ctx_.stream());
  }

  // ── Kernel arguments ──────────────────────────────────────────────
  unsigned int ant = antennas;
  unsigned int pts = points;
  float sr = sample_rate_;
  float na = noise_amplitude_;
  float nv = norm_val_;
  uint32_t ns = noise_seed_;
  if (ns == 0 && na > 0.0f) {
    ns = static_cast<uint32_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()
        & 0xFFFFFFFF);
  }

  void* args[] = {
    &input_ptr,
    &output_ptr,
    &matrix_buf_,
    &delay_buf_,
    &ant,
    &pts,
    &sr,
    &na,
    &nv,
    &ns
  };

  // ── Kernel (async) ────────────────────────────────────────────────
  hipEvent_t ev_k_start = nullptr, ev_k_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_k_start);
    hipEventCreate(&ev_k_end);
    hipEventRecord(ev_k_start, ctx_.stream());
  }

  // 2D grid: X=samples (ceil), Y=antennas — eliminates div/mod in kernel
  unsigned int grid_x = static_cast<unsigned int>(
      (points + kBlockSize - 1) / kBlockSize);
  unsigned int grid_y = antennas;

  err = hipModuleLaunchKernel(
      ctx_.GetKernel("lch_farrow_delay"),
      grid_x, grid_y, 1,
      kBlockSize, 1, 1,
      0, ctx_.stream(),
      args, nullptr);

  if (err != hipSuccess) {
    if (ev_up_start) { hipEventDestroy(ev_up_start); hipEventDestroy(ev_up_end); }
    if (ev_k_start)  { hipEventDestroy(ev_k_start);  hipEventDestroy(ev_k_end);  }
    (void)hipFree(output_ptr);
    // delay_buf_ — persistent буфер, не освобождаем здесь (живёт между вызовами Process)
    throw std::runtime_error(
        "LchFarrowROCm::Process: hipModuleLaunchKernel failed: " +
        std::string(hipGetErrorString(err)));
  }

  if (prof_events) {
    hipEventRecord(ev_k_end, ctx_.stream());
  }

  (void)hipStreamSynchronize(ctx_.stream());

  // delay_buf_ is persistent — no free here

  // ── Собрать prof_events (stream синхронизирован — данные готовы) ──
  if (prof_events) {
    prof_events->push_back({"Upload_delay",
        MakeROCmDataFromEvents(ev_up_start, ev_up_end, 1, "H2D_delay")});
    prof_events->push_back({"Kernel",
        MakeROCmDataFromEvents(ev_k_start, ev_k_end, 0, "lch_farrow_delay")});
  }

  drv_gpu_lib::InputData<void*> result;
  result.antenna_count = antennas;
  result.n_point       = points;
  result.data          = output_ptr;
  result.gpu_memory_bytes = buffer_size;
  result.sample_rate   = sample_rate_;
  return result;
}

drv_gpu_lib::InputData<void*>
LchFarrowROCm::ProcessFromCPU(
    const std::vector<std::complex<float>>& data,
    uint32_t antennas, uint32_t points,
    ROCmProfEvents* prof_events)
{
  size_t expected = static_cast<size_t>(antennas) * points;
  if (data.size() != expected) {
    throw std::invalid_argument(
        "LchFarrowROCm::ProcessFromCPU: input size " +
        std::to_string(data.size()) + " != expected " +
        std::to_string(expected));
  }

  // Upload input data to GPU
  size_t data_size = expected * sizeof(std::complex<float>);
  void* input_ptr = nullptr;
  hipError_t err = hipMalloc(&input_ptr, data_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "LchFarrowROCm::ProcessFromCPU: hipMalloc(input) failed");
  }

  // ── Upload_input (async) ─────────────────────────────────────────
  hipEvent_t ev_in_start = nullptr, ev_in_end = nullptr;
  if (prof_events) {
    hipEventCreate(&ev_in_start);
    hipEventCreate(&ev_in_end);
    hipEventRecord(ev_in_start, ctx_.stream());
  }

  err = hipMemcpyHtoDAsync(input_ptr,
                            const_cast<std::complex<float>*>(data.data()),
                            data_size, ctx_.stream());
  if (err != hipSuccess) {
    if (ev_in_start) { hipEventDestroy(ev_in_start); hipEventDestroy(ev_in_end); }
    (void)hipFree(input_ptr);
    throw std::runtime_error(
        "LchFarrowROCm::ProcessFromCPU: hipMemcpyHtoDAsync(input) failed");
  }

  if (prof_events) {
    hipEventRecord(ev_in_end, ctx_.stream());
  }

  // Process on GPU (Upload_delay + Kernel — добавляются в prof_events через Process)
  auto result = Process(input_ptr, antennas, points, prof_events);
  // Process() вызывает hipStreamSynchronize(ctx_.stream()) — stream синхронизирован

  // ── Собрать Upload_input (stream уже синхронизирован Process'ом) ──
  if (prof_events) {
    prof_events->push_back({"Upload_input",
        MakeROCmDataFromEvents(ev_in_start, ev_in_end, 1, "H2D_input")});
  }

  // Free the temporary input buffer
  (void)hipFree(input_ptr);

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 5: CPU Processing (reference)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<std::vector<std::complex<float>>>
LchFarrowROCm::ProcessCpu(
    const std::vector<std::vector<std::complex<float>>>& input,
    uint32_t antennas, uint32_t points)
{
  if (delay_us_.empty()) {
    delay_us_.assign(antennas, 0.0f);
  }

  std::vector<std::vector<std::complex<float>>> result(antennas);

  for (uint32_t a = 0; a < antennas; ++a) {
    result[a].resize(points, {0.0f, 0.0f});

    float delay_samples = delay_us_[a] * 1e-6f * sample_rate_;
    float L[5];

    for (uint32_t n = 0; n < points; ++n) {
      float read_pos = static_cast<float>(n) - delay_samples;
      if (read_pos < 0.0f) continue;

      int center = static_cast<int>(std::floor(read_pos));
      float frac = read_pos - static_cast<float>(center);
      uint32_t row = (static_cast<uint32_t>(frac * 48.0f)) % 48u;

      for (int k = 0; k < 5; ++k)
        L[k] = lagrange_matrix_[row * 5 + k];

      std::complex<float> val(0.0f, 0.0f);
      for (int k = 0; k < 5; ++k) {
        int idx = center - 1 + k;
        if (idx >= 0 && idx < static_cast<int>(points)) {
          val += L[k] * input[a][idx];
        }
      }
      result[a][n] = val;
    }
  }

  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 6: GPU Internals
// ═══════════════════════════════════════════════════════════════════════════

void LchFarrowROCm::EnsureCompiled() {
  if (compiled_) return;
  ctx_.CompileModule(kernels::GetLchFarrowKernelSource(), kKernelNames);
  compiled_ = true;
}

void LchFarrowROCm::UploadMatrix() {
  if (matrix_buf_) {
    (void)hipFree(matrix_buf_);
    matrix_buf_ = nullptr;
  }

  size_t matrix_size = lagrange_matrix_.size() * sizeof(float);
  hipError_t err = hipMalloc(&matrix_buf_, matrix_size);
  if (err != hipSuccess) {
    throw std::runtime_error(
        "LchFarrowROCm::UploadMatrix: hipMalloc failed: " +
        std::string(hipGetErrorString(err)));
  }

  err = hipMemcpyHtoDAsync(matrix_buf_, lagrange_matrix_.data(),
                            matrix_size, ctx_.stream());
  if (err != hipSuccess) {
    (void)hipFree(matrix_buf_);
    matrix_buf_ = nullptr;
    throw std::runtime_error(
        "LchFarrowROCm::UploadMatrix: hipMemcpyHtoDAsync failed");
  }
  (void)hipStreamSynchronize(ctx_.stream());
}

void LchFarrowROCm::ReleaseGpuResources() {
  // GpuContext manages kernel module — no manual hipModuleUnload needed
  if (matrix_buf_) {
    (void)hipFree(matrix_buf_);
    matrix_buf_ = nullptr;
  }
  if (delay_buf_) {
    (void)hipFree(delay_buf_);
    delay_buf_ = nullptr;
    delay_buf_size_ = 0;
  }
}

}  // namespace lch_farrow

#endif  // ENABLE_ROCM
