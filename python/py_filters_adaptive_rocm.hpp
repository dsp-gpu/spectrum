#pragma once

/**
 * @file py_filters_adaptive_rocm.hpp
 * @brief Python wrappers for MovingAverageFilterROCm, KalmanFilterROCm, KaufmanFilterROCm
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage:
 *   ma = gpuworklib.MovingAverageFilterROCm(ctx)
 *   ma.set_params("EMA", 10)
 *   out = ma.process(data)   # numpy complex64, shape = input shape
 *
 *   kalman = gpuworklib.KalmanFilterROCm(ctx)
 *   kalman.set_params(Q=0.1, R=25.0, x0=0.0, P0=25.0)
 *   out = kalman.process(data)
 *
 *   kauf = gpuworklib.KaufmanFilterROCm(ctx)
 *   kauf.set_params(er_period=10, fast=2, slow=30)
 *   out = kauf.process(data)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-10
 */

#include "filters/moving_average_filter_rocm.hpp"
#include "filters/kalman_filter_rocm.hpp"
#include "filters/kaufman_filter_rocm.hpp"

// ============================================================================
// Helper: parse input numpy array → (channels, points)
// ============================================================================

namespace {

// Returns (channels, points) — 1D input treated as (1, N)
inline std::pair<uint32_t, uint32_t> parse_filter_input(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>& arr)
{
  auto buf = arr.request();
  uint32_t channels, points;
  if (buf.ndim == 1) {
    channels = 1;
    points   = static_cast<uint32_t>(buf.shape[0]);
  } else if (buf.ndim == 2) {
    channels = static_cast<uint32_t>(buf.shape[0]);
    points   = static_cast<uint32_t>(buf.shape[1]);
  } else {
    throw std::invalid_argument("process: input must be 1D or 2D complex64");
  }
  return {channels, points};
}

// Convert flat vector → numpy, preserving original shape (channels, points).
// If original was 1D (channels==1), return 1D.
inline py::array_t<std::complex<float>> filter_result_to_numpy(
    std::vector<std::complex<float>>&& data,
    uint32_t channels, uint32_t points, bool was_1d)
{
  if (was_1d) {
    // Return 1D array
    py::array_t<std::complex<float>> out({(py::ssize_t)points});
    std::memcpy(out.mutable_data(), data.data(), points * sizeof(std::complex<float>));
    return out;
  }
  return vector_to_numpy_2d(std::move(data), channels, points);
}

// Parse MAType string
inline filters::MAType parse_ma_type(const std::string& s) {
  std::string up = s;
  std::transform(up.begin(), up.end(), up.begin(), ::toupper);
  if (up == "SMA")  return filters::MAType::SMA;
  if (up == "EMA")  return filters::MAType::EMA;
  if (up == "MMA")  return filters::MAType::MMA;
  if (up == "DEMA") return filters::MAType::DEMA;
  if (up == "TEMA") return filters::MAType::TEMA;
  throw std::invalid_argument("Unknown MAType: '" + s + "'. Use SMA/EMA/MMA/DEMA/TEMA");
}

inline std::string ma_type_to_str(filters::MAType t) {
  switch (t) {
    case filters::MAType::SMA:  return "SMA";
    case filters::MAType::EMA:  return "EMA";
    case filters::MAType::MMA:  return "MMA";
    case filters::MAType::DEMA: return "DEMA";
    case filters::MAType::TEMA: return "TEMA";
    default: return "UNKNOWN";
  }
}

}  // namespace

// ============================================================================
// PyMovingAverageFilterROCm
// ============================================================================

class PyMovingAverageFilterROCm {
public:
  explicit PyMovingAverageFilterROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), filt_(ctx.backend()) {}

  void set_params(const std::string& type_str, uint32_t window_size) {
    filt_.SetParams(parse_ma_type(type_str), window_size);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto [channels, points] = parse_filter_input(input);
    bool was_1d = (input.ndim() == 1);

    auto buf = input.request();
    std::vector<std::complex<float>> cpu_in(
        static_cast<const std::complex<float>*>(buf.ptr),
        static_cast<const std::complex<float>*>(buf.ptr) + channels * points);

    std::vector<std::complex<float>> out(channels * points);
    {
      py::gil_scoped_release rel;
      auto result = filt_.ProcessFromCPU(cpu_in, channels, points);
      ctx_.backend()->MemcpyDeviceToHost(
          out.data(), result.data, out.size() * sizeof(std::complex<float>));
      ctx_.backend()->Free(result.data);
    }
    return filter_result_to_numpy(std::move(out), channels, points, was_1d);
  }

  bool     is_ready()        const { return true; }
  uint32_t get_window_size() const { return filt_.GetWindowSize(); }
  std::string get_type()     const { return ma_type_to_str(filt_.GetType()); }

private:
  ROCmGPUContext& ctx_;
  filters::MovingAverageFilterROCm filt_;
};

// ============================================================================
// PyKalmanFilterROCm
// ============================================================================

class PyKalmanFilterROCm {
public:
  explicit PyKalmanFilterROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), filt_(ctx.backend()) {}

  void set_params(float Q, float R, float x0 = 0.0f, float P0 = 25.0f) {
    filt_.SetParams(Q, R, x0, P0);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto [channels, points] = parse_filter_input(input);
    bool was_1d = (input.ndim() == 1);

    auto buf = input.request();
    std::vector<std::complex<float>> cpu_in(
        static_cast<const std::complex<float>*>(buf.ptr),
        static_cast<const std::complex<float>*>(buf.ptr) + channels * points);

    std::vector<std::complex<float>> out(channels * points);
    {
      py::gil_scoped_release rel;
      auto result = filt_.ProcessFromCPU(cpu_in, channels, points);
      ctx_.backend()->MemcpyDeviceToHost(
          out.data(), result.data, out.size() * sizeof(std::complex<float>));
      ctx_.backend()->Free(result.data);
    }
    return filter_result_to_numpy(std::move(out), channels, points, was_1d);
  }

  bool is_ready() const { return true; }

  py::dict get_params() const {
    const auto& p = filt_.GetParams();
    py::dict d;
    d["Q"]  = p.Q;
    d["R"]  = p.R;
    d["x0"] = p.x0;
    d["P0"] = p.P0;
    return d;
  }

private:
  ROCmGPUContext& ctx_;
  filters::KalmanFilterROCm filt_;
};

// ============================================================================
// PyKaufmanFilterROCm
// ============================================================================

class PyKaufmanFilterROCm {
public:
  explicit PyKaufmanFilterROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), filt_(ctx.backend()) {}

  void set_params(uint32_t er_period, uint32_t fast = 2, uint32_t slow = 30) {
    filt_.SetParams(er_period, fast, slow);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto [channels, points] = parse_filter_input(input);
    bool was_1d = (input.ndim() == 1);

    auto buf = input.request();
    std::vector<std::complex<float>> cpu_in(
        static_cast<const std::complex<float>*>(buf.ptr),
        static_cast<const std::complex<float>*>(buf.ptr) + channels * points);

    std::vector<std::complex<float>> out(channels * points);
    {
      py::gil_scoped_release rel;
      auto result = filt_.ProcessFromCPU(cpu_in, channels, points);
      ctx_.backend()->MemcpyDeviceToHost(
          out.data(), result.data, out.size() * sizeof(std::complex<float>));
      ctx_.backend()->Free(result.data);
    }
    return filter_result_to_numpy(std::move(out), channels, points, was_1d);
  }

  bool is_ready() const { return true; }

  py::dict get_params() const {
    const auto& p = filt_.GetParams();
    py::dict d;
    d["er_period"]   = p.er_period;
    d["fast_period"] = p.fast_period;
    d["slow_period"] = p.slow_period;
    return d;
  }

private:
  ROCmGPUContext& ctx_;
  filters::KaufmanFilterROCm filt_;
};

// ============================================================================
// Registration
// ============================================================================

inline void register_filters_adaptive_rocm(py::module& m) {

  // ── MovingAverageFilterROCm ──────────────────────────────────────────────
  py::class_<PyMovingAverageFilterROCm>(m, "MovingAverageFilterROCm",
      "GPU moving average filter on ROCm (SMA/EMA/MMA/DEMA/TEMA).\n\n"
      "Usage:\n"
      "  ma = gpuworklib.MovingAverageFilterROCm(ctx)\n"
      "  ma.set_params('EMA', 10)\n"
      "  out = ma.process(data)  # numpy complex64, same shape as input\n")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"))
    .def("set_params", &PyMovingAverageFilterROCm::set_params,
         py::arg("type"), py::arg("window_size"),
         "Set filter type ('SMA'/'EMA'/'MMA'/'DEMA'/'TEMA') and window size.")
    .def("process", &PyMovingAverageFilterROCm::process, py::arg("data"),
         "Filter signal. Input: 1D or 2D complex64. Returns same shape.")
    .def("is_ready",        &PyMovingAverageFilterROCm::is_ready)
    .def("get_window_size", &PyMovingAverageFilterROCm::get_window_size)
    .def("get_type",        &PyMovingAverageFilterROCm::get_type,
         "Return filter type as string: 'SMA'/'EMA'/'MMA'/'DEMA'/'TEMA'.")
    .def("__repr__", [](const PyMovingAverageFilterROCm& f) {
        return "<MovingAverageFilterROCm type=" + f.get_type() +
               " N=" + std::to_string(f.get_window_size()) + ">";
    });

  // ── KalmanFilterROCm ─────────────────────────────────────────────────────
  py::class_<PyKalmanFilterROCm>(m, "KalmanFilterROCm",
      "1D scalar Kalman filter on ROCm (applied to Re and Im independently).\n\n"
      "Usage:\n"
      "  kalman = gpuworklib.KalmanFilterROCm(ctx)\n"
      "  kalman.set_params(Q=0.1, R=25.0, x0=0.0, P0=25.0)\n"
      "  out = kalman.process(data)  # numpy complex64\n\n"
      "Q/R ratio controls smoothing:\n"
      "  Q/R << 1 → strong smoothing, slow reaction\n"
      "  Q/R >> 1 → weak smoothing, fast reaction\n")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"))
    .def("set_params", &PyKalmanFilterROCm::set_params,
         py::arg("Q"), py::arg("R"),
         py::arg("x0") = 0.0f, py::arg("P0") = 25.0f,
         "Set Kalman parameters: Q (process noise), R (measurement noise), "
         "x0 (initial state), P0 (initial covariance).")
    .def("process", &PyKalmanFilterROCm::process, py::arg("data"),
         "Filter signal. Input: 1D or 2D complex64. Returns same shape.")
    .def("is_ready",   &PyKalmanFilterROCm::is_ready)
    .def("get_params", &PyKalmanFilterROCm::get_params,
         "Return dict with keys: Q, R, x0, P0.")
    .def("__repr__", [](const PyKalmanFilterROCm& f) {
        auto p = f.get_params();
        return "<KalmanFilterROCm Q=" +
               std::to_string(py::cast<float>(p["Q"])) + " R=" +
               std::to_string(py::cast<float>(p["R"])) + ">";
    });

  // ── KaufmanFilterROCm ────────────────────────────────────────────────────
  py::class_<PyKaufmanFilterROCm>(m, "KaufmanFilterROCm",
      "Kaufman Adaptive Moving Average (KAMA) on ROCm.\n\n"
      "Usage:\n"
      "  kauf = gpuworklib.KaufmanFilterROCm(ctx)\n"
      "  kauf.set_params(er_period=10, fast=2, slow=30)\n"
      "  out = kauf.process(data)  # numpy complex64\n\n"
      "er_period ≤ 128 (GPU ring-buffer limit).\n"
      "ER≈1 (trending) → fast tracking (alpha≈fast_sc)\n"
      "ER≈0 (noisy)    → slow tracking (alpha≈slow_sc)\n")
    .def(py::init<ROCmGPUContext&>(), py::arg("ctx"))
    .def("set_params", &PyKaufmanFilterROCm::set_params,
         py::arg("er_period"), py::arg("fast") = 2u, py::arg("slow") = 30u,
         "Set KAMA parameters: er_period (NN, max 128), fast EMA period, slow EMA period.")
    .def("process", &PyKaufmanFilterROCm::process, py::arg("data"),
         "Filter signal. Input: 1D or 2D complex64. Returns same shape.")
    .def("is_ready",   &PyKaufmanFilterROCm::is_ready)
    .def("get_params", &PyKaufmanFilterROCm::get_params,
         "Return dict with keys: er_period, fast_period, slow_period.")
    .def("__repr__", [](const PyKaufmanFilterROCm& f) {
        auto p = f.get_params();
        return "<KaufmanFilterROCm er=" +
               std::to_string(py::cast<uint32_t>(p["er_period"])) +
               " fast=" + std::to_string(py::cast<uint32_t>(p["fast_period"])) +
               " slow=" + std::to_string(py::cast<uint32_t>(p["slow_period"])) + ">";
    });
}
