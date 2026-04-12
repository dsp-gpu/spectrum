#pragma once

/**
 * @file py_lch_farrow_rocm.hpp
 * @brief Python wrapper for LchFarrowROCm (Lagrange 48x5 fractional delay, ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   proc = gpuworklib.LchFarrowROCm(ctx)
 *   proc.set_sample_rate(1e6)
 *   proc.set_delays([0.0, 2.7, 5.0])
 *   delayed = proc.process(signal)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-24
 */

#include "lch_farrow_rocm.hpp"

// ============================================================================
// PyLchFarrowROCm — standalone fractional delay processor (ROCm)
// ============================================================================

// ROCm-версия LchFarrow (Lagrange 48×5 дробная задержка).
// Отличие от OpenCL PyLchFarrow:
//   - ProcessFromCPU вместо Process: ROCm backend сам управляет GPU-памятью
//   - result.data — void* (HIP ptr), освобождать через backend->Free()
// Алгоритм и API идентичны OpenCL-версии — можно менять backend без правок Python-кода.
class PyLchFarrowROCm {
public:
  explicit PyLchFarrowROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), proc_(ctx.backend()) {}

  void set_delays(const std::vector<float>& delay_us) {
    proc_.SetDelays(delay_us);
  }

  void set_sample_rate(float sample_rate) {
    proc_.SetSampleRate(sample_rate);
  }

  void set_noise(float noise_amplitude, float norm_val = 0.7071067811865476f,
                 uint32_t noise_seed = 0) {
    proc_.SetNoise(noise_amplitude, norm_val, noise_seed);
  }

  void load_matrix(const std::string& json_path) {
    proc_.LoadMatrix(json_path);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto buf = input.request();
    uint32_t antennas, points;

    if (buf.ndim == 2) {
      antennas = static_cast<uint32_t>(buf.shape[0]);
      points   = static_cast<uint32_t>(buf.shape[1]);
    } else if (buf.ndim == 1) {
      antennas = 1;
      points   = static_cast<uint32_t>(buf.shape[0]);
    } else {
      throw std::invalid_argument("Input must be 1D or 2D");
    }

    size_t total = static_cast<size_t>(antennas) * points;
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> input_vec(ptr, ptr + total);

    drv_gpu_lib::InputData<void*> result;
    {
      py::gil_scoped_release release;
      result = proc_.ProcessFromCPU(input_vec, antennas, points);
    }

    // result.data — HIP device pointer, caller owns. Освобождать только через backend->Free().
    std::vector<std::complex<float>> data(total);
    ctx_.backend()->MemcpyDeviceToHost(data.data(), result.data,
                                        total * sizeof(std::complex<float>));
    ctx_.backend()->Free(result.data);

    if (antennas <= 1) return vector_to_numpy(std::move(data));
    return vector_to_numpy_2d(std::move(data), antennas, points);
  }

  float sample_rate() const { return proc_.GetSampleRate(); }

  py::list get_delays() const {
    py::list result;
    for (float d : proc_.GetDelays()) result.append(d);
    return result;
  }

private:
  ROCmGPUContext& ctx_;
  lch_farrow::LchFarrowROCm proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_lch_farrow_rocm(py::module& m) {
  py::class_<PyLchFarrowROCm>(m, "LchFarrowROCm",
      "Standalone fractional delay processor (Lagrange 48x5, ROCm).\n\n"
      "Same API as LchFarrow but uses ROCm/HIP backend.\n\n"
      "Algorithm:\n"
      "  delay_samples = delay_us * 1e-6 * sample_rate\n"
      "  D = floor(delay_samples), mu = frac(delay_samples)\n"
      "  row = int(mu * 48) %% 48\n"
      "  output[n] = sum(L[row][k] * input[n - D - 1 + k], k=0..4)\n\n"
      "Usage:\n"
      "  proc = gpuworklib.LchFarrowROCm(ctx)\n"
      "  proc.set_sample_rate(1e6)\n"
      "  proc.set_delays([0.0, 2.7, 5.0])\n"
      "  delayed = proc.process(signal)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create LchFarrowROCm processor bound to ROCm GPU context")

      .def("set_delays", &PyLchFarrowROCm::set_delays,
           py::arg("delay_us"),
           "Set per-antenna delays in microseconds.")

      .def("set_sample_rate", &PyLchFarrowROCm::set_sample_rate,
           py::arg("sample_rate"),
           "Set sample rate (Hz).")

      .def("set_noise", &PyLchFarrowROCm::set_noise,
           py::arg("noise_amplitude"),
           py::arg("norm_val") = 0.7071067811865476f,
           py::arg("noise_seed") = 0,
           "Set noise parameters (added AFTER delay).")

      .def("load_matrix", &PyLchFarrowROCm::load_matrix,
           py::arg("json_path"),
           "Load Lagrange 48x5 matrix from JSON.")

      .def("process", &PyLchFarrowROCm::process,
           py::arg("input"),
           "Apply fractional delay on GPU (ROCm).\n\n"
           "Args:\n"
           "  input: numpy complex64 (points,) or (antennas, points)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape, delayed signal")

      .def_property_readonly("sample_rate", &PyLchFarrowROCm::sample_rate)
      .def_property_readonly("delays", &PyLchFarrowROCm::get_delays)

      .def("__repr__", [](const PyLchFarrowROCm& self) {
          return "<LchFarrowROCm sample_rate=" +
                 std::to_string(static_cast<int>(self.sample_rate())) + ">";
      });
}
