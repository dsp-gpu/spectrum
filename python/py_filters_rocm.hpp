#pragma once

/**
 * @file py_filters_rocm.hpp
 * @brief Python wrappers for FirFilterROCm and IirFilterROCm
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   fir = gpuworklib.FirFilterROCm(ctx)
 *   fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())
 *   result = fir.process(signal)
 *
 *   iir = gpuworklib.IirFilterROCm(ctx)
 *   iir.set_sections([{'b0':0.02, 'b1':0.04, 'b2':0.02, 'a1':-1.56, 'a2':0.64}])
 *   result = iir.process(signal)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-24
 */

#include "filters/fir_filter_rocm.hpp"
#include "filters/iir_filter_rocm.hpp"

// ============================================================================
// PyFirFilterROCm — GPU FIR convolution filter (ROCm)
// ============================================================================

// ROCm-версия FirFilter. Отличие от OpenCL (PyFirFilter):
//   - ProcessFromCPU вместо Process: ROCm backend сам копирует vector → GPU-память
//     (нет нужды вручную создавать cl_mem и освобождать его)
//   - результат result.data — void* (HIP-указатель), освобождать через backend->Free()
//     а не clReleaseMemObject. Никогда не вызывать hipFree напрямую — только через backend!
class PyFirFilterROCm {
public:
  explicit PyFirFilterROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), fir_(ctx.backend()) {}

  void load_config(const std::string& path) { fir_.LoadConfig(path); }

  void set_coefficients(const std::vector<float>& c) {
    fir_.SetCoefficients(c);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto buf = input.request();
    uint32_t channels, points;

    if (buf.ndim == 2) {
      channels = static_cast<uint32_t>(buf.shape[0]);
      points   = static_cast<uint32_t>(buf.shape[1]);
    } else if (buf.ndim == 1) {
      channels = 1;
      points   = static_cast<uint32_t>(buf.shape[0]);
    } else {
      throw std::invalid_argument("Input must be 1D or 2D");
    }

    size_t total = static_cast<size_t>(channels) * points;
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> input_vec(ptr, ptr + total);

    drv_gpu_lib::InputData<void*> result;
    {
      py::gil_scoped_release release;
      result = fir_.ProcessFromCPU(input_vec, channels, points);
    }

    // result.data — HIP device pointer, caller owns. Освобождаем через backend→Free()
    // (не hipFree напрямую — backend может использовать пул памяти).
    std::vector<std::complex<float>> data(total);
    ctx_.backend()->MemcpyDeviceToHost(data.data(), result.data,
                                        total * sizeof(std::complex<float>));
    ctx_.backend()->Free(result.data);

    if (channels <= 1) return vector_to_numpy(std::move(data));
    return vector_to_numpy_2d(std::move(data), channels, points);
  }

  py::list get_coefficients() const {
    py::list result;
    for (float c : fir_.GetCoefficients()) result.append(c);
    return result;
  }

  uint32_t num_taps() const { return fir_.GetNumTaps(); }

private:
  ROCmGPUContext& ctx_;
  filters::FirFilterROCm fir_;
};

// ============================================================================
// PyIirFilterROCm — GPU IIR biquad cascade filter (ROCm)
// ============================================================================

class PyIirFilterROCm {
public:
  explicit PyIirFilterROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), iir_(ctx.backend()) {}

  void load_config(const std::string& path) { iir_.LoadConfig(path); }

  void set_sections(py::list sections_list) {
    std::vector<filters::BiquadSection> sections;
    for (auto item : sections_list) {
      auto dict = item.cast<py::dict>();
      filters::BiquadSection sec;
      sec.b0 = dict["b0"].cast<float>();
      sec.b1 = dict["b1"].cast<float>();
      sec.b2 = dict["b2"].cast<float>();
      sec.a1 = dict["a1"].cast<float>();
      sec.a2 = dict["a2"].cast<float>();
      sections.push_back(sec);
    }
    iir_.SetBiquadSections(sections);
  }

  py::array_t<std::complex<float>> process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
  {
    auto buf = input.request();
    uint32_t channels, points;

    if (buf.ndim == 2) {
      channels = static_cast<uint32_t>(buf.shape[0]);
      points   = static_cast<uint32_t>(buf.shape[1]);
    } else if (buf.ndim == 1) {
      channels = 1;
      points   = static_cast<uint32_t>(buf.shape[0]);
    } else {
      throw std::invalid_argument("Input must be 1D or 2D");
    }

    size_t total = static_cast<size_t>(channels) * points;
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    std::vector<std::complex<float>> input_vec(ptr, ptr + total);

    drv_gpu_lib::InputData<void*> result;
    {
      py::gil_scoped_release release;
      result = iir_.ProcessFromCPU(input_vec, channels, points);
    }

    std::vector<std::complex<float>> data(total);
    ctx_.backend()->MemcpyDeviceToHost(data.data(), result.data,
                                        total * sizeof(std::complex<float>));
    ctx_.backend()->Free(result.data);

    if (channels <= 1) return vector_to_numpy(std::move(data));
    return vector_to_numpy_2d(std::move(data), channels, points);
  }

  py::list get_sections() const {
    py::list result;
    for (const auto& sec : iir_.GetSections()) {
      py::dict d;
      d["b0"] = sec.b0; d["b1"] = sec.b1; d["b2"] = sec.b2;
      d["a1"] = sec.a1; d["a2"] = sec.a2;
      result.append(d);
    }
    return result;
  }

  uint32_t num_sections() const { return iir_.GetNumSections(); }

private:
  ROCmGPUContext& ctx_;
  filters::IirFilterROCm iir_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_fir_filter_rocm(py::module& m) {
  py::class_<PyFirFilterROCm>(m, "FirFilterROCm",
      "GPU FIR filter (ROCm, direct-form convolution).\n\n"
      "Same API as FirFilter but uses ROCm/HIP backend.\n\n"
      "Usage:\n"
      "  fir = gpuworklib.FirFilterROCm(ctx)  # ctx = ROCmGPUContext\n"
      "  fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())\n"
      "  result = fir.process(signal)  # (channels, points) complex64\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create FIR filter bound to ROCm GPU context")
      .def("load_config", &PyFirFilterROCm::load_config,
           py::arg("json_path"),
           "Load FIR coefficients from JSON file.")
      .def("set_coefficients", &PyFirFilterROCm::set_coefficients,
           py::arg("coefficients"),
           "Set FIR coefficients (list of float).\n"
           "From scipy: firwin(N, cutoff).tolist()")
      .def("process", &PyFirFilterROCm::process,
           py::arg("input"),
           "Apply FIR filter on GPU (ROCm).\n\n"
           "Args:\n"
           "  input: numpy complex64 (points,) or (channels, points)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape, filtered signal")
      .def_property_readonly("num_taps", &PyFirFilterROCm::num_taps)
      .def_property_readonly("coefficients", &PyFirFilterROCm::get_coefficients)
      .def("__repr__", [](const PyFirFilterROCm& self) {
          return "<FirFilterROCm num_taps=" + std::to_string(self.num_taps()) + ">";
      });
}

inline void register_iir_filter_rocm(py::module& m) {
  py::class_<PyIirFilterROCm>(m, "IirFilterROCm",
      "GPU IIR biquad cascade filter (ROCm, DFII-Transposed).\n\n"
      "Same API as IirFilter but uses ROCm/HIP backend.\n\n"
      "Note: GPU IIR is efficient ONLY with many channels (>= 8).\n"
      "      Single-channel IIR is faster on CPU!\n\n"
      "Usage:\n"
      "  iir = gpuworklib.IirFilterROCm(ctx)\n"
      "  iir.set_sections([{'b0':0.02,'b1':0.04,'b2':0.02,'a1':-1.56,'a2':0.64}])\n"
      "  result = iir.process(signal)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create IIR filter bound to ROCm GPU context")
      .def("load_config", &PyIirFilterROCm::load_config,
           py::arg("json_path"),
           "Load IIR biquad sections from JSON file.")
      .def("set_sections", &PyIirFilterROCm::set_sections,
           py::arg("sections"),
           "Set biquad sections from list of dicts.\n"
           "Each dict: {'b0':f, 'b1':f, 'b2':f, 'a1':f, 'a2':f}")
      .def("process", &PyIirFilterROCm::process,
           py::arg("input"),
           "Apply IIR biquad cascade on GPU (ROCm).\n\n"
           "Args:\n"
           "  input: numpy complex64 (points,) or (channels, points)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape, filtered signal")
      .def_property_readonly("num_sections", &PyIirFilterROCm::num_sections)
      .def_property_readonly("sections", &PyIirFilterROCm::get_sections)
      .def("__repr__", [](const PyIirFilterROCm& self) {
          return "<IirFilterROCm num_sections=" + std::to_string(self.num_sections()) + ">";
      });
}
