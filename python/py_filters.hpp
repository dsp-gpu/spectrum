#pragma once

/**
 * @file py_filters.hpp
 * @brief Python wrappers for FirFilter and IirFilter
 *
 * Include AFTER GPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   fir = gpuworklib.FirFilter(ctx)
 *   fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())
 *   result = fir.process(signal)
 *
 *   iir = gpuworklib.IirFilter(ctx)
 *   iir.set_sections([{'b0':0.02, 'b1':0.04, 'b2':0.02, 'a1':-1.56, 'a2':0.64}])
 *   result = iir.process(signal)
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include "filters/fir_filter.hpp"
#include "filters/iir_filter.hpp"

// ============================================================================
// PyFirFilter — GPU FIR convolution filter
// ============================================================================

// Python-обёртка над FirFilter (OpenCL).
// FirFilter.Process() ожидает cl_mem на входе — поэтому мы сами создаём
// буфер через clCreateBuffer+CL_MEM_COPY_HOST_PTR и освобождаем после Process.
// Для ROCm-версии используется ProcessFromCPU — backend сам управляет памятью.
class PyFirFilter {
public:
  explicit PyFirFilter(GPUContext& ctx)
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

    // Upload to GPU: FirFilter ожидает cl_mem (не vector), поэтому загружаем вручную.
    // CL_MEM_COPY_HOST_PTR — синхронная копия при создании буфера, input_buf готов сразу.
    cl_context cl_ctx = static_cast<cl_context>(ctx_.backend()->GetNativeContext());
    cl_int err;
    cl_mem input_buf = clCreateBuffer(cl_ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total * sizeof(std::complex<float>),
        const_cast<std::complex<float>*>(ptr), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("PyFirFilter: upload failed (" +
                               std::to_string(err) + ")");

    drv_gpu_lib::InputData<cl_mem> result;
    {
      // Освобождаем GIL на время GPU-вычисления — другие Python потоки могут работать.
      // FirFilter.Process() блокирует до завершения kernel + копирования.
      py::gil_scoped_release release;
      result = fir_.Process(input_buf, channels, points);
    }
    clReleaseMemObject(input_buf);

    // Readback: result.data — cl_mem, caller owns. FirFilter не освобождает его —
    // наша ответственность вызвать clReleaseMemObject после чтения.
    std::vector<std::complex<float>> data(total);
    clEnqueueReadBuffer(ctx_.queue(), result.data, CL_TRUE, 0,
                        total * sizeof(std::complex<float>),
                        data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    if (channels <= 1) {
      return vector_to_numpy(std::move(data));
    }
    return vector_to_numpy_2d(std::move(data), channels, points);
  }

  py::list get_coefficients() const {
    py::list result;
    for (float c : fir_.GetCoefficients()) result.append(c);
    return result;
  }

  uint32_t num_taps() const { return fir_.GetNumTaps(); }

private:
  GPUContext& ctx_;
  filters::FirFilter fir_;
};

// ============================================================================
// PyIirFilter — GPU IIR biquad cascade filter
// ============================================================================

class PyIirFilter {
public:
  explicit PyIirFilter(GPUContext& ctx)
      : ctx_(ctx), iir_(ctx.backend()) {}

  void load_config(const std::string& path) { iir_.LoadConfig(path); }

  /**
   * @brief Set biquad sections from Python list of dicts
   * Each dict: {'b0':float, 'b1':float, 'b2':float, 'a1':float, 'a2':float}
   */
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

    cl_context cl_ctx = static_cast<cl_context>(ctx_.backend()->GetNativeContext());
    cl_int err;
    cl_mem input_buf = clCreateBuffer(cl_ctx,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        total * sizeof(std::complex<float>),
        const_cast<std::complex<float>*>(ptr), &err);
    if (err != CL_SUCCESS)
      throw std::runtime_error("PyIirFilter: upload failed (" +
                               std::to_string(err) + ")");

    drv_gpu_lib::InputData<cl_mem> result;
    {
      py::gil_scoped_release release;
      result = iir_.Process(input_buf, channels, points);
    }
    clReleaseMemObject(input_buf);

    std::vector<std::complex<float>> data(total);
    clEnqueueReadBuffer(ctx_.queue(), result.data, CL_TRUE, 0,
                        total * sizeof(std::complex<float>),
                        data.data(), 0, nullptr, nullptr);
    clReleaseMemObject(result.data);

    if (channels <= 1) {
      return vector_to_numpy(std::move(data));
    }
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
  GPUContext& ctx_;
  filters::IirFilter iir_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_filters(py::module& m) {
  // FirFilter
  py::class_<PyFirFilter>(m, "FirFilter",
      "GPU FIR filter (OpenCL, direct-form convolution).\n\n"
      "Applies FIR filter to multi-channel complex signal.\n"
      "Each channel processed in parallel on GPU.\n\n"
      "Usage:\n"
      "  fir = gpuworklib.FirFilter(ctx)\n"
      "  fir.set_coefficients(scipy.signal.firwin(64, 0.1).tolist())\n"
      "  result = fir.process(signal)  # (channels, points) complex64\n")
      .def(py::init<GPUContext&>(), py::arg("ctx"),
           "Create FIR filter bound to GPU context")
      .def("load_config", &PyFirFilter::load_config,
           py::arg("json_path"),
           "Load FIR coefficients from JSON file.")
      .def("set_coefficients", &PyFirFilter::set_coefficients,
           py::arg("coefficients"),
           "Set FIR coefficients (list of float).\n"
           "From scipy: firwin(N, cutoff).tolist()")
      .def("process", &PyFirFilter::process,
           py::arg("input"),
           "Apply FIR filter on GPU.\n\n"
           "Args:\n"
           "  input: numpy complex64 (points,) or (channels, points)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape, filtered signal")
      .def_property_readonly("num_taps", &PyFirFilter::num_taps)
      .def_property_readonly("coefficients", &PyFirFilter::get_coefficients)
      .def("__repr__", [](const PyFirFilter& self) {
          return "<FirFilter num_taps=" + std::to_string(self.num_taps()) + ">";
      });

  // IirFilter
  py::class_<PyIirFilter>(m, "IirFilter",
      "GPU IIR biquad cascade filter (OpenCL, DFII-Transposed).\n\n"
      "Applies IIR filter as cascade of biquad sections.\n"
      "Each channel processed in parallel on GPU.\n\n"
      "Note: GPU IIR is efficient ONLY with many channels (>= 8).\n"
      "      Single-channel IIR is faster on CPU!\n\n"
      "Usage:\n"
      "  iir = gpuworklib.IirFilter(ctx)\n"
      "  iir.set_sections([{'b0':0.02,'b1':0.04,'b2':0.02,'a1':-1.56,'a2':0.64}])\n"
      "  result = iir.process(signal)\n")
      .def(py::init<GPUContext&>(), py::arg("ctx"),
           "Create IIR filter bound to GPU context")
      .def("load_config", &PyIirFilter::load_config,
           py::arg("json_path"),
           "Load IIR biquad sections from JSON file.")
      .def("set_sections", &PyIirFilter::set_sections,
           py::arg("sections"),
           "Set biquad sections from list of dicts.\n"
           "Each dict: {'b0':f, 'b1':f, 'b2':f, 'a1':f, 'a2':f}\n"
           "From scipy: butter(N, Wn, output='sos') -> convert to dicts")
      .def("process", &PyIirFilter::process,
           py::arg("input"),
           "Apply IIR biquad cascade on GPU.\n\n"
           "Args:\n"
           "  input: numpy complex64 (points,) or (channels, points)\n\n"
           "Returns:\n"
           "  numpy.ndarray complex64: same shape, filtered signal")
      .def_property_readonly("num_sections", &PyIirFilter::num_sections)
      .def_property_readonly("sections", &PyIirFilter::get_sections)
      .def("__repr__", [](const PyIirFilter& self) {
          return "<IirFilter num_sections=" + std::to_string(self.num_sections()) + ">";
      });
}
