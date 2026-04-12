#pragma once

/**
 * @file py_complex_to_mag_rocm.hpp
 * @brief Python wrapper for ComplexToMagPhaseROCm::ProcessMagnitude / ProcessMagnitudeToGPU
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Usage from Python:
 *   proc = gpuworklib.ComplexToMagROCm(ctx)
 *
 *   # GPU in → CPU out (numpy float32)
 *   mag = proc.process_magnitude(data, beam_count=4, norm_coeff=-1.0)
 *   # shape: (beam_count, n_point) float32
 *
 * norm_coeff:
 *   0.0  → no normalization (multiply by 1)
 *  -1.0  → divide by n_point (multiply by 1/n_point)
 *  >0.0  → multiply by norm_coeff
 *
 * @author Kodo (AI Assistant)
 * @date 2026-03-11
 */

#include "complex_to_mag_phase_rocm.hpp"
#include "types/mag_phase_types.hpp"

// ============================================================================
// PyComplexToMagROCm — magnitude-only conversion (GPU, ROCm)
// ============================================================================

// Magnitude-only вариант ComplexToMagPhaseROCm — не вычисляет фазу.
// process_magnitude: принимает numpy complex64, возвращает numpy float32 (magnitudes).
// norm_coeff управляет нормировкой: 0=без норм., -1=делить на n_point, >0=умножить.
class PyComplexToMagROCm {
public:
  explicit PyComplexToMagROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), proc_(ctx.backend()) {}

  // ── ProcessMagnitude: GPU in → CPU out ──────────────────────────────────

  py::object process_magnitude(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      uint32_t beam_count = 0,
      float norm_coeff = 1.0f)
  {
    auto buf = data.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (buf.ndim == 2) {
      if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
    } else {
      if (beam_count == 0) beam_count = 1;
    }

    uint32_t n_point = static_cast<uint32_t>(total / beam_count);
    if (beam_count == 0 || total % beam_count != 0) {
      throw std::invalid_argument(
          "process_magnitude: total=" + std::to_string(total) +
          " not divisible by beam_count=" + std::to_string(beam_count));
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    fft_processor::MagPhaseParams params;
    params.beam_count  = beam_count;
    params.n_point     = n_point;
    params.norm_coeff  = norm_coeff;

    // Сначала загружаем на GPU (используем Process с вектором), а потом ProcessMagnitude
    // через временный GPU буфер — но ProcessMagnitude принимает void* или vector нет.
    // Вместо этого вызываем Process (mag+phase), но нам нужна только mag.
    // Правильно: загрузить вектор на GPU вручную, затем вызвать ProcessMagnitude(void*).
    // Краткий путь: используем Process (внутри C++ ProcessMagnitude с вектором
    // добавим через отдельную перегрузку). Пока используем GPU upload + ProcessMagnitude.
    void* gpu_ptr = nullptr;
    size_t bytes = total * sizeof(std::complex<float>);
    hipError_t err = hipMalloc(&gpu_ptr, bytes);
    if (err != hipSuccess) {
      throw std::runtime_error("process_magnitude: hipMalloc failed");
    }
    err = hipMemcpy(gpu_ptr, vec.data(), bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
      (void)hipFree(gpu_ptr);
      throw std::runtime_error("process_magnitude: hipMemcpy H2D failed");
    }

    std::vector<fft_processor::MagnitudeResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ProcessMagnitude(gpu_ptr, params, bytes);
    }
    (void)hipFree(gpu_ptr);

    if (results.size() == 1) {
      return vector_to_numpy(std::move(results[0].magnitude));
    }

    // 2D output: (beam_count, n_point) float32
    std::vector<float> flat;
    flat.reserve(beam_count * n_point);
    for (auto& r : results)
      flat.insert(flat.end(), r.magnitude.begin(), r.magnitude.end());
    return vector_to_numpy_2d(std::move(flat), results.size(), n_point);
  }

private:
  ROCmGPUContext& ctx_;
  fft_processor::ComplexToMagPhaseROCm proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_complex_to_mag_rocm(py::module& m) {
  py::class_<PyComplexToMagROCm>(m, "ComplexToMagROCm",
      "Magnitude-only conversion from complex IQ data (ROCm).\n\n"
      "Faster than ComplexToMagPhaseROCm when phase is not needed.\n\n"
      "Usage:\n"
      "  proc = gpuworklib.ComplexToMagROCm(ctx)\n"
      "  mag = proc.process_magnitude(iq_data, beam_count=4, norm_coeff=-1.0)\n"
      "  # mag.shape = (4, n_point) float32\n\n"
      "norm_coeff:\n"
      "  0.0  → no normalization (×1)\n"
      " -1.0  → divide by n_point (×1/n_point)\n"
      "  >0   → multiply by norm_coeff\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create ComplexToMagROCm bound to ROCm GPU context")

      .def("process_magnitude", &PyComplexToMagROCm::process_magnitude,
           py::arg("data"),
           py::arg("beam_count") = 0,
           py::arg("norm_coeff") = 1.0f,
           "Convert complex IQ data to magnitudes (GPU).\n\n"
           "Args:\n"
           "  data: numpy complex64, shape (N,) or (beam_count, n_point)\n"
           "  beam_count: number of beams (0=auto from shape)\n"
           "  norm_coeff: 0=none, -1=÷n_point, >0=×norm_coeff\n\n"
           "Returns:\n"
           "  numpy float32: shape (n_point,) for 1 beam, (beam_count, n_point) for N beams")

      .def("__repr__", [](const PyComplexToMagROCm&) {
          return "<ComplexToMagROCm>";
      });
}
