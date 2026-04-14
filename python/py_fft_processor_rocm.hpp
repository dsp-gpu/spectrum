#pragma once

/**
 * @file py_fft_processor_rocm.hpp
 * @brief Python wrapper for FFTProcessorROCm (hipFFT, ROCm backend)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Same API as PyFFTProcessor (OpenCL) — drop-in replacement:
 *   proc = gpuworklib.FFTProcessorROCm(ctx)
 *   spec = proc.process_complex(data, sample_rate=1e6)
 *   mp   = proc.process_mag_phase(data, sample_rate=1e6)
 *   nfft = proc.nfft
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-10
 */

#include <spectrum/fft_processor_rocm.hpp>

// ============================================================================
// PyFFTProcessorROCm — pythonic wrapper over fft_processor::FFTProcessorROCm
// ============================================================================

// ROCm-версия FFTProcessor (hipFFT + hiprtc).
// API зеркально совпадает с PyFFTProcessor (OpenCL) — меняется только ctx и класс.
// process_complex  → numpy complex64 (спектр)
// process_mag_phase → dict {magnitude, phase, frequency, nFFT, sample_rate}
// Два-plan LRU-2 кеш: не пересоздаёт план при чередовании двух размеров batch.
class PyFFTProcessorROCm {
public:
  explicit PyFFTProcessorROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), fft_(ctx.backend()) {}

  // ── Complex FFT ──────────────────────────────────────────────────────────

  py::array_t<std::complex<float>> process_complex(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      float sample_rate,
      uint32_t beam_count = 0,
      uint32_t n_point = 0)
  {
    auto buf = data.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (buf.ndim == 2) {
      if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
      if (n_point   == 0) n_point    = static_cast<uint32_t>(buf.shape[1]);
    } else {
      if (beam_count == 0) beam_count = 1;
      if (n_point   == 0) n_point    = static_cast<uint32_t>(total / beam_count);
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    fft_processor::FFTProcessorParams params;
    params.beam_count  = beam_count;
    params.n_point     = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = fft_processor::FFTOutputMode::COMPLEX;

    std::vector<fft_processor::FFTComplexResult> results;
    {
      py::gil_scoped_release release;
      results = fft_.ProcessComplex(vec, params);
    }

    if (results.size() == 1) {
      return vector_to_numpy(std::move(results[0].spectrum));
    }

    uint32_t nFFT = results[0].nFFT;
    std::vector<std::complex<float>> flat;
    flat.reserve(results.size() * nFFT);
    for (auto& r : results)
      flat.insert(flat.end(), r.spectrum.begin(), r.spectrum.end());
    return vector_to_numpy_2d(std::move(flat), results.size(), nFFT);
  }

  // ── Magnitude + Phase FFT ────────────────────────────────────────────────

  py::dict process_mag_phase(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data,
      float sample_rate,
      uint32_t beam_count = 0,
      uint32_t n_point = 0,
      bool include_freq = true)
  {
    auto buf = data.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (buf.ndim == 2) {
      if (beam_count == 0) beam_count = static_cast<uint32_t>(buf.shape[0]);
      if (n_point   == 0) n_point    = static_cast<uint32_t>(buf.shape[1]);
    } else {
      if (beam_count == 0) beam_count = 1;
      if (n_point   == 0) n_point    = static_cast<uint32_t>(total / beam_count);
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    fft_processor::FFTProcessorParams params;
    params.beam_count  = beam_count;
    params.n_point     = n_point;
    params.sample_rate = sample_rate;
    params.output_mode = include_freq
        ? fft_processor::FFTOutputMode::MAGNITUDE_PHASE_FREQ
        : fft_processor::FFTOutputMode::MAGNITUDE_PHASE;

    std::vector<fft_processor::FFTMagPhaseResult> results;
    {
      py::gil_scoped_release release;
      results = fft_.ProcessMagPhase(vec, params);
    }

    py::dict out;
    out["nFFT"]        = results[0].nFFT;
    out["sample_rate"] = results[0].sample_rate;

    if (results.size() == 1) {
      out["magnitude"] = vector_to_numpy(std::move(results[0].magnitude));
      out["phase"]     = vector_to_numpy(std::move(results[0].phase));
      if (include_freq && !results[0].frequency.empty())
        out["frequency"] = vector_to_numpy(std::move(results[0].frequency));
    } else {
      uint32_t nFFT = results[0].nFFT;
      std::vector<float> all_mag, all_phase, all_freq;
      all_mag.reserve(results.size() * nFFT);
      all_phase.reserve(results.size() * nFFT);
      if (include_freq) all_freq.reserve(results.size() * nFFT);

      for (auto& r : results) {
        all_mag.insert(all_mag.end(),   r.magnitude.begin(), r.magnitude.end());
        all_phase.insert(all_phase.end(), r.phase.begin(),     r.phase.end());
        if (include_freq && !r.frequency.empty())
          all_freq.insert(all_freq.end(), r.frequency.begin(), r.frequency.end());
      }

      out["magnitude"] = vector_to_numpy_2d(std::move(all_mag),   results.size(), nFFT);
      out["phase"]     = vector_to_numpy_2d(std::move(all_phase),  results.size(), nFFT);
      if (!all_freq.empty())
        out["frequency"] = vector_to_numpy_2d(std::move(all_freq), results.size(), nFFT);
    }

    return out;
  }

  // ── Info ─────────────────────────────────────────────────────────────────

  py::dict get_profiling() const {
    auto p = fft_.GetProfilingData();
    py::dict d;
    d["upload_ms"]          = p.upload_time_ms;
    d["fft_ms"]             = p.fft_time_ms;
    d["post_processing_ms"] = p.post_processing_time_ms;
    d["download_ms"]        = p.download_time_ms;
    d["total_ms"]           = p.total_time_ms;
    return d;
  }

  uint32_t get_nfft() const { return fft_.GetNFFT(); }

private:
  ROCmGPUContext& ctx_;
  fft_processor::FFTProcessorROCm fft_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_fft_processor_rocm(py::module& m) {
  py::class_<PyFFTProcessorROCm>(m, "FFTProcessorROCm",
      "FFT processor on ROCm/HIP GPU (hipFFT + hiprtc).\n\n"
      "Drop-in replacement for FFTProcessor (OpenCL).\n"
      "Two-plan LRU-2 cache avoids plan recreation when alternating two batch sizes.\n\n"
      "Usage:\n"
      "  proc = gpuworklib.FFTProcessorROCm(ctx)\n"
      "  spec = proc.process_complex(data, sample_rate=1e6)\n"
      "  mp   = proc.process_mag_phase(data, sample_rate=1e6)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create FFTProcessorROCm bound to ROCm GPU context.")

      .def("process_complex", &PyFFTProcessorROCm::process_complex,
           py::arg("data"),
           py::arg("sample_rate"),
           py::arg("beam_count") = 0,
           py::arg("n_point")    = 0,
           "FFT with complex output.\n\n"
           "Args:\n"
           "  data: numpy complex64 (n_point,) or (beam_count, n_point)\n"
           "  sample_rate: Hz\n"
           "  beam_count: 0 = auto-detect from shape\n"
           "  n_point: 0 = auto-detect from shape\n\n"
           "Returns:\n"
           "  numpy complex64: (nFFT,) or (beam_count, nFFT)")

      .def("process_mag_phase", &PyFFTProcessorROCm::process_mag_phase,
           py::arg("data"),
           py::arg("sample_rate"),
           py::arg("beam_count")   = 0,
           py::arg("n_point")      = 0,
           py::arg("include_freq") = true,
           "FFT with magnitude + phase output.\n\n"
           "Args:\n"
           "  data: numpy complex64 (n_point,) or (beam_count, n_point)\n"
           "  sample_rate: Hz\n"
           "  include_freq: include frequency axis in result dict\n\n"
           "Returns:\n"
           "  dict with keys: magnitude, phase, frequency (if include_freq),\n"
           "                  nFFT, sample_rate")

      .def("get_profiling", &PyFFTProcessorROCm::get_profiling,
           "Get local profiling data (ms per stage).\n\n"
           "Keys: upload_ms, fft_ms, post_processing_ms, download_ms, total_ms.")

      .def_property_readonly("nfft", &PyFFTProcessorROCm::get_nfft,
           "Current FFT size (nextPow2(n_point) * repeat_count).")

      .def("__repr__", [](const PyFFTProcessorROCm& self) {
          return "<FFTProcessorROCm nfft=" + std::to_string(self.get_nfft()) + ">";
      });
}
