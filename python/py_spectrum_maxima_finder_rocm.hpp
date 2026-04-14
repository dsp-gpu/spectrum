#pragma once

/**
 * @file py_spectrum_maxima_finder_rocm.hpp
 * @brief Python wrapper for SpectrumProcessorROCm (hipFFT peak finding, ROCm)
 *
 * Include AFTER ROCmGPUContext and vector_to_numpy definitions.
 *
 * Provides two methods compatible with the OpenCL SpectrumMaximaFinder:
 *
 *   process(data, n_point, sample_rate) → list[dict{antenna_id, freq_hz, magnitude, ...}]
 *     — ONE_PEAK mode: find single best peak per beam with parabolic interpolation
 *
 *   find_all_maxima(fft_data, n_point, sample_rate) → dict or list[dict]
 *     — ALL_MAXIMA mode: find ALL local maxima per beam
 *     Compatible output format with PySpectrumMaximaFinder (OpenCL).
 *
 * Usage:
 *   finder = gpuworklib.SpectrumMaximaFinderROCm(ctx)
 *   result = finder.process(signal, n_point=1024, sample_rate=1e6)
 *   maxima = finder.find_all_maxima(fft_spectrum, n_point=1024, sample_rate=1e6)
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-10
 */

#include <spectrum/processors/spectrum_processor_rocm.hpp>

// ============================================================================
// PySpectrumMaximaFinderROCm — ROCm wrapper
// ============================================================================

// ROCm-версия поиска максимумов спектра (hipFFT + hiprtc ядра).
// API совместим с PySpectrumMaximaFinder (OpenCL):
//   process()       → ONE_PEAK: один пик на луч с параболической интерполяцией
//   find_all_maxima() → ALL_MAXIMA: все локальные максимумы, формат аналогичен OpenCL
// Инициализация ленивая: Initialize(params) вызывается при первом process/find_all_maxima
// или когда параметры изменились.
class PySpectrumMaximaFinderROCm {
public:
  explicit PySpectrumMaximaFinderROCm(ROCmGPUContext& ctx)
      : ctx_(ctx), proc_(ctx.backend()) {}

  // ── ONE_PEAK: find single best peak per beam ────────────────────────────

  py::object process(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal,
      uint32_t n_point,
      float sample_rate,
      uint32_t antenna_count = 0,
      uint32_t repeat_count = 1)
  {
    auto buf = signal.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (antenna_count == 0) {
      if (buf.ndim == 2)
        antenna_count = static_cast<uint32_t>(buf.shape[0]);
      else
        antenna_count = 1;
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    ensure_init(antenna_count, n_point, sample_rate, repeat_count,
                antenna_fft::PeakSearchMode::ONE_PEAK);

    std::vector<antenna_fft::SpectrumResult> results;
    {
      py::gil_scoped_release release;
      results = proc_.ProcessFromCPU(vec);
    }

    return make_spectrum_result_list(results);
  }

  // ── ALL_MAXIMA: find all local maxima in FFT spectrum ───────────────────

  // Input: готовый FFT спектр (complex64) — НЕ raw signal!
  // Compatible output with PySpectrumMaximaFinder.find_all_maxima() (OpenCL).
  py::object find_all_maxima(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> fft_data,
      uint32_t nFFT,
      float sample_rate,
      uint32_t beam_count = 0,
      uint32_t search_start = 0,
      uint32_t search_end = 0)
  {
    auto buf = fft_data.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (beam_count == 0) {
      if (buf.ndim == 2)
        beam_count = static_cast<uint32_t>(buf.shape[0]);
      else
        beam_count = 1;
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    // AllMaxima не нужен FFT: n_point = nFFT (данные уже FFT)
    ensure_init(beam_count, nFFT, sample_rate, 1,
                antenna_fft::PeakSearchMode::ALL_MAXIMA);

    antenna_fft::AllMaximaResult result;
    {
      py::gil_scoped_release release;
      result = proc_.AllMaximaFromCPU(
          vec, beam_count, nFFT, sample_rate,
          antenna_fft::OutputDestination::CPU,
          search_start, search_end);
    }

    return make_all_maxima_result(result, beam_count);
  }

  // ── find_all_maxima from raw signal (includes FFT step) ─────────────────

  py::object find_all_maxima_from_signal(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal,
      uint32_t n_point,
      float sample_rate,
      uint32_t antenna_count = 0,
      uint32_t repeat_count = 1,
      uint32_t search_start = 0,
      uint32_t search_end = 0)
  {
    auto buf = signal.request();
    auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
    size_t total = static_cast<size_t>(buf.size);

    if (antenna_count == 0) {
      if (buf.ndim == 2)
        antenna_count = static_cast<uint32_t>(buf.shape[0]);
      else
        antenna_count = 1;
    }

    std::vector<std::complex<float>> vec(ptr, ptr + total);

    ensure_init(antenna_count, n_point, sample_rate, repeat_count,
                antenna_fft::PeakSearchMode::ALL_MAXIMA);

    antenna_fft::AllMaximaResult result;
    {
      py::gil_scoped_release release;
      result = proc_.FindAllMaximaFromCPU(
          vec, antenna_fft::OutputDestination::CPU,
          search_start, search_end);
    }

    return make_all_maxima_result(result, antenna_count);
  }

  // ── Info ─────────────────────────────────────────────────────────────────

  bool is_initialized() const { return proc_.IsInitialized(); }

  py::dict get_params() const {
    const auto& p = proc_.GetParams();
    py::dict d;
    d["antenna_count"]  = p.antenna_count;
    d["n_point"]        = p.n_point;
    d["repeat_count"]   = p.repeat_count;
    d["sample_rate"]    = p.sample_rate;
    d["nFFT"]           = p.nFFT;
    return d;
  }

private:
  // ─── Private helpers ─────────────────────────────────────────────────────

  // Ленивая инициализация: Initialize(params) вызывается если параметры изменились
  void ensure_init(uint32_t antenna_count, uint32_t n_point, float sample_rate,
                   uint32_t repeat_count, antenna_fft::PeakSearchMode mode)
  {
    const auto& p = proc_.GetParams();
    if (proc_.IsInitialized()
        && p.antenna_count == antenna_count
        && p.n_point       == n_point
        && p.sample_rate   == sample_rate
        && p.repeat_count  == repeat_count
        && p.peak_mode     == mode) {
      return;  // уже инициализирован с теми же параметрами
    }

    antenna_fft::SpectrumParams params;
    params.antenna_count = antenna_count;
    params.n_point       = n_point;
    params.sample_rate   = sample_rate;
    params.repeat_count  = repeat_count;
    params.peak_mode     = mode;
    proc_.Initialize(params);
  }

  // Serialize ONE_PEAK results → list[dict] (or dict for single beam)
  py::object make_spectrum_result_list(
      const std::vector<antenna_fft::SpectrumResult>& results)
  {
    auto make_beam_dict = [](const antenna_fft::SpectrumResult& r) {
      py::dict d;
      d["antenna_id"] = r.antenna_id;
      d["freq_hz"]    = r.interpolated.refined_frequency;
      d["magnitude"]  = r.interpolated.magnitude;
      d["phase"]      = r.interpolated.phase;
      d["index"]      = r.interpolated.index;
      d["freq_offset"]= r.interpolated.freq_offset;
      return d;
    };

    if (results.size() == 1) return make_beam_dict(results[0]);

    py::list out;
    for (const auto& r : results) out.append(make_beam_dict(r));
    return out;
  }

  // Serialize AllMaximaResult → dict (single) or list[dict] (multi)
  // Format compatible with OpenCL PySpectrumMaximaFinder.find_all_maxima()
  py::object make_all_maxima_result(
      const antenna_fft::AllMaximaResult& result, uint32_t beam_count)
  {
    auto make_beam = [](const antenna_fft::AllMaximaBeamResult& b) {
      std::vector<uint32_t> positions;  positions.reserve(b.maxima.size());
      std::vector<float>    magnitudes; magnitudes.reserve(b.maxima.size());
      std::vector<float>    frequencies; frequencies.reserve(b.maxima.size());
      for (const auto& m : b.maxima) {
        positions.push_back(m.index);
        magnitudes.push_back(m.magnitude);
        frequencies.push_back(m.refined_frequency);
      }
      py::dict d;
      d["antenna_id"]  = b.antenna_id;
      d["positions"]   = vector_to_numpy(std::move(positions));
      d["magnitudes"]  = vector_to_numpy(std::move(magnitudes));
      d["frequencies"] = vector_to_numpy(std::move(frequencies));
      d["num_maxima"]  = b.num_maxima;
      return d;
    };

    if (beam_count == 1 && result.beams.size() == 1)
      return make_beam(result.beams[0]);

    py::list out;
    for (const auto& b : result.beams) out.append(make_beam(b));
    return out;
  }

  ROCmGPUContext& ctx_;
  antenna_fft::SpectrumProcessorROCm proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_spectrum_maxima_finder_rocm(py::module& m) {
  py::class_<PySpectrumMaximaFinderROCm>(m, "SpectrumMaximaFinderROCm",
      "Spectrum maxima finder on ROCm/HIP GPU (hipFFT + hiprtc).\n\n"
      "API compatible with SpectrumMaximaFinder (OpenCL).\n\n"
      "Methods:\n"
      "  process()                    — ONE_PEAK: best peak per beam with parabolic interpolation\n"
      "  find_all_maxima()            — ALL_MAXIMA from FFT spectrum (no FFT step)\n"
      "  find_all_maxima_from_signal()— ALL_MAXIMA from raw signal (includes FFT step)\n\n"
      "Usage:\n"
      "  finder = gpuworklib.SpectrumMaximaFinderROCm(ctx)\n"
      "  result = finder.process(signal, n_point=1024, sample_rate=1e6)\n"
      "  maxima = finder.find_all_maxima(fft_spec, nFFT=1024, sample_rate=1e6)\n")
      .def(py::init<ROCmGPUContext&>(), py::arg("ctx"),
           "Create SpectrumMaximaFinderROCm bound to ROCm GPU context.")

      .def("process", &PySpectrumMaximaFinderROCm::process,
           py::arg("signal"),
           py::arg("n_point"),
           py::arg("sample_rate"),
           py::arg("antenna_count") = 0,
           py::arg("repeat_count")  = 1,
           "Find single best peak per beam (ONE_PEAK + parabolic interpolation).\n\n"
           "Args:\n"
           "  signal: numpy complex64 (n_point,) or (antenna_count, n_point) — raw signal\n"
           "  n_point: samples per beam\n"
           "  sample_rate: Hz\n"
           "  antenna_count: 0 = auto-detect\n"
           "  repeat_count: zero-padding factor (nFFT = nextPow2(n_point) * repeat_count)\n\n"
           "Returns:\n"
           "  dict (single beam) or list[dict] (multi):\n"
           "  {antenna_id, freq_hz, magnitude, phase, index, freq_offset}")

      .def("find_all_maxima", &PySpectrumMaximaFinderROCm::find_all_maxima,
           py::arg("fft_data"),
           py::arg("nFFT"),
           py::arg("sample_rate"),
           py::arg("beam_count")    = 0,
           py::arg("search_start")  = 0,
           py::arg("search_end")    = 0,
           "Find ALL local maxima in ready FFT spectrum (no FFT step).\n\n"
           "Args:\n"
           "  fft_data: numpy complex64 (nFFT,) or (beam_count, nFFT) — FFT result\n"
           "  nFFT: FFT size (= n_point for power-of-2 input)\n"
           "  sample_rate: Hz\n"
           "  search_start/end: bin range (0 = auto = [1, nFFT/2])\n\n"
           "Returns:\n"
           "  dict or list[dict]: {antenna_id, positions, magnitudes, frequencies, num_maxima}\n"
           "  Compatible with SpectrumMaximaFinder.find_all_maxima() (OpenCL).")

      .def("find_all_maxima_from_signal",
           &PySpectrumMaximaFinderROCm::find_all_maxima_from_signal,
           py::arg("signal"),
           py::arg("n_point"),
           py::arg("sample_rate"),
           py::arg("antenna_count") = 0,
           py::arg("repeat_count")  = 1,
           py::arg("search_start")  = 0,
           py::arg("search_end")    = 0,
           "Find ALL local maxima from raw signal (performs FFT internally).\n\n"
           "Args:\n"
           "  signal: numpy complex64 (n_point,) or (antenna_count, n_point)\n"
           "  n_point: samples per beam\n"
           "  sample_rate: Hz\n\n"
           "Returns:\n"
           "  dict or list[dict]: {antenna_id, positions, magnitudes, frequencies, num_maxima}")

      .def_property_readonly("initialized", &PySpectrumMaximaFinderROCm::is_initialized,
           "True if processor is initialized (Initialize() was called).")

      .def("get_params", &PySpectrumMaximaFinderROCm::get_params,
           "Get current processor parameters.\n\n"
           "Returns dict: {antenna_count, n_point, repeat_count, sample_rate, nFFT}")

      .def("__repr__", [](const PySpectrumMaximaFinderROCm& self) {
          return std::string("<SpectrumMaximaFinderROCm initialized=") +
                 (self.is_initialized() ? "True" : "False") + ">";
      });
}
