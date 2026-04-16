/**
 * @file dsp_spectrum_module.cpp
 * @brief pybind11 bindings for dsp::spectrum
 *
 * Python API:
 *   import dsp_spectrum
 *   proc = dsp_spectrum.FFTProcessorROCm(ctx)
 *   spec = proc.process_complex(data, sample_rate=1e6)
 *
 * Экспортируемые классы:
 *   FFTProcessorROCm         — FFT через hipFFT
 *   SpectrumMaximaFinderROCm — поиск максимумов спектра
 *   ComplexToMagROCm         — |FFT| без фаз
 *   FirFilter / IirFilter    — фильтры (OpenCL)
 *   FirFilterROCm / IirFilterROCm / MovingAverageFilterROCm
 *   KalmanFilterROCm / KaufmanFilterROCm — адаптивные фильтры
 *   LchFarrow / LchFarrowROCm — дробная задержка Лагранжа
 */

#include "py_helpers.hpp"

#include "py_cpu_fft.hpp"

// ROCm GPU биндинги — используют ROCmGPUContext из core/python/py_gpu_context.hpp
#if ENABLE_ROCM
#include "py_gpu_context.hpp"
#include "py_fft_processor_rocm.hpp"
#include "py_spectrum_maxima_finder_rocm.hpp"
#include "py_complex_to_mag_rocm.hpp"
#include "py_filters_rocm.hpp"
#include "py_filters_adaptive_rocm.hpp"
#include "py_lch_farrow_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_spectrum, m) {
    m.doc() = "dsp::spectrum — FFT, filters, fractional delay (LchFarrow)\n\n"
              "Functions (CPU, pocketfft):\n"
              "  cpu_fft_c2c / cpu_ifft_c2c  - complex FFT forward/inverse\n"
              "  cpu_fft_r2c / cpu_fft_r2c_full - real→complex FFT\n"
              "  magnitude                   - |X| or |X|² from spectrum\n\n"
              "Classes (ROCm GPU):\n"
              "  FFTProcessorROCm         - hipFFT spectrum processor\n"
              "  SpectrumMaximaFinderROCm - spectral maxima search\n"
              "  ComplexToMagROCm         - complex → magnitude\n"
              "  FirFilterROCm / IirFilterROCm / LchFarrowROCm - GPU filters\n";

    register_cpu_fft(m);

#if ENABLE_ROCM
    // ROCmGPUContext — создаёт GPU backend, передаётся в конструкторы процессоров
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context (creates HIP backend for AMD GPU).\n\n"
        "Usage:\n"
        "  ctx = dsp_spectrum.ROCmGPUContext(device_index=0)\n"
        "  fft = dsp_spectrum.FFTProcessorROCm(ctx)\n")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index);

    register_fft_processor_rocm(m);
    register_spectrum_maxima_finder_rocm(m);
    register_complex_to_mag_rocm(m);
    register_fir_filter_rocm(m);
    register_iir_filter_rocm(m);
    register_filters_adaptive_rocm(m);
    register_lch_farrow_rocm(m);
#endif
}
