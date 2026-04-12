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

#include "py_filters.hpp"
#include "py_lch_farrow.hpp"

#if ENABLE_ROCM
#include "py_fft_processor_rocm.hpp"
#include "py_spectrum_maxima_finder_rocm.hpp"
#include "py_complex_to_mag_rocm.hpp"
#include "py_filters_rocm.hpp"
#include "py_filters_adaptive_rocm.hpp"
#include "py_lch_farrow_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_spectrum, m) {
    m.doc() = "dsp::spectrum — FFT, filters, fractional delay (LchFarrow)\n\n"
              "Classes:\n"
              "  FFTProcessorROCm         - hipFFT spectrum processor (ROCm)\n"
              "  SpectrumMaximaFinderROCm - spectral maxima search (ROCm)\n"
              "  ComplexToMagROCm         - complex → magnitude (ROCm)\n"
              "  FirFilter / IirFilter    - FIR/IIR filters (OpenCL)\n"
              "  FirFilterROCm / IirFilterROCm / LchFarrowROCm - ROCm filters\n";

    register_filters(m);
    register_lch_farrow(m);

#if ENABLE_ROCM
    register_fft_processor_rocm(m);
    register_spectrum_maxima_finder_rocm(m);
    register_complex_to_mag_rocm(m);
    register_fir_filter_rocm(m);
    register_iir_filter_rocm(m);
    register_filters_adaptive_rocm(m);
    register_lch_farrow_rocm(m);
#endif
}
