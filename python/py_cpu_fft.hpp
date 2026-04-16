#pragma once

/**
 * @file py_cpu_fft.hpp
 * @brief Python wrapper for CpuFFT (pocketfft CPU reference)
 *
 * Python API:
 *   import dsp_spectrum
 *
 *   # Complex-to-Complex
 *   spectrum = dsp_spectrum.cpu_fft_c2c(data)          # forward
 *   signal   = dsp_spectrum.cpu_ifft_c2c(spectrum)     # inverse (1/N)
 *
 *   # Real-to-Complex
 *   half     = dsp_spectrum.cpu_fft_r2c(real_data)     # N/2+1 bins
 *   full     = dsp_spectrum.cpu_fft_r2c_full(real_data) # N bins (mirror)
 *
 *   # Magnitude
 *   mag      = dsp_spectrum.magnitude(spectrum)         # |X|
 *   mag2     = dsp_spectrum.magnitude(spectrum, squared=True) # |X|²
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-16
 */

#include <spectrum/utils/cpu_fft.hpp>

inline void register_cpu_fft(py::module_& m) {

  m.def("cpu_fft_c2c",
    [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data)
        -> py::array_t<std::complex<float>>
    {
      auto buf = data.request();
      auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
      std::vector<std::complex<float>> vec(ptr, ptr + buf.size);
      auto result = spectrum_utils::CpuFFT::ForwardC2C(vec);
      return vector_to_numpy(std::move(result));
    },
    py::arg("data"),
    "Forward FFT complex→complex (CPU, pocketfft).\n"
    "Input: np.ndarray complex64 [N]\n"
    "Output: np.ndarray complex64 [N]");

  m.def("cpu_ifft_c2c",
    [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> data)
        -> py::array_t<std::complex<float>>
    {
      auto buf = data.request();
      auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
      std::vector<std::complex<float>> vec(ptr, ptr + buf.size);
      auto result = spectrum_utils::CpuFFT::InverseC2C(vec);
      return vector_to_numpy(std::move(result));
    },
    py::arg("data"),
    "Inverse FFT complex→complex with 1/N normalization (CPU, pocketfft).\n"
    "Input: np.ndarray complex64 [N] (spectrum)\n"
    "Output: np.ndarray complex64 [N] (signal)");

  m.def("cpu_fft_r2c",
    [](py::array_t<float, py::array::c_style | py::array::forcecast> data)
        -> py::array_t<std::complex<float>>
    {
      auto buf = data.request();
      auto* ptr = static_cast<float*>(buf.ptr);
      std::vector<float> vec(ptr, ptr + buf.size);
      auto result = spectrum_utils::CpuFFT::ForwardR2C(vec);
      return vector_to_numpy(std::move(result));
    },
    py::arg("data"),
    "Forward FFT real→complex, half-spectrum (CPU, pocketfft).\n"
    "Input: np.ndarray float32 [N]\n"
    "Output: np.ndarray complex64 [N/2+1]");

  m.def("cpu_fft_r2c_full",
    [](py::array_t<float, py::array::c_style | py::array::forcecast> data)
        -> py::array_t<std::complex<float>>
    {
      auto buf = data.request();
      auto* ptr = static_cast<float*>(buf.ptr);
      std::vector<float> vec(ptr, ptr + buf.size);
      auto result = spectrum_utils::CpuFFT::ForwardR2C_Full(vec);
      return vector_to_numpy(std::move(result));
    },
    py::arg("data"),
    "Forward FFT real→complex, full spectrum with mirror (CPU, pocketfft).\n"
    "Input: np.ndarray float32 [N]\n"
    "Output: np.ndarray complex64 [N]");

  m.def("magnitude",
    [](py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> spectrum,
       bool squared)
        -> py::array_t<float>
    {
      auto buf = spectrum.request();
      auto* ptr = static_cast<std::complex<float>*>(buf.ptr);
      std::vector<std::complex<float>> vec(ptr, ptr + buf.size);
      auto result = spectrum_utils::CpuFFT::Magnitude(vec, squared);
      return vector_to_numpy(std::move(result));
    },
    py::arg("spectrum"),
    py::arg("squared") = false,
    "Compute magnitude from complex spectrum.\n"
    "squared=False: |X| (default)\n"
    "squared=True:  |X|² (power spectrum)");
}
