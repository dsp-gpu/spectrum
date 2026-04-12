cmake_minimum_required(VERSION 3.25)
project(DspSpectrum VERSION 0.1.0 LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_SPECTRUM_BUILD_TESTS  "Build tests"            ON)
option(DSP_SPECTRUM_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip    REQUIRED)   # lowercase — Linux case-sensitive!
find_package(hipfft REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspSpectrum STATIC)
add_library(DspSpectrum::DspSpectrum ALIAS DspSpectrum)

target_sources(DspSpectrum PRIVATE
  # fft_func
  src/fft_func/src/fft_processor_rocm.cpp
  src/fft_func/src/complex_to_mag_phase_rocm.cpp
  src/fft_func/src/spectrum_processor_rocm.cpp
  src/fft_func/src/all_maxima_pipeline_rocm.cpp
  src/fft_func/src/spectrum_processor_factory.cpp
  # filters
  src/filters/src/fir_filter_rocm.cpp
  src/filters/src/iir_filter_rocm.cpp
  src/filters/src/kalman_filter_rocm.cpp
  src/filters/src/kaufman_filter_rocm.cpp
  src/filters/src/moving_average_filter_rocm.cpp
  # lch_farrow
  src/lch_farrow/src/lch_farrow_rocm.cpp
)

target_include_directories(DspSpectrum
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # Внутренние include каждого субмодуля
    src/fft_func/include
    src/filters/include
    src/lch_farrow/include
    # Kernels — hiprtc runtime, не компилируем, но include нужен
    kernels/
    src
)
target_link_libraries(DspSpectrum PUBLIC DspCore::DspCore hip::hipfft)
target_compile_features(DspSpectrum PUBLIC cxx_std_17)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspSpectrum EXPORT DspSpectrumTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspSpectrumConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspSpectrumTargets NAMESPACE DspSpectrum::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspSpectrumConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspSpectrum
)

if(DSP_SPECTRUM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_SPECTRUM_BUILD_PYTHON)
  add_subdirectory(python)
endif()
