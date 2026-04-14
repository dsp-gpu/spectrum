cmake_minimum_required(VERSION 3.25)
project(DspSpectrum VERSION 0.1.0
  DESCRIPTION "FFT processing, FIR/IIR filters, Farrow interpolation"
  LANGUAGES CXX HIP)

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
    # Стандарт AMD: #include <spectrum/operations/...>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    # Kernels — hiprtc runtime, не компилируем, но include нужен
    kernels/
    src
)
target_link_libraries(DspSpectrum PUBLIC DspCore::DspCore hip::hipfft)
target_compile_features(DspSpectrum PUBLIC cxx_std_17)

# Прокинуть ENABLE_ROCM=1 — нужно для #if ENABLE_ROCM блоков в коде.
# PUBLIC чтобы клиенты (downstream модули) тоже видели.
if(ENABLE_ROCM)
  target_compile_definitions(DspSpectrum PUBLIC ENABLE_ROCM=1)
endif()

# ── Git Version ────────────────────────────────────────────────────
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

add_custom_target(git_version_DspSpectrum ALL
    COMMAND ${CMAKE_COMMAND}
        -D SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D MODULE_PREFIX=DSPSPECTRUM
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
    COMMENT "[DspSpectrum] Checking git version..."
    VERBATIM
)
add_dependencies(DspSpectrum git_version_DspSpectrum)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspSpectrum EXPORT DspSpectrumTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DspSpectrum
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/DspSpectrum
)

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
