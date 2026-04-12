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
  # Phase 1 skeleton — sources added in Phase 2
)

target_include_directories(DspSpectrum
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
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
