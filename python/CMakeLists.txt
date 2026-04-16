# ── DspSpectrum Python bindings ────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dsp_spectrum dsp_spectrum_module.cpp)
target_link_libraries(dsp_spectrum PRIVATE DspSpectrum::DspSpectrum)
target_include_directories(dsp_spectrum PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/fft_func/include
  ${CMAKE_SOURCE_DIR}/src/filters/include
  ${CMAKE_SOURCE_DIR}/src/lch_farrow/include
  ${CMAKE_SOURCE_DIR}/third_party/pocketfft
  # py_gpu_context.hpp — ROCmGPUContext shared header
  ${FETCHCONTENT_SOURCE_DIR_DSPCORE}/python)
target_compile_definitions(dsp_spectrum PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
install(TARGETS dsp_spectrum DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
