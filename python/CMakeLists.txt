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

# ── Auto-deploy to DSP/Python/libs/ ─────────────────────────────────
# Опция определена в 8 файлах (Q1=B, автономность модуля);
# первое определение создаёт CACHE-переменную, остальные — no-op.
option(DSP_DEPLOY_PYTHON_LIB "Auto-copy .so to DSP/Python/libs/" ON)

# Путь от {repo}/python/CMakeLists.txt → корень репо → sibling DSP/Python/libs/
# CMAKE_CURRENT_LIST_DIR — устойчив к супер-проекту (см. ревью §B4)
set(DSP_PYTHON_LIB_DIR "${CMAKE_CURRENT_LIST_DIR}/../../DSP/Python/libs"
    CACHE PATH "Where to deploy compiled .so for tests")

if(DSP_DEPLOY_PYTHON_LIB)
  # PRE_BUILD: удалить старую версию (даже если сборка упадёт — не будет stale)
  add_custom_command(TARGET dsp_spectrum PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -f
      "${DSP_PYTHON_LIB_DIR}/$<TARGET_FILE_NAME:dsp_spectrum>"
    COMMENT "Remove stale dsp_spectrum from DSP/Python/libs/")

  # POST_BUILD: скопировать новую (только при успешной сборке)
  add_custom_command(TARGET dsp_spectrum POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DSP_PYTHON_LIB_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:dsp_spectrum>" "${DSP_PYTHON_LIB_DIR}/"
    COMMENT "Deploy dsp_spectrum to DSP/Python/libs/")
endif()
