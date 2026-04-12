# ════════════════════════════════════════════════════════════════════════════
# fft_func Module - CMakeLists.txt  [BRANCH: main]
# ════════════════════════════════════════════════════════════════════════════
#
# Ветка: main (Linux / AMD GPU / ROCm + hipFFT)
#
# Реализации:
#   ROCm/hipFFT — FFTProcessorROCm, SpectrumMaximaFinderROCm
#   OpenCL/clFFT — только в ветке nvidia
#
# CMake-таргеты:
#   fft_func             — основной
#   GPUWorkLib::fft_func — алиас
#
# Сборка: cmake --preset Debian-Radeon9070
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME fft_func)

message(STATUS "Configuring module: ${MODULE_NAME} [main branch - ROCm/hipFFT]")

# ════════════════════════════════════════════════════════════════════════════
# Зависимости
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED OR NOT HIPFFT_FOUND)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm/hipFFT not available on this platform)")
    return()
endif()

# ════════════════════════════════════════════════════════════════════════════
# Sources — только ROCm/hipFFT
# ════════════════════════════════════════════════════════════════════════════

set(FFT_FUNC_SOURCES
    src/fft_processor_rocm.cpp
    src/complex_to_mag_phase_rocm.cpp
    src/spectrum_processor_rocm.cpp
    src/all_maxima_pipeline_rocm.cpp
    src/spectrum_processor_factory.cpp
)

# ════════════════════════════════════════════════════════════════════════════
# Library
# ════════════════════════════════════════════════════════════════════════════

add_library(${MODULE_NAME} STATIC ${FFT_FUNC_SOURCES})

set_property(TARGET ${MODULE_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ─── ROCm / hipFFT linkage ───────────────────────────────────────────────────
if(hipfft_FOUND)
    target_link_libraries(${MODULE_NAME} PUBLIC hip::hipfft)
else()
    target_link_libraries(${MODULE_NAME} PUBLIC ${HIPFFT_LIB})
    target_include_directories(${MODULE_NAME} PRIVATE ${HIPFFT_INCLUDE_DIR})
endif()

set_target_properties(${MODULE_NAME} PROPERTIES
    LINK_INTERFACE_MULTIPLICITY 2
)

# ════════════════════════════════════════════════════════════════════════════
# Aliases
# ════════════════════════════════════════════════════════════════════════════

add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

# ════════════════════════════════════════════════════════════════════════════
# Install
# ════════════════════════════════════════════════════════════════════════════

install(TARGETS ${MODULE_NAME}
    EXPORT GPUWorkLibTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/${MODULE_NAME}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

message(STATUS "  ${MODULE_NAME}: configured [main branch]")
message(STATUS "    - FFTProcessorROCm (hipFFT)")
message(STATUS "    - ComplexToMagPhaseROCm")
message(STATUS "    - SpectrumProcessorROCm, AllMaximaPipelineROCm")
message(STATUS "    - SpectrumProcessorFactory (ROCm)")
