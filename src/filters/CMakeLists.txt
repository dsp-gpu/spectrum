# ════════════════════════════════════════════════════════════════════════════
# Filters Module - CMakeLists.txt
# ════════════════════════════════════════════════════════════════════════════
#
# GPU FIR & IIR filters for multi-channel parallel signal processing
#   - FIR: Direct-form convolution, __constant coefficients
#   - IIR: Biquad cascade, Direct Form II Transposed
#
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME filters)

message(STATUS "Configuring module: ${MODULE_NAME}")

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Library (ROCm-only: OpenCL filters removed from main branch)
# ════════════════════════════════════════════════════════════════════════════

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

add_library(${MODULE_NAME} STATIC
    src/fir_filter_rocm.cpp
    src/iir_filter_rocm.cpp
    src/moving_average_filter_rocm.cpp
    src/kalman_filter_rocm.cpp
    src/kaufman_filter_rocm.cpp
)

target_include_directories(${MODULE_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/DrvGPU
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
        drvgpu
        hip::host
)

target_compile_features(${MODULE_NAME} PUBLIC cxx_std_17)

# ════════════════════════════════════════════════════════════════════════════
# Kernel cache paths (for on-disk kernel storage via KernelCacheService)
# ════════════════════════════════════════════════════════════════════════════

set(FILTERS_KERNELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")

target_compile_definitions(${MODULE_NAME} PRIVATE
    FILTERS_KERNELS_DIR="${FILTERS_KERNELS_DIR}"
)

# Alias
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

message(STATUS "  ${MODULE_NAME}: configured")
message(STATUS "    - FIR: direct-form, __constant coeffs, multi-channel parallel")
message(STATUS "    - IIR: biquad cascade, DFII-T, per-channel parallel")
message(STATUS "    - GPUProfiler integration")
message(STATUS "    - Kernels dir: ${FILTERS_KERNELS_DIR}")
