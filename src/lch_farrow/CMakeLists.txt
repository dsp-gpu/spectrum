# ════════════════════════════════════════════════════════════════════════════
# LCH Farrow Module - CMakeLists.txt (ROCm-only)
# ════════════════════════════════════════════════════════════════════════════

cmake_minimum_required(VERSION 3.20)

set(MODULE_NAME lch_farrow)

if(NOT ROCM_ENABLED)
    message(STATUS "⏭️  ${MODULE_NAME}: skipped (ROCm disabled)")
    return()
endif()

message(STATUS "Configuring module: ${MODULE_NAME} (ROCm)")

add_library(${MODULE_NAME} STATIC
    src/lch_farrow_rocm.cpp
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
add_library(GPUWorkLib::${MODULE_NAME} ALIAS ${MODULE_NAME})

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

message(STATUS "  ${MODULE_NAME}: configured (ROCm only)")
