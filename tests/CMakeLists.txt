# ── DspSpectrum tests ──────────────────────────────────────────────
add_executable(test_spectrum_main
  main.cpp
)
target_link_libraries(test_spectrum_main PRIVATE
  DspSpectrum::DspSpectrum
  DspCore::TestUtils
)
target_include_directories(test_spectrum_main PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(test_spectrum_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>
)
