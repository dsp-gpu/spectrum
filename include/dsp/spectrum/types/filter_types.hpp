#pragma once

/**
 * @file filter_types.hpp
 * @brief Result types and common typedefs for filter module
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include <complex>
#include <vector>
#include <cstdint>

namespace filters {

/// Alias for complex float vector (flat: channels * points)
using ComplexVector = std::vector<std::complex<float>>;

/// Alias for 2D complex data [channels][points]
using ComplexVector2D = std::vector<std::vector<std::complex<float>>>;

} // namespace filters
