#pragma once

/**
 * @file filter_types.hpp
 * @brief Общие алиасы фильтров (ComplexVector, ComplexVector2D).
 *
 * @note Тип B (technical header): только typedef-алиасы, без struct/enum/логики.
 *
 * История:
 *   - Создан:  2026-02-18
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
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
