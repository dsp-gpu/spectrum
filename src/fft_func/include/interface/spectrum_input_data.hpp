#pragma once

/**
 * @file spectrum_input_data.hpp
 * @brief Обёртка над DrvGPU/interface — InputData, OutputDestination, type traits
 *
 * Реальные типы в DrvGPU/interface. Здесь — re-export для antenna_fft.
 * DriverType заменён на drv_gpu_lib::BackendType.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include "interface/input_data.hpp"
#include "interface/output_destination.hpp"
#include "interface/input_data_traits.hpp"
#include "common/backend_type.hpp"

namespace antenna_fft {

using drv_gpu_lib::InputData;
using drv_gpu_lib::ProcessingParams;
using drv_gpu_lib::OutputDestination;
using drv_gpu_lib::BackendType;
using drv_gpu_lib::is_cpu_vector;
using drv_gpu_lib::is_cpu_vector_v;
using drv_gpu_lib::is_svm_pointer;
using drv_gpu_lib::is_svm_pointer_v;

/// Обратная совместимость: DriverType = BackendType (OPENCL, ROCm, AUTO)
using DriverType = drv_gpu_lib::BackendType;

}  // namespace antenna_fft
