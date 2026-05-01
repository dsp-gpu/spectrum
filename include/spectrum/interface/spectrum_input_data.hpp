#pragma once

/**
 * @file spectrum_input_data.hpp
 * @brief Re-export типов core/interface (InputData, OutputDestination, traits) в namespace antenna_fft.
 */

#include <core/interface/input_data.hpp>
#include <core/interface/output_destination.hpp>
#include <core/interface/input_data_traits.hpp>
#include <core/common/backend_type.hpp>

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
