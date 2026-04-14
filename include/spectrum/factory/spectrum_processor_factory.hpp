#pragma once

/**
 * @file spectrum_processor_factory.hpp
 * @brief Factory for creating ISpectrumProcessor by BackendType
 *
 * GRASP: Creator — factory creates processor instances.
 * Part of SpectrumMaximaFinder refactoring (Phase 2).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-15
 */

#include <spectrum/interface/i_spectrum_processor.hpp>
#include <spectrum/interface/spectrum_input_data.hpp>
#include <core/interface/i_backend.hpp>

#include <memory>

namespace antenna_fft {

/**
 * @class SpectrumProcessorFactory
 * @brief Creates ISpectrumProcessor by BackendType
 *
 * Usage:
 *   auto proc = SpectrumProcessorFactory::Create(BackendType::OPENCL, backend);
 *   auto proc = SpectrumProcessorFactory::Create(BackendType::ROCm, backend);  // stub
 */
class SpectrumProcessorFactory {
public:
/**
 * @brief Create processor for given backend type
 * @param backend_type OPENCL or ROCm (AUTO defaults to OPENCL)
 * @param backend DrvGPU backend (non-owning)
 * @return unique_ptr to processor, never null
 * @throws std::runtime_error if ROCm requested (not implemented)
 */
    static std::unique_ptr<ISpectrumProcessor> Create(
        drv_gpu_lib::BackendType backend_type,
        drv_gpu_lib::IBackend* backend);
};

} // namespace antenna_fft
