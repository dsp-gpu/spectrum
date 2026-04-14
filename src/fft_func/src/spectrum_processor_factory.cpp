/**
 * @file spectrum_processor_factory.cpp
 * @brief Factory implementation for ISpectrumProcessor — ветка main (ROCm only)
 */

#include <spectrum/factory/spectrum_processor_factory.hpp>
#include <spectrum/processors/spectrum_processor_rocm.hpp>

#include <stdexcept>

namespace antenna_fft {

std::unique_ptr<ISpectrumProcessor> SpectrumProcessorFactory::Create(
    drv_gpu_lib::BackendType backend_type,
    drv_gpu_lib::IBackend* backend)
{
    if (!backend) {
        throw std::invalid_argument("SpectrumProcessorFactory: backend cannot be null");
    }

    auto effective = backend_type;
    if (effective == drv_gpu_lib::BackendType::AUTO ||
        effective == drv_gpu_lib::BackendType::OPENCLandROCm)
    {
        effective = drv_gpu_lib::BackendType::ROCm;
    }

    switch (effective) {
    case drv_gpu_lib::BackendType::ROCm:
        return std::make_unique<SpectrumProcessorROCm>(backend);
    case drv_gpu_lib::BackendType::OPENCL:
        throw std::runtime_error("SpectrumProcessorFactory: OpenCL not supported on this branch (use nvidia branch with ROCm disabled)");
    default:
        break;
    }
    throw std::invalid_argument("SpectrumProcessorFactory: unknown BackendType");
}

} // namespace antenna_fft
