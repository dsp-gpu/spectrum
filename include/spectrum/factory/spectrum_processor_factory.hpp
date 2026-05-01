#pragma once

// ============================================================================
// SpectrumProcessorFactory — Factory Method для ISpectrumProcessor
//
// ЧТО:    Статическая фабрика, создающая конкретную реализацию
//         ISpectrumProcessor по запрошенному BackendType (ROCm / OPENCL).
//         Возвращает unique_ptr, никогда не null (бросает при unsupported).
//
// ЗАЧЕМ:  Клиентский код (Python биндинги, тесты, radar pipeline) не должен
//         знать про конкретные классы SpectrumProcessorROCm / *OpenCL —
//         только про интерфейс. Factory скрывает выбор backend'а и
//         платформенно-зависимую сборку (#if ENABLE_ROCM / OPENCL).
//
// ПОЧЕМУ: - Factory Method (GoF) + GRASP Creator: factory знает контекст
//           создания (какой backend доступен, какой ctor вызвать).
//         - Static-only метод (без состояния) → нет нужды в инстансе фабрики.
//         - unique_ptr<ISpectrumProcessor> → caller владеет, RAII освобождает.
//         - throw runtime_error при unsupported backend → fail-fast лучше
//           чем return nullptr (caller не забудет проверить).
//
// Использование:
//   // main-ветка, Linux + AMD + ROCm 7.2+:
//   auto proc = SpectrumProcessorFactory::Create(
//       drv_gpu_lib::BackendType::ROCm, backend);
//   proc->ProcessFFT(input, params);
//
// История:
//   - Создан: 2026-02-15 (Phase 2 рефакторинга SpectrumMaximaFinder)
// ============================================================================

#include <spectrum/interface/i_spectrum_processor.hpp>
#include <spectrum/interface/spectrum_input_data.hpp>
#include <core/interface/i_backend.hpp>

#include <memory>

namespace antenna_fft {

/**
 * @class SpectrumProcessorFactory
 * @brief Фабрика создания ISpectrumProcessor по BackendType.
 *
 * @note Static-only: инстансы не создаются.
 * @note Бросает std::runtime_error если backend не поддерживается на платформе.
 * @see ISpectrumProcessor
 * @see drv_gpu_lib::BackendType
 */
class SpectrumProcessorFactory {
public:
    /**
     * @brief Создать процессор спектра для заданного backend'а.
     * @param backend_type ROCm (main-ветка) или OPENCL (legacy nvidia-ветка).
     *                     AUTO выбирает ROCm на Linux, OPENCL на Windows.
     * @param backend non-owning указатель на DrvGPU backend (должен жить дольше processor'а).
     * @return unique_ptr на процессор, никогда не null.
     * @throws std::runtime_error если backend_type не поддерживается на текущей платформе.
     */
    static std::unique_ptr<ISpectrumProcessor> Create(
        drv_gpu_lib::BackendType backend_type,
        drv_gpu_lib::IBackend* backend);
};

} // namespace antenna_fft
