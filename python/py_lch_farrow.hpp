#pragma once

/**
 * @file py_lch_farrow.hpp
 * @brief Python wrapper for LchFarrow (Lagrange 48x5 fractional delay)
 *
 * Standalone fractional delay processor.
 * Independent from signal generators — works with any complex input.
 *
 * Include AFTER GPUContext and vector_to_numpy definitions.
 *
 * @author Kodo (AI Assistant)
 * @date 2026-02-18
 */

#include <spectrum/lch_farrow.hpp>

// ============================================================================
// PyLchFarrow — standalone fractional delay processor
// ============================================================================

// Применяет точную дробную задержку к входному сигналу через Lagrange-интерполяцию.
// Матрица 48×5: 48 строк = 48 субинтервалов фракционной части (шаг 1/48),
//               5 столбцов = 5 коэффициентов полинома 4-й степени.
// Принимает любой complex64 сигнал — независим от генератора сигналов.
class PyLchFarrow {
public:
    explicit PyLchFarrow(GPUContext& ctx)
        : ctx_(ctx), proc_(ctx.backend()) {}

    void set_delays(const std::vector<float>& delay_us) {
        proc_.SetDelays(delay_us);
    }

    void set_sample_rate(float sample_rate) {
        proc_.SetSampleRate(sample_rate);
    }

    // norm_val = 1/sqrt(2) ≈ 0.7071 — нормировка RMS шума (Philox + Box-Muller даёт σ≈1,
    // умножаем на norm_val чтобы RMS = noise_amplitude). noise_seed = 0 → детерминировано.
    void set_noise(float noise_amplitude, float norm_val = 0.7071067811865476f,
                   uint32_t noise_seed = 0) {
        proc_.SetNoise(noise_amplitude, norm_val, noise_seed);
    }

    void load_matrix(const std::string& json_path) {
        proc_.LoadMatrix(json_path);
    }

    py::array_t<std::complex<float>> process(
        py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> input)
    {
        auto buf = input.request();
        uint32_t antennas, points;

        if (buf.ndim == 2) {
            antennas = static_cast<uint32_t>(buf.shape[0]);
            points = static_cast<uint32_t>(buf.shape[1]);
        } else if (buf.ndim == 1) {
            antennas = 1;
            points = static_cast<uint32_t>(buf.shape[0]);
        } else {
            throw std::invalid_argument("Input must be 1D or 2D");
        }

        size_t total = static_cast<size_t>(antennas) * points;
        auto* ptr = static_cast<std::complex<float>*>(buf.ptr);

        // Upload to GPU: LchFarrow.Process() ожидает cl_mem (как FirFilter).
        cl_context cl_ctx = static_cast<cl_context>(ctx_.backend()->GetNativeContext());
        cl_int err;
        cl_mem input_buf = clCreateBuffer(cl_ctx,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            total * sizeof(std::complex<float>),
            const_cast<std::complex<float>*>(ptr), &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("PyLchFarrow: upload failed (" +
                                     std::to_string(err) + ")");

        drv_gpu_lib::InputData<cl_mem> result;
        {
            py::gil_scoped_release release;
            result = proc_.Process(input_buf, antennas, points);
        }

        clReleaseMemObject(input_buf);

        // Readback: result.data — cl_mem, caller owns (LchFarrow не освобождает)
        std::vector<std::complex<float>> data(total);
        clEnqueueReadBuffer(ctx_.queue(), result.data, CL_TRUE, 0,
                            total * sizeof(std::complex<float>),
                            data.data(), 0, nullptr, nullptr);
        clReleaseMemObject(result.data);

        if (antennas <= 1) {
            return vector_to_numpy(std::move(data));
        }
        return vector_to_numpy_2d(std::move(data), antennas, points);
    }

    float sample_rate() const { return proc_.GetSampleRate(); }

    py::list get_delays() const {
        py::list result;
        for (float d : proc_.GetDelays()) result.append(d);
        return result;
    }

private:
    GPUContext& ctx_;
    lch_farrow::LchFarrow proc_;
};

// ============================================================================
// Binding registration
// ============================================================================

inline void register_lch_farrow(py::module& m) {
    py::class_<PyLchFarrow>(m, "LchFarrow",
        "Standalone fractional delay processor (Lagrange 48x5).\n\n"
        "Applies per-antenna fractional delay to an existing complex signal.\n\n"
        "Algorithm:\n"
        "  delay_samples = delay_us * 1e-6 * sample_rate\n"
        "  D = floor(delay_samples), mu = frac(delay_samples)\n"
        "  row = int(mu * 48) %% 48\n"
        "  output[n] = sum(L[row][k] * input[n - D - 1 + k], k=0..4)\n\n"
        "Usage:\n"
        "  proc = gpuworklib.LchFarrow(ctx)\n"
        "  proc.set_sample_rate(1e6)\n"
        "  proc.set_delays([0.0, 2.7, 5.0])\n"
        "  delayed = proc.process(signal)\n")
        .def(py::init<GPUContext&>(), py::arg("ctx"),
             "Create LchFarrow processor bound to GPU context")

        .def("set_delays", &PyLchFarrow::set_delays,
             py::arg("delay_us"),
             "Set per-antenna delays in microseconds.")

        .def("set_sample_rate", &PyLchFarrow::set_sample_rate,
             py::arg("sample_rate"),
             "Set sample rate (Hz).")

        .def("set_noise", &PyLchFarrow::set_noise,
             py::arg("noise_amplitude"),
             py::arg("norm_val") = 0.7071067811865476f,
             py::arg("noise_seed") = 0,
             "Set noise parameters (added AFTER delay).")

        .def("load_matrix", &PyLchFarrow::load_matrix,
             py::arg("json_path"),
             "Load Lagrange 48x5 matrix from JSON.")

        .def("process", &PyLchFarrow::process,
             py::arg("input"),
             "Apply fractional delay on GPU.\n\n"
             "Args:\n"
             "  input: numpy complex64 (points,) or (antennas, points)\n\n"
             "Returns:\n"
             "  numpy.ndarray complex64: same shape, delayed signal")

        .def_property_readonly("sample_rate", &PyLchFarrow::sample_rate)
        .def_property_readonly("delays", &PyLchFarrow::get_delays)

        .def("__repr__", [](const PyLchFarrow& self) {
            return "<LchFarrow sample_rate=" +
                   std::to_string(static_cast<int>(self.sample_rate())) + ">";
        });
}
