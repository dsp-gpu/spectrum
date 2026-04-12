#pragma once
/**
 * @file py_helpers.hpp
 * @brief Общие вспомогательные шаблоны для Python биндингов dsp-gpu
 *
 * Включать ПЕРЕД любым py_*.hpp файлом.
 * Определяет vector_to_numpy и vector_to_numpy_2d (zero-copy через capsule).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <vector>
#include <complex>

namespace py = pybind11;

// Передаём numpy владение данными через capsule — Python GC вызовет delete.
// Zero-copy: данные не копируются второй раз.
template<typename T>
py::array_t<T> vector_to_numpy(std::vector<T>&& data) {
    auto* vec = new std::vector<T>(std::move(data));
    auto capsule = py::capsule(vec, [](void* ptr) {
        delete static_cast<std::vector<T>*>(ptr);
    });
    std::vector<py::ssize_t> shape   = { static_cast<py::ssize_t>(vec->size()) };
    std::vector<py::ssize_t> strides = { static_cast<py::ssize_t>(sizeof(T)) };
    return py::array_t<T>(shape, strides, vec->data(), capsule);
}

// 2D вариант (rows × cols), row-major (C-contiguous).
template<typename T>
py::array_t<T> vector_to_numpy_2d(std::vector<T>&& data, size_t rows, size_t cols) {
    auto* vec = new std::vector<T>(std::move(data));
    auto capsule = py::capsule(vec, [](void* ptr) {
        delete static_cast<std::vector<T>*>(ptr);
    });
    std::vector<py::ssize_t> shape = {
        static_cast<py::ssize_t>(rows),
        static_cast<py::ssize_t>(cols)
    };
    std::vector<py::ssize_t> strides = {
        static_cast<py::ssize_t>(cols * sizeof(T)),
        static_cast<py::ssize_t>(sizeof(T))
    };
    return py::array_t<T>(shape, strides, vec->data(), capsule);
}
