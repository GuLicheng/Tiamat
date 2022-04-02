#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <cstdint>

namespace py = pybind11;

namespace detail
{

template <typename ContiguousIterator>
std::vector<int> count_region(ContiguousIterator superpixel, ContiguousIterator label, int len, int pixelnum, int ignore_index)
{
    std::vector<std::unordered_map<int, int>> table;
    table.resize(pixelnum);

    for (int i = 0; i < len; ++i)
    {
        auto superpixel_id = superpixel[i];
        auto label_id = label[i];
        if (label_id != ignore_index)
            table[superpixel_id][label_id]++;
    }

    std::vector<int> result(pixelnum, ignore_index);
    for (std::size_t i = 0; i < result.size(); ++i)
    {
        auto iter = std::max_element(table[i].begin(), table[i].end(), [](const auto& x, const auto& y) {
                return x.second < y.second;
            });
        if (iter != table[i].end())
            result[i] = iter->first;
    }
    return result;
}

template <typename ContiguousIterator>
void superpixel_fill(ContiguousIterator superpixel, ContiguousIterator dest, int len, const std::vector<int>& region)
{
    for (int i = 0; i < len; ++i)
    {
        auto superpixel_id = superpixel[i];
        auto class_id = region[superpixel_id];
        dest[i] = class_id;
    }
}

template <typename ContiguousIterator>
void super_pixel_expand(ContiguousIterator superpixel, ContiguousIterator label, ContiguousIterator dest, int length, int superpixel_num, int ignore_index)
{
    auto region = count_region(superpixel, label, length, superpixel_num, ignore_index);
    superpixel_fill(superpixel, dest, length, region);
}

}

/*
 * input1: superpixel
 * input2: label
 * pixelnum: num of superpixel, equal to superpixel.max() + 1
 * ignore_index: default class, for scribble, it will be 255 and for saliency it may be 0
 */
template <typename T>
py::array_t<T> pixel_fill(py::array_t<T> input1, py::array_t<T> input2, int superpixel_num, int ignore_index)
{
    py::buffer_info buf1 = input1.request(), buf2 = input2.request();
    
    if (buf1.shape != buf2.shape || buf1.strides != buf2.strides)
        throw std::runtime_error("Input shapes must match");

    /* No pointer is passed, so NumPy will allocate the buffer */
    auto result = py::array_t<T>(buf1.shape, buf1.strides);
    py::buffer_info buf3 = result.request();

    T *ptr1 = static_cast<T*>(buf1.ptr);
    T *ptr2 = static_cast<T*>(buf2.ptr);
    T *ptr3 = static_cast<T*>(buf3.ptr);

    detail::super_pixel_expand(ptr1, ptr2, ptr3, buf1.size, superpixel_num, ignore_index);
    return result;
}

PYBIND11_MODULE(cpp, m) 
{
    m.def("label_expand", &pixel_fill<std::int32_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"))
     .def("label_expand", &pixel_fill<std::int64_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"))
     .def("label_expand", &pixel_fill<std::uint8_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"));
}