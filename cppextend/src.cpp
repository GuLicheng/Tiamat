#include "superpixel.hpp"


PYBIND11_MODULE(cpp, m) 
{
    m.def("label_expand", &pixel_fill<std::int32_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"))
     .def("label_expand", &pixel_fill<std::int64_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"))
     .def("label_expand", &pixel_fill<std::uint8_t>, py::arg("superpixel"), py::arg("label"), py::arg("superpixel_num"), py::arg("ignore_index"));
}