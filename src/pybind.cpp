#include <pybind/pybind11/include/pybind11/pybind11.h>

#include "tenzing/operation.hpp"
#include "tenzing/graph.hpp"
#include "tenzing/sequence.hpp"
#include "tenzing/state.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tenzing, m) {

    py::class_<OpBase, std::shared_ptr<OpBase>>(m, "OpBase")
        .def("name", &OpBase::name)
        ;

    py::class_<NoOp, OpBase, std::shared_ptr<NoOp>>(m, "NoOp")
        .def(py::init<const std::string &>())
        ;

    py::class_<Sequence<BoundOp>>(m, "Sequence")
        .def(py::init())
        ;

    py::class_<Graph<OpBase>>(m, "Graph")
        .def(py::init())
        .def("vertex_size", &Graph<OpBase>::vertex_size)
        .def("start_then", &Graph<OpBase>::start_then)
        .def("then_finish", &Graph<OpBase>::then_finish)
        .def("then", &Graph<OpBase>::then)
        ;

    py::class_<SDP::State>(m, "State")
        .def(py::init<const Graph<OpBase>&>())
        .def("graph", &SDP::State::graph)
        .def("sequence", &SDP::State::sequence)
        ;
}

