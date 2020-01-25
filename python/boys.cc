#include "boys/boys.h"
#include "boys/reference.h"
#include "boys/chebyshev.h"

#include <pybind11/pybind11.h>

namespace boys {

  struct PyBoys : public Boys {

    //using Boys::Boys;

    double compute(double t, int m) const override {
      PYBIND11_OVERLOAD_PURE(
        double,  // Return type */
        Boys,    // Parent class */
        compute, // Name of function in C++ (must match Python name) */
        t, m     // Argument(s) */
      );
    }

  };

}

PYBIND11_MODULE(boys, m) {

  using namespace boys;
  namespace py = pybind11;

  py::class_<Boys, PyBoys>(m, "Boys")
    //.def(py::init<>(&Boys::instance), py::arg("type") = "")
    .def("compute", &Boys::compute)
    ;

  m.def("reference", boys::reference);
  m.def("chebyshev", boys::chebyshev);

}
