#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "libintx.h"
#include "libintx/hf/engine.h"
#include "libintx/hf/guess.h"
#include "libintx/utility.h"

namespace py = pybind11;

namespace libintx::python::hf {

  auto fock_engine(
    const std::vector< std::tuple<int, std::array<double,3> > > &atoms,
    const std::vector<PyGaussian> &basis,
    double precision)
  {
    auto wfn = Wfn<Gaussian>(atoms, make_basis(basis));
    return libintx::hf::fock_engine(wfn, precision);
  }

  // void init_ao_screening(libintx::hf::FockEngine &engine, double tol) {
  //   engine.init_ao_screening(ao::screening::norm2, tol);
  // }

  auto init2(libintx::hf::FockEngine &engine, double smin) {
    return engine.init2(smin);
  }

  void V(libintx::hf::FockEngine &engine, std::ptrdiff_t V, size_t ldV) {
    engine.V({reinterpret_cast<double*>(V), ldV});
  }

  void S(libintx::hf::FockEngine &engine, std::ptrdiff_t S, size_t ldS) {
    engine.S({reinterpret_cast<double*>(S), ldS});
  }

  void T(libintx::hf::FockEngine &engine, std::ptrdiff_t T, size_t ldT) {
    engine.T({reinterpret_cast<double*>(T), ldT});
  }

  auto X(libintx::hf::FockEngine &engine, std::ptrdiff_t X, size_t ldX) {
    return engine.X({reinterpret_cast<double*>(X), ldX});
  }

  void initial_guess(
    libintx::hf::FockEngine &engine,
    std::ptrdiff_t F, size_t ldF,
    double precision)
  {
    libintx::hf::SOAD guess;
    guess.precision = precision;
    guess.num_threads = { engine.num_threads };
    MatrixRef<double> f{ reinterpret_cast<double*>(F), ldF };
    guess.fock(engine.wfn(), f);
  }

  void D(
    libintx::hf::FockEngine &engine,
    std::ptrdiff_t F, size_t ldF,
    std::ptrdiff_t X, size_t ldX, size_t NX,
    std::ptrdiff_t D, size_t ldD,
    std::ptrdiff_t C, size_t ldC)
  {
    engine.D(
      MatrixRef{ reinterpret_cast<const double*>(F), ldF },
      MatrixRef{ reinterpret_cast<const double*>(X), ldX }, NX,
      MatrixRef{ reinterpret_cast<double*>(D), ldD },
      MatrixRef{ reinterpret_cast<double*>(C), ldC }
    );
  }

  void init(py::module m) {

    py::class_<libintx::hf::FockEngine>(m, "FockEngine")
      //.def("init_ao_screening", &hf::init_ao_screening)
      .def_readwrite("precision", &libintx::hf::FockEngine::precision)
      .def_property(
        "num_threads",
        nullptr,
        [](libintx::hf::FockEngine &obj, int num_threads) {
          obj.num_threads = { num_threads };
        }
      )
      .def("init2", &hf::init2, py::arg("smin")=1e-12)
      .def("S", &hf::S)
      .def("T", &hf::T)
      .def("V", &hf::V)
      .def("X", &hf::X)
      .def("initial_guess", &hf::initial_guess)
      .def("D", &hf::D)
      ;

    m.def(
      "fock_engine",
      &fock_engine,
      py::arg("basis"), py::arg("centers"), py::arg("precision") = 1e-16
    );

  }

}
