#include "libintx.h"
#include "libintx/ao/engine.h"

namespace py = pybind11;

namespace libintx::python::ao {

  using libintx::ao::IntegralEngine;

  template<int Centers, typename Bra, typename Ket>
  void compute(
    IntegralEngine<> &engine,
    const std::vector<Bra> &bra,
    const std::vector<Ket> &ket,
    std::ptrdiff_t dst,
    const std::array<size_t,2> &dims)
  {
    dynamic_cast<IntegralEngine<Centers>&>(engine)
      .compute(Coulomb, bra, ket, {}, reinterpret_cast<double*>(dst), dims);
  };


}

#ifdef LIBINTX_GPU
#include "libintx/gpu/md/engine.h"

namespace libintx::python::gpu {

  auto integral_engine(
    int centers,
    const std::vector<PyGaussian> &bra,
    const std::vector<PyGaussian> &ket,
    std::ptrdiff_t stream)
  {
    using libintx::gpu::integral_engine;
    std::unique_ptr< IntegralEngine<> > eri;
    if (centers == 3) {
      eri = integral_engine<3>(basis_cast(bra), basis_cast(ket), gpuStream_t(stream));
    }
    if (centers == 4) {
      eri = integral_engine<4>(basis_cast(bra), basis_cast(ket), gpuStream_t(stream));
    }
    return py::cast(std::move(eri));
  }

}

#endif // LIBINTX_GPU

namespace libintx::python::hf {
  void init(py::module);
}

PYBIND11_MODULE(libintx, m) {

  py::class_< libintx::ao::IntegralEngine<> >(m, "IntegralEngine")
    .def(
      "compute",
      &libintx::python::ao::compute<4, libintx::Index2, libintx::Index2>,
      py::arg("bra"),
      py::arg("ket"),
      py::arg("dst"),
      py::arg("dims"),
      "Compute 4-center integrals\n"
      "bra,ket - list of bra,ket shell indices\n"
      "dst,dims - destination array"
    )
    // .def(
    //   "compute",
    //   &libintx::python::ao::compute<3, libintx::Index1, libintx::Index2>,
    //   py::arg("bra"),
    //   py::arg("ket"),
    //   py::arg("dst"),
    //   py::arg("dims"),
    //   "Compute 3-center integrals\n"
    //   "bra,ket - list of bra,ket shell indices\n"
    //   "dst,dims - destination array"
    // )
    ;

  libintx::python::hf::init(m.def_submodule("hf"));

#ifdef LIBINTX_GPU
  auto gpu = m.def_submodule("gpu");
  gpu.def(
    "integral_engine",
    &libintx::python::gpu::integral_engine,
    py::arg("centers"),
    py::arg("bra"),
    py::arg("ket"),
    py::arg("stream")=0
  );
#endif

}
