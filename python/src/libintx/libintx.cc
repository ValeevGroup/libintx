#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "libintx/ao/engine.h"
#include "libintx/gpu/md/engine.h"

namespace py = pybind11;

namespace pybind11::detail {

  template <typename Type, size_t Size>
  struct type_caster<libintx::array<Type, Size>>
    : array_caster<libintx::array<Type, Size>, Type, false, Size> {};

  template <>
  struct type_caster<libintx::Index2>
    : tuple_caster<libintx::Index2, int, int> {};

}

namespace libintx::python {

  using double2 = std::tuple<double,double>;
  using double3 = std::tuple<double,double,double>;
  using PyGaussian = std::tuple<int,std::vector<double2>,double3>;

  auto make_gaussian(const PyGaussian &obj) {
    auto& [L,ps,r] = obj;
    std::vector<Gaussian::Primitive> prims;
    for (auto [a,C] : ps) {
      prims.push_back({a,C});
    }
    auto [r0,r1,r2] = r;
    return Gaussian(L,{r0,r1,r2},prims);
  }

  auto basis_cast(const std::vector<PyGaussian> &pybasis) {
    Basis<libintx::Gaussian> basis;
    for (auto &g : pybasis) {
      basis.push_back(make_gaussian(g));
    }
    return basis;
  }

  namespace ao {

    using libintx::ao::IntegralEngine;

    auto engine(
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

    template<int Centers, typename Bra, typename Ket>
    void compute(
      IntegralEngine<> &engine,
      const std::vector<Bra> &bra,
      const std::vector<Ket> &ket,
      std::ptrdiff_t dst,
      const std::array<size_t,2> &dims)
    {
      dynamic_cast<IntegralEngine<Centers>&>(engine)
        .compute(Coulomb, bra, ket, reinterpret_cast<double*>(dst), dims);
    };


  }

}

PYBIND11_MODULE(libintx, m) {

  // py::class_<libintx::Gaussian>(m, "Gaussian")
  //   .def(py::init(&libintx::python::make_gaussian));

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
    .def(
      "compute",
      &libintx::python::ao::compute<3, libintx::Index1, libintx::Index2>,
      py::arg("bra"),
      py::arg("ket"),
      py::arg("dst"),
      py::arg("dims"),
      "Compute 3-center integrals\n"
      "bra,ket - list of bra,ket shell indices\n"
      "dst,dims - destination array"
    )
    ;

#ifdef LIBINTX_GPU
  auto gpu = m.def_submodule("gpu");
  gpu.def(
    "aoeri",
    &libintx::python::ao::engine,
    py::arg("centers"),
    py::arg("bra"),
    py::arg("ket"),
    py::arg("stream")=0
  );
#endif

}
