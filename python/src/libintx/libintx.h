#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libintx/shell.h"

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

  inline auto make_gaussian(const PyGaussian &obj) {
    auto& [L,ps,r] = obj;
    std::vector<Gaussian::Primitive> prims;
    for (auto [a,C] : ps) {
      prims.push_back({a,C});
    }
    auto [r0,r1,r2] = r;
    return Gaussian(L,{r0,r1,r2},prims);
  }

  inline auto make_basis(const std::vector<PyGaussian> &pybasis) {
    auto basis = std::make_shared< libintx::Basis<libintx::Gaussian> >();
    for (auto &g : pybasis) {
      basis->push_back(make_gaussian(g));
    }
    return basis;
  }

}
