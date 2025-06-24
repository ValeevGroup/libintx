#ifndef LIBINTX_WFN_H
#define LIBINTX_WFN_H

#include "libintx/shell.h"
#include <array>
#include <tuple>
#include <vector>
#include <memory>

namespace libintx {

  template<typename Shell>
  struct Wfn {
    using Atom = std::tuple<int, std::array<double,3> >;
    template<typename BasisSet>
    Wfn(const std::vector<Atom> &atoms, const BasisSet &basis_set)
      : Wfn(atoms, make_basis(atoms, basis_set))
    {
    }
    Wfn(const std::vector<Atom> &atoms, const std::shared_ptr< Basis<Shell> > &basis)
      : atoms_(atoms), basis_(basis)
    {
    }
    const auto& atoms() const { return this->atoms_; }
    const auto& basis() const { return this->basis_; }
    int nocc2() const {
      int nocc = 0;
      for (auto [Z,r] : atoms_) {
        nocc += Z;
      }
      return nocc/2;
    }
  protected:
    std::vector<Atom> atoms_;
    std::shared_ptr< Basis<Shell> > basis_;
  };

}

#endif /* LIBINTX_SHELL_H */
