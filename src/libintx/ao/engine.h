#ifndef LIBINTX_AO_ENGINE_H
#define LIBINTX_AO_ENGINE_H

#include "libintx/forward.h"

#include <vector>
#include <array>
#include <cstddef>

namespace libintx::ao {

  template<>
  struct IntegralEngine<3> : IntegralEngine<> {

    virtual ~IntegralEngine() = default;

    virtual void compute(
      Operator,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double*,
      const std::array<size_t,2> &dims
    ) = 0;

    void coulomb(
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double* V,
      const std::array<size_t,2> &dims)
    {
      this->compute(libintx::Coulomb,bra,ket,norms,V,dims);
    }

  };

  template<>
  struct IntegralEngine<4> : IntegralEngine<> {
    virtual ~IntegralEngine() = default;
    virtual void compute(
      Operator op,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double *V,
      const std::array<size_t,2> &dims
    ) = 0;
    size_t max_memory = 0;
  };

}

#endif /* LIBINTX_AO_ENGINE_H */
