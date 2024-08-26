#ifndef LIBINTX_ENGINE_H
#define LIBINTX_ENGINE_H

#include "libintx/forward.h"

#include <vector>
#include <array>
#include <cstddef>

namespace libintx {

  template<>
  struct IntegralEngine<1,2> : IntegralEngine<> {
    virtual ~IntegralEngine() = default;
    virtual void compute(
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      double*,
      std::array<size_t,2>
    ) = 0;
  };

  template<>
  struct IntegralEngine<2,2> : IntegralEngine<> {
    virtual ~IntegralEngine() = default;
    virtual void compute(
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      double*,
      std::array<size_t,2>
    ) = 0;
    size_t max_memory = 0;
  };

}

#endif /* LIBINTX_ENGINE_H */
