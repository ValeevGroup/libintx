#ifndef LIBINTX_SCREENING_H
#define LIBINTX_SCREENING_H

#include "libintx/forward.h"
#include <functional>
#include <tuple>
#include <cmath>

namespace libintx::ao::screening {

  template<typename T>
  using Norm = std::function<T(size_t, const T*, size_t)>;

  template<int N, typename T = double>
  T norm(size_t n, const T* v, size_t inc);

  template<typename T>
  T absmax(size_t, const T*, size_t);

  double compute(Norm<double>, const Gaussian&, const Gaussian&);

  template<typename T>
  std::vector< std::tuple<Index2,T> > pairs(
    const Basis<Gaussian>&,
    Norm<double>, T = 1e-16,
    libintx::num_threads = { 1 }
  );

}

#endif /* LIBINTX_SCREENING_H */
