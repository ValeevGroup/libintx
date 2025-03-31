#ifndef LIBINTX_HF_GUESS_H
#define LIBINTX_HF_GUESS_H

#include "libintx/forward.h"
#include "libintx/ao/screening.h"
#include "libintx/wfn.h"

#include <functional>
#include <vector>
#include <memory>

namespace libintx::hf {

  struct SOAD {
    using Atom = std::tuple<int, std::array<double,3> >;
    static std::vector<double> density(const std::vector<Atom> &);
    void fock(
      const Wfn<Gaussian>& basis,
      std::function<void(Index2, const double*)>,
      const std::vector<Index1> &is,
      const std::vector<Index1> &js
    );
    void fock(const Wfn<Gaussian>& basis, MatrixRef<double> F);
    ao::screening::Norm<double> norm = ao::screening::norm<2>;
    double precision = 0;
    libintx::num_threads num_threads = { 1 };
  };

}

#endif /* LIBINTX_HF_SOAD_H */
