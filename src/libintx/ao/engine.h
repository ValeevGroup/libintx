#ifndef LIBINTX_AO_ENGINE_H
#define LIBINTX_AO_ENGINE_H

#include "libintx/forward.h"
#include "libintx/shell.h"

#include <vector>
#include <array>
#include <tuple>
#include <cstddef>
#include <functional>
#include <memory>

namespace libintx {

  struct Overlap::Operator::Parameters {
  };

  struct Kinetic::Operator::Parameters {
  };

  struct Coulomb::Operator::Parameters {
  };

  struct Nuclear::Operator::Parameters {
    using Zr = std::tuple< int,std::array<double,3> >;
    std::vector<Zr> centers;
  };

}

namespace libintx::ao {

  template<>
  struct IntegralEngine<2> : IntegralEngine<> {

    virtual ~IntegralEngine() = default;
    virtual void set(const Nuclear::Operator::Parameters&) = 0;
    virtual void compute(Operator, const std::vector<Index2>&, double*) = 0;

    void overlap(const std::vector<Index2> &ij, double *V) {
      this->compute(Overlap,ij,V);
    };

    void kinetic(const std::vector<Index2> &ij, double *V) {
      this->compute(Kinetic,ij,V);
    }

    void nuclear(const std::vector<Index2> &ij, double *V) {
      this->compute(Nuclear,ij,V);
    }

  };

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

  template<int ... Args>
  std::unique_ptr< IntegralEngine<Args...> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  ) = delete;

  template<>
  std::unique_ptr< IntegralEngine<2> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

  template<>
  std::unique_ptr< IntegralEngine<3> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

  template<>
  std::unique_ptr< IntegralEngine<4> > integral_engine(
    const Basis<Gaussian>&,
    const Basis<Gaussian>&
  );

}

#endif /* LIBINTX_AO_ENGINE_H */
