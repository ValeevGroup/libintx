#ifndef LIBINTX_HF_ENGINE_H
#define LIBINTX_HF_ENGINE_H

#include "libintx/forward.h"
#include "libintx/wfn.h"
#include "libintx/ao/screening.h"

#include <cstddef>
#include <memory>

namespace libintx::hf {

  extern int num_threads;

  struct FockEngine {

    explicit FockEngine(
      const Wfn<Gaussian>&,
      ao::screening::Norm<double> norm = ao::screening::norm<2>,
      double precision = 1e-16
    );

    ~FockEngine();

    std::tuple<size_t,size_t> init2(double smin = 1e-12);

    const auto& wfn() const { return wfn_; }
    const auto& ao_pairs() const { return ao_pairs_; }

    void S(MatrixRef<double> S);
    void T(MatrixRef<double> T);
    void V(MatrixRef<double> V);

    std::tuple<size_t,double,double> X(MatrixRef<double> X);

    void D(
      MatrixRef<const double> F,
      MatrixRef<const double> X, size_t NX,
      MatrixRef<double> D,
      MatrixRef<double> C = {}
    );

    double energy(
      MatrixRef<const double> F,
      MatrixRef<const double> H,
      MatrixRef<const double> D
    );

  protected:
    Wfn<Gaussian> wfn_;
    ao::screening::Norm<double> norm_;
    std::vector< std::tuple<Index2,float> > ao_pairs_;

  public:
    double precision = 0;
    libintx::num_threads num_threads = { 1 };

  };

  std::unique_ptr<FockEngine> fock_engine(const Wfn<Gaussian>&, double precision);

}

#endif /* LIBINTX_HF_ENGINE_H */
