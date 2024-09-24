#ifndef LIBINTX_AO_MD_ENGINE_H
#define LIBINTX_AO_MD_ENGINE_H

#include "libintx/ao/engine.h"
#include "libintx/shell.h"

#include <memory>

namespace libintx::md {

  template<int>
  struct IntegralEngine;

  template<>
  struct IntegralEngine<2> : libintx::ao::IntegralEngine<2> {

    // integral batch size
    int Batch = 0;

    using Visitor = std::function<void(size_t batch, size_t idx, const double*, size_t)>;

    explicit IntegralEngine(const Basis<Gaussian>&, const Basis<Gaussian>&);

    ~IntegralEngine();

    template<typename Operator>
    void set(const typename Operator::Parameters &p) {
      std::get<Operator::Parameters>(params_) = p;
    }

    void set(const Nuclear::Operator::Parameters &p) override {
      std::get<Nuclear::Operator::Parameters>(params_) = p;
    }

    void compute(Operator, const std::vector<Index2>&, double*) override;

    void compute(Operator, const std::vector<Index2>&, const Visitor&);

  public:
    int num_threads = 1;

  private:
    template<typename T, Operator, typename Params>
    void compute(const Params&, const std::vector<Index2>&, const Visitor&);

  private:
    Basis<Gaussian> bra_, ket_;

    std::tuple<
      Nuclear::Operator::Parameters
      > params_;

  };

  template<>
  struct IntegralEngine<3> : libintx::ao::IntegralEngine<3> {

    using Visitor = std::function<
      void(size_t, BraKet<size_t>, const double*, size_t)
      >;

    explicit IntegralEngine(
      const std::shared_ptr< Basis<Gaussian> > &bra,
      const std::shared_ptr< Basis<Gaussian> > &ket
    )
      : IntegralEngine({bra,ket,ket}) {}

    explicit IntegralEngine(const std::shared_ptr< Basis<Gaussian> > (&basis)[3]);

    ~IntegralEngine();

    void compute(
      Operator op,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      const Visitor &V
    );

    void compute(
      Operator op,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double *V,
      const std::array<size_t,2> &dims
    ) override;

    const auto& basis(size_t idx) const {
      return *basis_[idx];
    }

  private:

    template<Operator Op, typename Params>
    void compute(
      const Params &params,
      const std::vector<Index1> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      const Visitor &V
    );

  public:
    int num_threads = 1;

  private:
    std::shared_ptr< Basis<Gaussian> > basis_[3] = {};

  };

  template<>
  struct IntegralEngine<4> : libintx::ao::IntegralEngine<4> {

    using Visitor = std::function<
      void(BraKet<size_t>, BraKet<size_t>, const double*, size_t)
      >;

    explicit IntegralEngine(const std::shared_ptr< Basis<Gaussian> > &basis)
      : IntegralEngine({basis,basis,basis,basis}) {}

    explicit IntegralEngine(
      const std::shared_ptr< Basis<Gaussian> > &bra,
      const std::shared_ptr< Basis<Gaussian> > &ket
    )
      : IntegralEngine({bra,bra,ket,ket}) {}

    explicit IntegralEngine(const std::shared_ptr< Basis<Gaussian> > (&basis)[4]);

    ~IntegralEngine();

    void compute(
      Operator,
      const std::vector<Index2>&,
      const std::vector<Index2>&,
      BraKet<const double*> norms,
      const Visitor&
    );

    void compute(
      Operator op,
      const std::vector<Index2> &bra,
      const std::vector<Index2> &ket,
      BraKet<const double*> norms,
      double *V,
      const std::array<size_t,2> &dims
    ) override;

    const auto& basis(size_t idx) const {
      return *basis_[idx];
    }

  private:

    template<Operator, typename Params>
    void compute(
      const Params&,
      const std::vector<Index2>&,
      const std::vector<Index2>&,
      BraKet<const double*> norms,
      const Visitor&
    );

  public:
    int num_threads = 1;

  private:
    std::shared_ptr< Basis<Gaussian> > basis_[4] = {};

  };

}

#endif /* LIBINTX_AO_MD_ENGINE_H */
