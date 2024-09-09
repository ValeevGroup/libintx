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

}

#endif /* LIBINTX_AO_MD_ENGINE_H */
