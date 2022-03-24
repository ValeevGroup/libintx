#ifndef RYSQ_RYSQ_H
#define RYSQ_RYSQ_H

#include "rysq/config.h"
#include "rysq/constants.h"
#include "rysq/shell.h"
#include "rysq/vector.h"

#include <memory>
#include <string>

namespace rysq {

  using ::rysq::shell::Shell;
  using ::rysq::shell::Bra;
  using ::rysq::shell::Ket;

  struct Parameters {
    double cutoff = 0;
    size_t K = 0;
  };

  // template<int Order, class ... BraKet>
  // struct Kernel;

  template<class Bra, class Ket>
  struct Kernel {

    const Bra bra;
    const Ket ket;

    Kernel(const Bra &bra, const Ket &ket, Parameters = Parameters{})
      : bra(bra), ket(ket)
    {
    }

    virtual ~Kernel() {}

    template<class ... Rs>
    const double* compute(const Rs& ... rs) {
      static_assert(
        sizeof...(Rs) == (Bra::size + Ket::size),
        "wrong number of shell centers"
      );
      return this->compute(shell::centers<Bra,Ket>{rs...});
    }

    virtual const double* compute(const shell::centers<Bra,Ket>&) = 0;

  };

  // template<>
  // struct Kernel<2> : Kernel<2, Bra<1>, Ket<1> > {};

  // template<>
  // struct Kernel<3> : Kernel<3, Bra<2>, Ket<1> > {};

  // template<>
  // struct Kernel<4> : Kernel<4, Bra<2>, Ket<2> > {};


  template<class Bra, class Ket>
  size_t flops(const Kernel<Bra,Ket> &kernel) {
    int N = (L(kernel.bra) + L(kernel.ket))/2 + 1;
    return N*3*nbf(kernel.bra)*nbf(kernel.ket);
  }

  template<class Bra, class Ket>
  std::string str(const Kernel<Bra,Ket> &kernel) {
    return "[" + shell::str(kernel.bra) + "|" + shell::str(kernel.ket) + "]";
  }

  typedef Kernel< Bra<1>, Ket<1> > Kernel2;
  typedef Kernel< Bra<2>, Ket<1> > Kernel3;
  typedef Kernel< Bra<2>, Ket<2> > Kernel4;

  std::unique_ptr<Kernel2> kernel(const Bra<1> &bra, const Ket<1> &ket, const Parameters& = {});

  std::unique_ptr<Kernel3> kernel(const Bra<2> &bra, const Ket<1> &ket, const Parameters& = {});

  std::unique_ptr<Kernel4> kernel(const Bra<2> &bra, const Ket<2> &ket, const Parameters& = {});

}

#endif /* RYSQ_RYSQ_H */
