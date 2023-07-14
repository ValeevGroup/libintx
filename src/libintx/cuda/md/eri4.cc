#include "libintx/cuda/md/eri4.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include "libintx/tuple.h"
#include <numeric>

namespace libintx::cuda::md {

  ERI4::ERI4(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, cudaStream_t stream) {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
  }

  void ERI4::compute(
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    double *V)
  {

    using Kernel = std::function<void(
      ERI4 &eri, const Basis2&, const Basis2&, double*, size_t, cudaStream_t
    )>;

    static auto ab_cd_kernels = make_array<Kernel,2*LMAX+1,2*LMAX+1>(
      [](auto ab, auto cd) {
        return &ERI4::compute<ab,cd>;
      }
    );

    auto stream = this->stream_;
    Basis2 p = make_basis(bra_, bra_, bra, this->p_, stream);
    Basis2 q = make_basis(ket_, ket_, ket, this->q_, stream);
    auto kernel = ab_cd_kernels[p.first.L+p.second.L][q.first.L+q.second.L];
    kernel(*this, p, q, V, nbf(ket_), stream);

  }

  template<>
  std::unique_ptr< IntegralEngine<4,2> > eri<4>(
    const Basis<Gaussian> &bra,
    const Basis<Gaussian> &ket,
    const cudaStream_t &stream)
  {
    return std::make_unique<ERI4>(bra, ket, stream);
  }

}
