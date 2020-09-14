#include "libintx/cuda/eri.h"
#include "libintx/cuda/eri/os/eri.h"
#include "libintx/config.h"

namespace libintx::cuda {

  using ERI3 = ERI<3,ObaraSaika>;

  void ERI<3>::set_centers(const std::vector< Double<3> > &centers) {
    this->centers_.assign(centers.data(), centers.size());
  }

  void ERI<3>::compute(
    const Gaussian &A, const Gaussian &B,
    const Gaussian &X,
    const IntegralList<3> &list)
  {
    auto impl = ERI<3,ObaraSaika>(A, B, X);
    impl.compute(list.size(), list.data(), this->centers_.data());
    ::cuda::outstanding_error::ensure_none("ERI<3>::compute failed");
    assert(cuda::device::synchronize());
  }

  template<>
  std::unique_ptr< ERI<3> > eri() {
    return std::make_unique< ERI<3> >();
  }

}
