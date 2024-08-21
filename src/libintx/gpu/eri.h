#ifndef LIBINTX_CUDA_ERI_H
#define LIBINTX_CUDA_ERI_H

#include "libintx/array.h"
#include "libintx/tuple.h"
#include "libintx/shell.h"
#include "libintx/engine.h"
#include "libintx/cuda/api/api.h"

#include <vector>
#include <utility>
#include <memory>

namespace libintx::gpu {

  template<int N>
  struct alignas(8) IntegralTuple {
    array<int32_t,N> index;
    double* data;
    double scale;
  };

  template<int N>
  using IntegralList = cuda::host::vector< IntegralTuple<N> >;


  template<int Order, class ... Args>
  struct ERI;

  template<>
  struct ERI<1> {
    virtual ~ERI() = default;
  };

  template<>
  struct ERI<2> {
    virtual ~ERI() = default;
  };

  template<>
  struct ERI<3> {
    virtual ~ERI() = default;
    virtual void set_centers(const std::vector< Double<3> > &centers);
    virtual void compute(
      const Gaussian &P, const Gaussian &Q,
      const Gaussian &R,
      const IntegralList<3> &list
    );
  protected:
    cuda::device::vector< Double<3> > centers_;
  };

  template<>
  struct ERI<4> {
    virtual ~ERI() = default;
  };

  template<int Order>
  std::unique_ptr< ERI<Order> > eri() = delete;

  template<>
  std::unique_ptr< ERI<1> > eri();

  template<>
  std::unique_ptr< ERI<2> > eri();

  template<>
  std::unique_ptr< ERI<3> > eri();

  template<>
  std::unique_ptr< ERI<4> > eri();

}

#endif /* LIBINTX_CUDA_ERI_H */
