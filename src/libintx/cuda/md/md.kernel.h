#ifndef LIBINTX_CUDA_MD_MD_KERNEL_H
#define LIBINTX_CUDA_MD_MD_KERNEL_H

#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/engine/md/r1.h"

#include "libintx/cuda/api/thread_group.h"
#include "libintx/pure.transform.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  template<int X>
  struct Basis1 {
    static constexpr int Centers = 1;
    static constexpr int L = X;
    static constexpr int nherm = nherm1(L);
    static constexpr int nbf = npure(L);
    const int K, N;
    const Hermite *data;
    LIBINTX_GPU_ENABLED
    const Hermite* hdata(int idx, int k) const {
      return data + idx + k*N;
    }
  };

  template<int ... Args>
  struct Basis2;

  template<int AB>
  struct Basis2<AB> {
    static constexpr int Centers = 2;
    static constexpr int L = AB;
    static constexpr int nherm = nherm2(L);
    const Shell first, second;
    const int nbf;
    const int K;
    const int N;
    const double *data;
    const int stride;
    Basis2(Shell a, Shell b, int K, int N, const double *H)
      : first(a), second(b),
        nbf(libintx::nbf(a)*libintx::nbf(b)),
        K(K), N(N), data(H),
        stride(sizeof(Hermite)/sizeof(double)+nherm*nbf)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int p, int k) const {
      return reinterpret_cast<const Hermite*>(data + k*stride + p*K*stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int p, int k) const {
      return reinterpret_cast<const double*>(hdata(p,k)+1);
    }
  };

  template<int _A, int _B>
  struct Basis2<_A,_B> {
    static constexpr int Centers = 2;
    static constexpr int First = _A;
    static constexpr int Second = _B;
    static constexpr int L = _A + _B;
    static constexpr int nherm = nherm2(L);
    static constexpr int nbf = npure(_A)*npure(_B);
    static constexpr int stride = sizeof(Hermite)/sizeof(double) + nherm*nbf;
    const int K;
    const int N;
    const double *data;
    const double *pure_transform;
    Basis2(int K, int N, const double *H, const double *pure_transform)
      : K(K), N(N), data(H), pure_transform(pure_transform)
    {
    }
    explicit Basis2(const Basis2<L> &basis)
      : K(basis.K), N(basis.N), data(basis.data)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int p, int k = 0) const {
      return reinterpret_cast<const Hermite*>(data + k*stride + p*K*stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int p, int k = 0) const {
      return reinterpret_cast<const double*>(hdata(p,k)+1);
    }
  };


  LIBINTX_GPU_DEVICE
  constexpr auto orbitals2 = hermite::orbitals<2*LMAX>;

  LIBINTX_GPU_DEVICE
  constexpr auto orbitals1 = std::tuple{
    hermite::orbitals1<XMAX,0>,
    hermite::orbitals1<XMAX,1>
  };

  template<int ... Args>
  constexpr auto& orbitals(const Basis2<Args...>&) {
    static_assert(orbitals2.size());
    return orbitals2;
  }

  template<int X>
  constexpr auto& orbitals(const Basis1<X>&) {
    return std::get<X%2>(orbitals1);
  }

}

#endif /* LIBINTX_CUDA_MD_MD_KERNEL_H */
