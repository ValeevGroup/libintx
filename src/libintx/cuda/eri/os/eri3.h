#ifndef LIBINTX_CUDA_ERI_OS_OS3_H
#define LIBINTX_CUDA_ERI_OS_OS3_H

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/pure.h"
#include "libintx/boys/cuda/chebyshev.h"
#include "libintx/cuda/api/kernel.h"
#include "libintx/cuda/api/thread_group.h"

#include "libintx/cuda/eri/os/vrr1.h"
#include "libintx/cuda/eri/os/vrr2.h"
#include "libintx/cuda/eri/os/hrr.h"

#include <utility>
#include <functional>
#include <memory>
#include <type_traits>
#include <cassert>
#include <iostream>

namespace libintx::cuda::os {

using Vector3 = array<double,3>;

template<int N, int _AB, int _X, class Boys, class Derived>
struct ERI;

template<int _AB, int _X, class Boys, class Derived>
struct ERI<3,_AB,_X,Boys,Derived> {

  static constexpr int MaxShmem = LIBINTX_CUDA_MAX_SHMEM;

  static constexpr int num_threads(int A, int B, int X) {
    int min_threads = std::max<int>(
      ncart(A+B),
      hrr::num_threads(A, B, X)
    );
    return 32*((min_threads+31)/32);
  }

  struct AB {
    static constexpr int KMAX = libintx::KMAX*libintx::KMAX;
    static constexpr int L = _AB;
    const int K;
    struct Primitive {
      double a, b, C;
    };
    Primitive prims[KMAX];
  } AB;

  struct X {
    static constexpr int KMAX = libintx::KMAX;
    static constexpr int L = _X;
    const bool pure;
    const int K;
    Gaussian::Primitive prims[KMAX];
    explicit X(const Gaussian &x)
      : pure(x.pure), K(x.K)
    {
      assert(0 < K && K <= KMAX);
      for (int k = 0; k < K; ++k) {
        this->prims[k] = x.prims[k];
      }
    }
  } X;

  static constexpr int L = (AB::L + X::L);
  static constexpr int N = (AB::L + 1);

  const Shell A, B;
  const Boys boys;

  struct Stack {
    static constexpr int ldV1 = vrr1::leading_dimension<AB::L>();
    int top = 0;
    int max = 0;
    int NV2, V2;
    int V1;
    int T;
    int alpha_over_p;
    int one_over_2p;
    int Xpa;
    int Xpq;
    int push(int n) {
      int s = top;
      top += n;
      max = std::max(max, top);
      return s;
    }
    Stack() = default;
    Stack(int A, int B, int X, int K) {
      this->NV2 = (ncartsum(A+B)-ncartsum(A-1))*ncart(X);
      if (X) this->NV2 += K*ncart(X);
      this->V2 = push(NV2);
      this->V1 = push(ldV1*K);
      this->T = push(K);
      this->alpha_over_p = push(K);
      this->one_over_2p = push(K);
      this->Xpa = push(3*K);
      this->Xpq = push(3*K);
      this->top = 0;
      int NH = std::max(
        ncart(A)*ncart(B)*npure(X),
        hrr::memory(A,B,X)
      );
      //printf("stack: K=%i, max(%i,%i)\n", K, this->max, NH);
      // hrr builds in V2 memory
      this->push(std::max(NV2, NH));
    }
  };

  Stack stack;
  int KB = 1;
  int min_threads = 32;
  int shmem = 0;

  ERI(const Gaussian &A, const Gaussian &B, const Gaussian &X, const Boys &boys)
    : A(A), B(B), AB{A.K*B.K}, X(X), boys(boys)
  {

    if (A.L < B.L) throw std::domain_error("ObaraSaika3::Kernel: A.L < B.L");
    if (AB.K > AB::KMAX) throw std::domain_error("ObaraSaika3::Kernel: AB.K > KMAX");
    if (X.L > 1 && !X.pure) {
      throw std::domain_error("ObaraSaika3::Kernel: X must be pure if L>1");
    }

    KB = AB.K;
    this->stack = Stack(A.L, B.L, X.L, KB);
    // attempt to fit at least 1 blocks
    int NB = 1;
    while (KB > 1 && stack.max*sizeof(double)+80 > MaxShmem) {
      NB++;
      KB = (AB.K+NB-1)/NB;
      stack = Stack(A.L, B.L, X.L, KB);
    }
    this->shmem = stack.max*sizeof(double);
    this->min_threads = num_threads(A.L, B.L, X.L);

    if (this->shmem > MaxShmem) {
      throw std::domain_error("ObaraSaika3::Kernel: kernel.shmem > MaxShmem");
    }

    for (int j = 0, ij = 0; j < B.K; ++j) {
      for (int i = 0; i < A.K; ++i, ++ij) {
        double a = A.prims[i].a;
        double b = B.prims[j].a;
        double C = A.prims[i].C*B.prims[j].C;
        C *= 2*pow(M_PI,2.5);
        AB.prims[ij] = { a, b, C };
      }
    }

  }


  static constexpr cuda::kernel::launch_bounds launch_bounds() {
    constexpr int A = _AB-_AB/2;
    constexpr int B = _AB/2;
    constexpr int X = _X;
    constexpr int shmem = ncart(A)*ncart(B)*npure(X)*sizeof(double);
    constexpr int max_threads = ERI::num_threads(A, B, X);
    // simple heruestic for good launch bounds
    constexpr int min_blocks = std::min(MaxShmem/shmem, 768/max_threads);
    return { max_threads, min_blocks/2 };
    //return { max_threads, min_blocks };
  }

  template<class ... Args>
  void launch(dim3 grid, cudaStream_t stream, Args ... args) const {

    Derived eri = static_cast<const Derived&>(*this);

    dim3 block = { 32, uint(eri.min_threads+31)/32, 1 };
    int shmem = eri.shmem;

    void (*kernel_launch)(Derived,Args...) = nullptr;

    static const int max_threads = eri.launch_bounds().max_threads;
    static const int min_blocks = eri.launch_bounds().min_blocks;
    kernel_launch = &cuda::kernel::launch<max_threads, 1, Derived, Args...>;

    int max_blocks = MaxShmem/(eri.shmem+128);

    if (max_blocks < min_blocks - min_blocks/4) {
      kernel_launch = &cuda::kernel::launch<max_threads, min_blocks - min_blocks/2, Derived, Args...>;
    }

    if (max_blocks > min_blocks + min_blocks/4) {
      kernel_launch = &cuda::kernel::launch<max_threads, min_blocks + min_blocks/2, Derived, Args...>;
    }

    // printf(
    //   "# kernel<%i,%i> block=(%i,%i), shmem=%i, Kb=%i, max_threads=%i\n",
    //   AB.L, X.L, block.x, block.y, shmem, KB,
    //   max_threads
    // );

    //cuda::kernel::set_prefered_shared_memory_carveout(launch, 100);
    cuda::kernel::set_max_dynamic_shared_memory_size(kernel_launch, shmem);

    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    //printf("transform=%p, kernel.transform=%p\n", transform, kernel.transform);
    kernel_launch<<<grid,block,shmem,stream>>>(eri, args...);
    cuda::error::ensure_none("Obara Saika ERI-3 failed");

    assert(cudaStreamSynchronize(stream) == cudaSuccess);


  };


  template<class Primitive2, class Primitive1>
  __device__  __forceinline__ // __noinline__
  static void update_vrr2(
    const Shell &A, const Shell &B, const Boys &boys,
    const Vector3 &rA, const Vector3 &rB, const Vector3 &rX,
    int K, const Primitive2 *ab, const Primitive1 &x,
    const Stack &stack, double *shmem)
  {

    using namespace libintx::cuda::os;

    static constexpr int ldV1 = vrr1::leading_dimension<AB::L>();

    double *V1 = shmem + stack.V1;
    double *T = shmem + stack.T;
    double *alpha_over_p = shmem + stack.alpha_over_p;
    double *one_over_2p = shmem + stack.one_over_2p;
    double *Xpa = shmem + stack.Xpa;
    double *Xpq = shmem + stack.Xpq;

    const auto &thread_rank = this_thread_block().thread_rank();
    const auto &num_threads = this_thread_block().size();

    for (int k = thread_rank; k < K; k += num_threads) {
      double a = ab[k].a;
      double b = ab[k].b;
      double q = x.a;
      Vector3 P = center_of_charge(a, rA, b, rB);
      const auto &Q = rX;
      double p = (a+b);
      double alpha = (p*q)/(p+q);
      one_over_2p[k] = 0.5/p;
      alpha_over_p[k] = alpha/p;
      for (int i = 0; i < 3; ++i) {
        Xpa[i+3*k] = P[i]-rA[i];
        Xpq[i+3*k] = alpha_over_p[k]*(P[i]-rX[i]);
      }
      // for (int m = 0; m < N; ++m) {
      //   V1[m + k*ldV1] = C;
      // }
      T[k] = alpha*norm(P,Q);
    }

    __syncthreads();

    for (int idx = thread_rank; idx < K*N; idx += num_threads) {
      int m = idx%N;
      int k = idx/N;
      double s = boys.compute(T[k], (X::L + m));
      V1[m + k*ldV1] = s;
    }

    __syncthreads();

    // loops broken apart less registers
    for (int k = thread_rank; k < K; k += num_threads) {
      //const auto &Q = rX;
      double a = ab[k].a;
      double b = ab[k].b;
      double p = (a+b);
      double q = x.a;
      double Kab = exp(-(a*b)/p*norm(rA,rB));
      double Kcd = 1;//exp(-norm(Q));
      double C = ab[k].C*x.C;
      double pq = p*q;
      C *= Kab*Kcd;
      C *= rsqrt(pq*pq*(p+q));
      for (int m = 0; m < N; ++m) {
        V1[m + k*ldV1] *= C;
      }
    }

    __syncthreads();

    vrr1::vrr1<AB::L>(K, Xpa, one_over_2p, Xpq, alpha_over_p, V1, ldV1);

    double *alpha_over_2pc = alpha_over_p;

    __syncthreads();

    for (int k = thread_rank; k < K; k += num_threads) {
      double a = ab[k].a;
      double b = ab[k].b;
      double q = x.a;
      double p = (a+b);
      double p_over_q = p/q;
      for (int i = 0; i < 3; ++i) {
        // (alpha/q)*Xpq = (p/q)*(alpha/p)*Xpq
        Xpq[i + 3*k] = p_over_q*(Xpq[i+3*k]);
      }
      alpha_over_2pc[k] = alpha_over_p[k]/(2*q);
    }

    __syncthreads();
    __threadfence();

    double *V2 = shmem + stack.V2;
    vrr2::vrr2<AB::L, X::L>(A.L, B.L, K, Xpq, alpha_over_2pc, V1, ldV1, V2);

  }

  __device__  __forceinline__
  double* compute(Vector3 &rA, Vector3 &rB, Vector3 &rX) const {

    using namespace libintx::cuda::os;

    const auto &thread_rank = this_thread_block().thread_rank();
    const auto &num_threads = this_thread_block().size();

    extern __shared__ double shmem[];

    for (int i = thread_rank; i < stack.NV2; i += num_threads) {
      shmem[stack.V2+i] = 0;
    }

    for (int kx = 0; kx < X.K; ++kx) {
      const auto &x = X.prims[kx];
      for (int k = 0; k < AB.K; k += KB) {
        int K = min(AB.K-k,KB);
        update_vrr2(A, B, boys, rA, rB, rX,  K, AB.prims+k, x, stack, shmem);
      }
    }

    static constexpr int NX = npure(X::L);
    double *V2 = shmem + stack.V2;

    if (X.pure || X::L > 1) {
      int NAB = ncartsum(AB::L)-ncartsum(A.L-1);
      std::integral_constant<int,X::L> L;
      cartesian_to_pure(L, NAB, V2, this_thread_block());
    }

    if (B.L) {
      if (thread_rank < 3) {
        rA[thread_rank] = rA[thread_rank] - rB[thread_rank];
      }
      hrr::hrr<AB::L, NX>(A.L, B.L, rA.data, V2);
    }

    {
      __syncthreads();
      constexpr int NX = npure(X::L);
      cartesian_to_pure<NX>(
        std::pair<bool,bool>{A.pure,B.pure},
        A.L, B.L, V2,
        this_thread_block()
      );
    }

    return V2;

  }

};

}

#endif /* LIBINTX_CUDA_ERI_OS_OS3_H */
