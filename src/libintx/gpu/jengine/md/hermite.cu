// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/jengine/md/forward.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/orbital.h"
#include "libintx/pure.transform.h"
#include <iostream>

namespace libintx::gpu::jengine::md {
namespace {

  LIBINTX_GPU_CONSTANT
  constexpr auto pure_transform = pure::make_transform(std::make_index_sequence<max(XMAX,LMAX)+1>());

  __device__
  void cartesian_to_pure(int A, double *G, auto &&thread_group) {
    assert(npure(A) <= thread_group.size());
    auto cartesian_to_pure = [&](auto &&transform) {
      double a = 0;
      int ia = thread_group.thread_rank();
      if (ia < npure(A)) {
        a = transform.cartesian_to_pure(ia, G);
      }
      thread_group.sync();
      if (ia < npure(A)) G[ia] = a;
    };
    pure_transform.apply(cartesian_to_pure, A);
    thread_group.sync();
  }

  __device__
  void pure_to_cartesian(int A, double *G, auto &&thread_group) {
    assert(ncart(A) <= thread_group.size());
    auto pure_to_cartesian = [&](auto &&transform) {
      double a = 0;
      int ia = thread_group.thread_rank();
      if (ia < ncart(A)) {
        a = transform.pure_to_cartesian(ia, G);
      }
      thread_group.sync();
      if (ia < ncart(A)) G[ia] = a;
    };
    pure_transform.apply(pure_to_cartesian, A);
    thread_group.sync();
  }

  __device__
  void cartesian_to_pure(const Shell &A, const Shell &B, double *T, auto &&g){
    auto rank = g.thread_rank();
    int na = nbf(A);
    int nb = nbf(B);
    // T[A,B] -> T[A',B]
    if (A.pure) {
      g.sync();
      assert(ncart(B) <= g.size());
      auto cartesian_to_pure = [&](const auto &transform) {
        auto A = transform.L;
        na = npure(A);
        double p[npure(A)] = {};
        auto *U = T+rank;
        if (rank < nb) {
          transform.cartesian_to_pure(
            [&](auto &&i) { return U[index(i)*nb]; },
            [&](auto &&i, auto v) { p[index(i)] = v; }
          );
        }
        g.sync();
        if (rank < nb) {
          for (size_t i = 0; i < std::size(p); ++i) {
            T[rank+i*nb] = p[i];
          }
        }
      };
      pure_transform.apply(cartesian_to_pure, A.L);
    }
    // T[A',B] -> T[A',B']
    if (B.pure) {
      g.sync();
      auto cartesian_to_pure = [&](auto &&transform) {
        auto B = transform.L;
        nb = npure(B);
        auto *U = T+rank*ncart(B);
        double p[npure(B)] = {};
        if (rank < na) {
          transform.cartesian_to_pure(
            [&](auto &&i) { return U[index(i)]; },
            [&](auto &&i, auto v) { p[index(i)] = v; }
          );
        }
        g.sync();
        if (rank < na) {
          for (size_t i = 0; i < std::size(p); ++i) {
            T[i+rank*npure(B)] = p[i];
          }
        }
      };
      pure_transform.apply(cartesian_to_pure, B.L);
    }
    g.sync();
  }

  __device__
  void pure_to_cartesian(const Shell &A, const Shell &B, double *T, auto &&g) {
    auto rank = g.thread_rank();
    int na = nbf(A);
    int nb = nbf(B);
    if (!A.pure) {
      g.sync();
      auto pure_to_cartesian = [&](auto &&transform) {
        auto A = transform.L;
        na = ncart(A);
        double p[npure(A)];
#define T(i) T[rank + i*nb]
        if (rank < nb) {
#pragma unroll
          for (size_t i = 0; i < std::size(p); ++i) {
            p[i] = T(i);
          }
        }
        g.sync();
        if (rank < na) {
          transform.pure_to_cartesian(
            [&](auto &&i) { return p[index(i)]; },
            [&](auto &&i, auto v) { T(index(i)) = v; }
          );
        }
#undef T
      };
      pure_transform.apply(pure_to_cartesian, A.L);
    }
    if (!B.pure) {
      g.sync();
      auto pure_to_cartesian = [&](auto &&transform) {
        auto B = transform.L;
        double p[npure(B)];
        if (rank < na) {
#pragma unroll
          for (size_t i = 0; i < std::size(p); ++i) {
            p[i] = T[i+rank*npure(B)];
          }
        }
        g.sync();
        if (rank < na) {
          transform.pure_to_cartesian(
            [&](auto &&i) { return p[index(i)]; },
            [&](auto &&i, auto v) { T[index(i) + rank*ncart(B)] = v; }
          );
        }
      };
      pure_transform.apply(pure_to_cartesian, B.L);
    }
    g.sync();
  }

  struct E2 {

    __device__
    auto& value(int i, int j, int k, int x) {
      return data[i*strides[0]+j*strides[1]+k*strides[2]+x*strides[3]];
    }

    template<typename T>
    __device__
    auto operator()(T &&a, T &&b, T &&p) {
      double v = 1;
      for (int i = 0; i < 3; ++i) {
        v *= value(a[i], b[i], p[i], i);
      }
      return v;
    }

    template<typename R>
    __device__
    E2(double *data, int A, int B, double a, double b, const R &r) {
      this->data = data;
      strides[0] = (A+B+1)*(B+1);
      strides[1] = (A+B+1);
      strides[2] = 1;
      strides[3] = (A+B+1)*(B+1)*(A+1);
      auto p = a + b;
      assert(p);
      auto q = ((a ? a : 1)*(b ? b : 1))/p;
      assert(q);
      for (int i = threadIdx.x; i < 3*(A+B+1)*(B+1)*(A+1); i += 32) {
        this->data[i] = 0;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        value(0,0,0,0) = 1;
        value(0,0,0,1) = 1;
        value(0,0,0,2) = 1;
      }
      __syncthreads();
      for (int i = 1; i <= A; ++i) {
        __syncthreads();
        if (threadIdx.x > i) continue;
        auto k = threadIdx.x;
#pragma unroll
        for (int x = 0; x < 3; ++x) {
          double v0 = (k ? value(i-1,0,k-1,x) : 0);
          double v1 = value(i-1,0,k,x);
          double v2 = (k < i ? value(i-1,0,k+1,x) : 0);
          double v = (1/(2*p))*v0 - (q*r[x]/a)*v1 + (k+1)*v2;
          value(i,0,k,x) = v;
        }
      }
      // j
      for (int j = 1; j <= B; ++j) {
        for (int i = 0; i <= A; ++i) {
          __syncthreads();
          if (threadIdx.x > i+j) continue;
          auto k = threadIdx.x;
          for (int x = 0; x < 3; ++x) {
            double v0 = (k ? value(i,j-1,k-1,x) : 0);
            double v1 = value(i,j-1,k,x);
            double v2 = (k < i+j ? value(i,j-1,k+1,x) : 0);
            double v = (1/(2*p))*v0 + (q*r[x]/b)*v1 + (k+1)*v2;
            value(i,j,k,x) = v;
          }
        }
      }
    }

    double *data;
    int strides[4];

  };

  struct cartesian_to_hermite {

    template<typename Hermite>
    __device__
    static void apply(
      const Hermite &hermite,
      int A, double a, int B, double b,
      const Center &AB,
      double C, const double *G, double *H,
      thread_group g, double *shmem)
    {
      E2 E(shmem, A, B, a, b, AB);
      g.sync();
      for (int k = g.thread_rank(); k < hermite.nherm; k += g.size()) {
        auto Pk = hermite.orbitals()[k];
        double v = 0;
        for (int i = 0, ij = 0; i < ncart(A); ++i) {
          for (int j = 0; j < ncart(B); ++j, ++ij) {
            auto Ai = cartesian::orbital(A,i);
            auto Bj = cartesian::orbital(B,j);
            double e = E(Ai,Bj,Pk);
            v += e*G[ij];
          }
        }
        //printf("h[%i]=%f\n", k, C*v);
        H[k] = C*v;
      }
    }

  };

  struct hermite_to_cartesian {

    template<typename Hermite>
    __device__
    static void apply(
      const Hermite &hermite,
      int A, double a, int B, double b,
      const Center &AB,
      double C, const double *H, double *G,
      thread_group g, double *shmem)
    {
      E2 E(shmem, A, B, a, b, AB);
      g.sync();
      for (int k = g.thread_rank(); k < hermite.nherm; k += g.size()) {
        auto Pk = hermite.orbitals()[k];
        double Hk = H[k];
        for (int i = 0, ij = 0; i < ncart(A); ++i) {
          for (int j = 0; j < ncart(B); ++j, ++ij) {
            auto Ai = cartesian::orbital(A,i);
            auto Bj = cartesian::orbital(B,j);
            double e = E(Ai,Bj,Pk);
            double v = C*e*Hk;
            //printf("hk=%f\n", hk);
            atomicAdd(&G[ij], v);
          }
        }
      }
    }

  };

  struct Hermite1 {

    struct Memory {
      double E[3*(XMAX+1)*(XMAX+1)];
      double G[ncart(XMAX)];
    };

    const int P, nherm;
    Orbital orbitals_[nherm1(XMAX)];

    explicit Hermite1(int P)
      : P(P), nherm(nherm1(P))
    {
      constexpr auto orbitals = hermite::orbitals2<XMAX>;
      int k = 0;
      for (auto p : orbitals) {
        if (p.L()%2 != P%2) continue;
        assert(k < nherm1(XMAX));
        orbitals_[k++] = p;
      }
    }

    __device__
    auto& orbitals() const {
      return orbitals_;
    }

    __device__
    inline void transform(
      hermite_to_cartesian,
      const Index1 &idx,
      const Shell &A,
      const double *H,
      double *G) const
    {
      auto thread_block = this_thread_block();
      __shared__ Memory shmem;
      fill(ncart(A.L), shmem.G, 0.0, thread_block);
      thread_block.sync();
      for (int k = 0; k < A.K; ++k) {
        //memcpy(np, H+idx.kherm+k*np, shmem.H, thread_block);
        auto& [a,C] = A.prims[k];
        hermite_to_cartesian::apply(
          *this,
          A.L, a, 0, 0.0, {0,0,0},
          C, H+idx.kherm+k*this->nherm, shmem.G,
          thread_block, shmem.E
        );
        thread_block.sync();
      }
      if (A.pure) {
        // if (!threadIdx.x) {
        //   for (size_t i = 0; i < ncart(A.L); ++i) {
        //     printf("G[%i]=%f\n", i, shmem.G[i]);
        //   }
        // }
        cartesian_to_pure(A.L, shmem.G, thread_block);
      }
      for (size_t i = thread_block.thread_rank(); i < nbf(A); i += thread_block.size()) {
        atomicAdd(&G[i+idx.kbf], shmem.G[i]);
      }
      //memcpy(nbf(A), shmem.G, G+idx.kbf, thread_block);
    }

    __device__
    inline void transform(
      cartesian_to_hermite,
      const Index1 &idx,
      const Shell &A,
      const double *G,
      double *H) const
    {
      auto thread_block = this_thread_block();
      __shared__ Memory shmem;
      memcpy(nbf(A), G+idx.kbf, shmem.G, thread_block);
      if (A.pure) {
        thread_block.sync();
        pure_to_cartesian(A.L, shmem.G, thread_block);
      }
      thread_block.sync();
      for (int k = 0; k < A.K; ++k) {
        auto& [a,C] = A.prims[k];
        cartesian_to_hermite::apply(
          *this,
          A.L, a, 0, 0.0, {0,0,0},
          C, shmem.G, H+idx.kherm+k*this->nherm,
          thread_block, shmem.E
        );
      }
    }

    template<typename T>
    __device__
    inline void transform(
      T,
      const Index1 *index1,
      const Shell *basis,
      const double *src,
      double *dst)
    {
      auto thread_block = this_thread_block();
      __shared__ union static_shmem {
        struct {
          Index1 idx;
          Shell A;
        };
        __device__ static_shmem() {}
      } static_shmem;
      auto &idx = static_shmem.idx;
      memcpy1(&index1[blockIdx.x], &idx, thread_block);
      thread_block.sync();
      auto &A = static_shmem.A;
      memcpy1(&basis[idx.shell], &A, thread_block);
      thread_block.sync();
      this->transform(T{}, idx, A, src, dst);
    }

  };


  struct Hermite2 {

    int P;
    int nherm;

    explicit Hermite2(int P)
      : P(P), nherm(nherm2(P))
    {
    }

    __device__
    static const auto& orbitals() {
      return cartesian::orbitals2;
    }

    __device__
    void transform(
      hermite_to_cartesian,
      const Index2 *index2,
      const Shell *basis,
      const double *H,
      double *G)
    {

      auto thread_block = this_thread_block();

      __shared__ union static_shmem {
        struct {
          Index2 ij;
          Shell A,B;
          Center AB;
        };
        __device__ static_shmem() {}
      } static_shmem;

      auto &ij = static_shmem.ij;
      memcpy1(&index2[blockIdx.x], &ij, thread_block);
      __syncthreads();

      auto &A = static_shmem.A;
      auto &B = static_shmem.B;
      memcpy1(&basis[ij.first], &A, thread_block);
      memcpy1(&basis[ij.second], &B, thread_block);
      __syncthreads();

      __shared__ Center AB;
      if (threadIdx.x == 0) AB = A.r-B.r;
      __syncthreads();

      extern __shared__ double shmem[];

      int nab = ncart(A)*ncart(B);

      double *shmem_G = shmem;
      fill(nab, shmem_G, 0.0, thread_block);

      for (int ki = 0, kij = ij.kprim; ki < A.K; ++ki) {
        for (int kj = 0; kj < B.K; ++kj, ++kij) {

          auto& [ai,Ci] = A.prims[ki];
          auto& [aj,Cj] = B.prims[kj];
            // P = (AB| overlap
          double Kab = std::exp(-(ai*aj)/(ai+aj)*norm(AB));
          Kab *= math::sqrt_4_pi5;
          double sij = (ij.first == ij.second ? 1 : 2);
          // shmem values
          double a = ai;
          double b = aj;
          double C = sij*Ci*Cj*Kab;

          thread_block.sync();

          hermite_to_cartesian::apply(
            *this,
            A.L, a, B.L, b, AB,
            C, H+kij*this->nherm, shmem_G,
            thread_block, shmem+nab
          );

          thread_block.sync();

        } // kj
      } // ki

      cartesian_to_pure(A, B, shmem_G, thread_block);
      memcpy(nbf(A)*nbf(B), shmem_G, &G[ij.kbf], thread_block);

    }

    __device__
    void transform(
      cartesian_to_hermite,
      const Index2 *index2,
      const Shell *basis, Primitive2 *P,
      const double *G, double *H)
    {

      auto thread_block = this_thread_block();

      __shared__ union static_shmem {
        struct {
          Index2 ij;
          Shell A,B;
          Center AB;
        };
        __device__ static_shmem() {}
      } static_shmem;

      auto &ij = static_shmem.ij;
      memcpy1(&index2[blockIdx.x], &ij, thread_block);
      __syncthreads();

      auto &A = static_shmem.A;
      auto &B = static_shmem.B;
      memcpy1(&basis[ij.first], &A, thread_block);
      memcpy1(&basis[ij.second], &B, thread_block);
      __syncthreads();

      auto &AB = static_shmem.AB;
      if (threadIdx.x == 0) AB = A.r-B.r;
      __syncthreads();

      extern __shared__ double shmem[];

      double *shmem_G = nullptr;

      if (G) {
        shmem_G = shmem;
        memcpy(nbf(A)*nbf(B), &G[ij.kbf], shmem_G, thread_block);
        pure_to_cartesian(A, B, shmem_G, thread_block);
      }

      int nab = ncart(A)*ncart(B);

      for (int ki = 0, kij = ij.kprim; ki < A.K; ++ki) {
        for (int kj = 0; kj < B.K; ++kj, ++kij) {

          __syncthreads();

          __shared__ Primitive2 p1;
          __shared__ double a, b, C;

          if (thread_block.thread_rank() == 0) {
            auto& [ai,Ci] = A.prims[ki];
            auto& [aj,Cj] = B.prims[kj];
            // P = (AB| overlap
            double Kab = std::exp(-(ai*aj)/(ai+aj)*norm(AB));
            Kab *= math::sqrt_4_pi5;
            double sij = (ij.first == ij.second ? 1 : 2);
            // shmem values
            a = ai;
            b = aj;
            C = sij*Ci*Cj*Kab;
            p1.exp = { ai, aj };
            p1.r = { A.r, B.r };
            p1.C = C;
            p1.norm = ij.norm;
          }

          __syncthreads();

          memcpy1(&p1, &P[kij], thread_block);

          if (!shmem_G) continue;
          cartesian_to_hermite::apply(
            *this,
            A.L, a, B.L, b, AB,
            C, shmem_G, H+kij*this->nherm,
            thread_block, shmem+nab
          );

        } // kj
      } // ki

    }

  };


  template<typename T, typename H, typename ... Args>
  __global__
  static void transform(H h, Args ... args) {
    h.transform(T{}, args...);
  }

}
}

namespace libintx::gpu::jengine::md {

  constexpr int num_threads = 32*((ncart(max(XMAX,LMAX))+31)/32);

  //__global__
  void cartesian_to_hermite_2(
    int P, int nij, const Index2 *ijs,
    const Shell *basis, Primitive2 *Ps,
    const double *G, double *H,
    gpuStream_t stream)
  {
    // if (!G) {
    //   int shmem = 0;
    //   cartesian_to_hermite_basis<<<nij,32,shmem>>>(P, ijs, basis, Ps);
    //   return;
    // }
    int A = (P+1)/2;
    int B = P/2;
    int shmem = (
      (A+1)*(B+1)*(A+B+1)*3*8 +
      ncart(A)*ncart(B)*8
    );
    //printf("shmem=%i\n", shmem);
    transform<cartesian_to_hermite><<<nij,num_threads,shmem,stream>>>(
      Hermite2{P},
      ijs, basis, Ps, G, H
    );
  }

  void hermite_to_cartesian_2(
    int P, int N,
    const Index2 *index2,
    const Shell *basis,
    const double *H,
    double *G,
    gpuStream_t stream)
  {
    int A = (P+1)/2;
    int B = P/2;
    int shmem = (
      (A+1)*(B+1)*(A+B+1)*3*8 +
      ncart(A)*ncart(B)*8
    );
    //printf("shmem=%i\n", shmem);
    transform<hermite_to_cartesian><<<N,num_threads,shmem,stream>>>(
      Hermite2{P},
      index2, basis, H, G
    );
  }

  void cartesian_to_hermite_1(
    int p, int n,
    const Index1 *index1,
    const Shell *basis,
    const double* G, double* H)
  {
    transform<cartesian_to_hermite><<<n,num_threads>>>(
      Hermite1{p},
      index1, basis, G, H
    );
  }

  void hermite_to_cartesian_1(
    int p, int n,
    const Index1 *index1,
    const Shell *basis,
    const double* H, double* G)
  {
    transform<hermite_to_cartesian><<<n,num_threads>>>(
      Hermite1{p},
      index1, basis, H, G
    );
  }

}
