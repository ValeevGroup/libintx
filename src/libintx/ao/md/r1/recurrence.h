#ifndef LIBINTX_MD_R1_RECURRENCE_H
#define LIBINTX_MD_R1_RECURRENCE_H

#include "libintx/forward.h"
#include "libintx/math.h"
#include "libintx/orbital.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#include "libintx/gpu/api/intrinsics.h"
#endif

namespace libintx::md::r1 {

  template<int L>
  struct Recurrence {

    struct Index {
      uint32_t n:6;
      uint32_t idx0:12;
      uint32_t x:2;
      uint32_t idx1:12;
    } __attribute__((aligned(4)));
    static_assert(sizeof(Index) == 4);

    array<Index,ncartsum(L)> table = make_table();

    constexpr Recurrence() = default;

    LIBINTX_GPU_DEVICE LIBINTX_ALWAYS_INLINE
    constexpr auto operator[](int idx) const {
#if defined(__CUDACC__) || defined(__HIPCC__)
      return gpu::ldg(this->table+idx);
#else
      return this->table[idx];
#endif
    }

    static constexpr auto make_table() {
      using cartesian::Orbital;
      using cartesian::index;
      constexpr auto orbitals = hermite::orbitals2<L>;
      array<Index,ncartsum(L)> table = {};
      for (int i = 1; i < ncartsum(L); ++i) {
        auto p = orbitals[i];
        for (int x = 0; x< 3; ++x) {
          if (!p[x]) continue;
          int n = p[x]-1;
          // [A+2] = X*[A+1] + n*[A+0]
          auto A0 = (n ? p - Orbital::Axis(x,2) : Orbital{0,0,0});
          auto A1 = p - Orbital::Axis(x,1);
          table[i] = {
            uint32_t(n), uint32_t(index<0>(A0)),
            uint32_t(x), uint32_t(index<0>(A1))
          };
          break;
        }
      }
      return table;
    }

  };

  template<int L, int Max, typename T>
  LIBINTX_GPU_DEVICE LIBINTX_ALWAYS_INLINE
  void compute(
    const Recurrence<Max> &recurrence,
    const array<T,3> &PQ,
    T* __restrict__ r1)
  {
    static_assert(L <= Max);
    for (int l = 1; l <= L; l += 1) {
      int nl = nherm2(l)-1;
      // [A+2] = X[A+1] + n[A+0];
      T v[nherm2(L)];
      auto* __restrict__ r1_l = (r1+L-l+1);
      for (int k = 0; k < nl; ++k) {
        const auto& [n,idx0,ix,idx1] = recurrence[1+k];
        v[k] = PQ[ix]*r1_l[idx1] + T(n)*r1_l[idx0];
      }
      for (int k = 0; k < nl; ++k) {
        (r1_l)[k] = v[k];
      }
    }
  }

#if defined(__CUDACC__) || defined(__HIPCC__)

  template<int L, int Max, typename G>
  LIBINTX_GPU_DEVICE LIBINTX_ALWAYS_INLINE
  void compute(
    const Recurrence<Max> &recurrence,
    const array<double,3> &PQ,
    double* __restrict__ r1,
    const G &thread_group)
  {
    static_assert(L <= Max);

    constexpr int num_threads = G::size();
    constexpr int N = (nherm2(L)-1+num_threads-1)/num_threads;
    int rank = thread_group.thread_rank();

    // [A+2] = X[A+1] + n[A+0];
    double X[N] = {};
    double n[N] = {};
    int idx[N][2] = {};

#pragma unroll
    for (int k = 0; k < N; ++k) {
      int i = rank + k*num_threads;
      if (k+1 == N && i >= nherm2(L)-1) break;
      auto idx1 = recurrence[1+i];
      X[k] = PQ[idx1.x];
      n[k] = idx1.n;
      idx[k][0] = idx1.idx0;
      idx[k][1] = idx1.idx1;
    }

    for (int l = 1; l <= L; l += 1) {
      int nl = nherm2(l)-1;
      int nk = (nl+num_threads-1)/num_threads;
      double v[N];
      double* __restrict__ r1_l = (r1+L-l+1);
      thread_group.sync();
#pragma unroll
      for (int k = 0; k < nk; ++k) {
        //int i = rank + k*num_threads;
        //if (k+1 == nk && i >= nl) break;
        v[k] = X[k]*r1_l[idx[k][1]] + n[k]*r1_l[idx[k][0]];
        // printf(
        //   "xxxx %i %f=%f*%f + %f*%f\n",
        //   i, v[k], X[k], r1_l[idx[k][1]], n[k], r1_l[idx[k][0]]
        // );
      }
      thread_group.sync();
#pragma unroll
      for (int k = 0; k < nk; ++k) {
        int i = rank + k*num_threads;
        if (k+1 == nk && i >= nl) break;
        (r1_l)[i] = v[k];
      }
    }

  }

#endif

}

#endif /* LIBINTX_MD_R1_RECURRENCE_H */
