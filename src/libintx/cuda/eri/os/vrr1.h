#ifndef LIBINTX_CUDA_OS_VRR1_H
#define LIBINTX_CUDA_OS_VRR1_H

#include "libintx/array.h"
#include "libintx/orbital.h"
#include "libintx/recurrence.h"
#include <utility>

namespace libintx::cuda::os::vrr1 {

  using libintx::recurrence::recurrence;

  struct Index {
    __device__
    Index() {}
    template<int A, int B, typename I, typename J, typename K>
    static constexpr uint32_t index(I i, J j, K k) {
      constexpr uint32_t idx = recurrence<A,B>(i,j,k).index;
      if constexpr (idx) {
          constexpr uint32_t Idx0 = cartesian::index(i+j+k-A-B);
          return (idx - Idx0);
      }
      return 0;
    }
    template<typename I, typename J, typename K>
    constexpr Index(I i, J j, K k)
    : idx02(index<2,0>(i,j,k)),
      idx11(index<1,1>(i,j,k)),
      idx12(index<1,2>(i,j,k)),
      axis1(recurrence<1>(i,j,k).axis),
      axis2(recurrence<1,1>(i,j,k).axis),
      c11(recurrence<1,1>(i,j,k).value),
      c02(recurrence<1>(i,j,k).value)
    {
      static_assert(index<2,0>(i,j,k) < 1<<7);
      static_assert(index<1,1>(i,j,k) < 1<<7);
      static_assert(index<1,2>(i,j,k) < 1<<6);
      static_assert(recurrence<1,1>(i,j,k).value < 1<<4);
      static_assert(recurrence<1>(i,j,k).value < 1<<4);
      //printf("vrr1::Index(%i,%i,%i) idx02=%i\n", i, j, k, idx02);
    }
    uint32_t idx02:7; //    ,A-2
    uint32_t idx11:7; // A-1,A-1
    uint32_t idx12:6; // A-1,A-2
    uint32_t c11:4;
    uint32_t c02:4;
    uint32_t axis1:2;
    uint32_t axis2:2;
    struct Generator;
  };

  static_assert(sizeof(Index) == 4);

  struct Index::Generator {
    template<size_t Idx>
    constexpr Index operator()(std::integral_constant<size_t,Idx> idx) const {
      constexpr auto f = cartesian::orbital(idx);
      constexpr int I = f[0];
      constexpr int J = f[1];
      constexpr int K = f[2];
      return Index(
        std::integral_constant<int,I>{},
        std::integral_constant<int,J>{},
        std::integral_constant<int,K>{}
      );
    }
  };

  __constant__
  //__device__
  const auto index = make_array<Index>(
    Index::Generator(),
    cartesian::index_sequence_ab
  );

  template<int L>
  __host__ __device__ __forceinline__
  constexpr int leading_dimension(int m = 1) {
    if (L == 0) return 1*m;
    if (L == 1) return 1 + 3*m;
    if (L == 2) return 1 + 9*m;
    if constexpr (L > 2) {
      int ld0 = m*(ncart(L) + ncart(L-1)) + ncartsum(L-2);
      int ld2 = leading_dimension<L-2>(m+2);
      return (ld0 > ld2 ? ld0 : ld2);
    }
  }


  // Compute (L+0)[M+1] and (L+1)[M]
  template<int M, int L>
  __device__  __forceinline__
  void transfer2(
    const Index &idx,
    const double *Xpa, const double &one_over_2p,
    const double *Xpq, const double &alpha_over_p,
    const double *A,
    double (&V0)[M+1], double (&V1)[M])
  {
    static constexpr int Stride = (ncart(L-1) + ncart(L-2));
#define A(idx,m) A[(idx) + (m)*Stride];
    // L+0
    {
      int axis = idx.axis2;
      int idx11 = cartesian::index(L-1) + idx.idx11;
      int idx12 = cartesian::index(L-2) + idx.idx12;
      double X0 = (Xpa[axis]);
      double X1 = (Xpq[axis]);
      double c0 = idx.c11*one_over_2p;
      double c1 = c0*alpha_over_p;
#pragma unroll
      for (int m = 0; m < M+1; ++m) {
        double A10 = A(idx11,m+0);
        double A11 = A(idx11,m+1);
        double A20 = A(idx12,m+0);
        double A21 = A(idx12,m+1);
        V0[m] = (X0*A10 - X1*A11) + (c0*A20 - c1*A21);
      }
    }
    // L+1
    {
      int axis = idx.axis1;
      int idx02 = cartesian::index(L-1) + idx.idx02;
      double Xa = (Xpa[axis]);
      double Xc = (Xpq[axis]);
      double c0 = idx.c02*one_over_2p;
      double c1 = c0*alpha_over_p;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        double A10 = V0[m+0];
        double A11 = V0[m+1];
        double A20 = A(idx02, m+0);
        double A21 = A(idx02, m+1);
        V1[m] = (Xa*A10 - Xc*A11) + (c0*A20 - c1*A21);
        // printf("d_%i[%i]=%f\n", m, idx, A1[m]);
        // printf("f_%i[%i]=%f, c02=%f\n", m, idx, V0[m], double(idx.c02));
      }
    }
#undef A
  }

  template<int L, int M>
  __device__  __forceinline__
  void transfer(
    int K,
    const double *Xpa, const double *one_over_2p,
    const double *Xpq, const double *alpha_over_p,
    double *A, int ldA)
  {

    __syncthreads();

    // Ks done by block at once
    int KB = (blockDim.x*blockDim.y)/ncart(L);

    Index idx;

    for (int kb = 0; kb < K; kb += KB) {
      int x = threadIdx.x + threadIdx.y*32;
      int k = x/ncart(L) + kb;
      double *Ak = nullptr;
      double A0[M+1] = {};
      double A1[M] = {};
      if (k < min(K,kb+KB)) {
        Ak = A + k*(ldA);
        if (kb == 0) {
          idx = vrr1::index[cartesian::index(L) + x%ncart(L)];// load index on first hit
        }
        transfer2<M,L-1>(
          idx,
          Xpa+3*k, one_over_2p[k],
          Xpq+3*k, alpha_over_p[k],
          Ak, A0, A1
        );
      }
      __syncthreads();
      if (Ak) {
        int idx = x%ncart(L);
        Ak += ncartsum(L-2);
#pragma unroll
        for (int m = 0; m < M; ++m) {
          if (idx < ncart(L-1)) {
            Ak[idx] = A0[m];
            //printf("VRR1(L=%i,m=%i)[%i]=%f\n", L-1, m, idx, A0[m]);
          }
          Ak[idx + ncart(L-1)] = A1[m];
          //printf("VRR1(L=%i,m=%i)[%i]=%f\n", L, m, idx, A1[m]);
          Ak += (ncart(L-1)+ncart(L));
        }
      }
    }

    if constexpr (M > 1) {
      transfer<L+2,M-2>(K, Xpa, one_over_2p, Xpq, alpha_over_p, A, ldA);
    }

  }

  template<int M>
  __device__  __forceinline__
  void transfer_p(
    int K,
    const double *Xpa, const double *one_over_2p,
    const double *Xpq, const double *alpha_over_p,
    double *A, int ldA)
  {

    __syncthreads();

    // each warp does 10 prims at time guarantees no inter-warp race conditions
    if (threadIdx.x >= 30) {
      return;
    }

    static const int L = 1;
    int idx = (threadIdx.x + threadIdx.y*30)%ncart(L);
    int k = (threadIdx.x + threadIdx.y*30)/ncart(L);

    for (; k < K; k += 10*blockDim.y) {
      auto active_mask = __activemask();
      double *Ak = A + k*(ldA);
      double s[M+1] = {};
#pragma unroll
      for (int m = 0; m < M+1; ++m) {
        s[m] = Ak[m];
      }
      __syncwarp(active_mask);
      double Xa = (Xpa[idx + 3*k]);
      double Xc = (Xpq[idx + 3*k]);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        double p = s[m+0]*Xa - s[m+1]*Xc;
        Ak[1 + idx + m*4] = p;
      }
      if (idx == 0) {
        for (int m = 0; m < M; ++m) {
          Ak[0 + m*4] = s[m];
        }
      }
    }
  }

  template<int M>
  __device__  __forceinline__
  void transfer_pd(
    int K,
    const double *Xpa, const double *one_over_2p,
    const double *Xpq, const double *alpha_over_p,
    double *A, int ldA)
  {

    __syncthreads();

    // each warp does 5 prims at time guarantees no inter-warp race conditions
    if (threadIdx.x >= 30) {
      return;
    }

    static const int L = 2;
    static const uint32_t index2 =
      0b0000 << 0  | // px,px
      0b0100 << 4  | // py,px
      0b1000 << 8  | // pz,px
      0b0101 << 12 | // py,py
      0b1001 << 16 | // pz,py
      0b1010 << 20   // pz,pz
      ;

    int k = (threadIdx.x + threadIdx.y*30)/ncart(L);
    int idx = (threadIdx.x + threadIdx.y*30)%ncart(L);

    int idx1 = (index2 >> idx*4+0) & 0b11;
    int idx2 = (index2 >> idx*4+2) & 0b11;
    int m12 = (idx1 == idx2);

    for (; k < K; k += 5*blockDim.y) {
      auto active_mask = __activemask();
      double ck = one_over_2p[k]*m12;
      double *Ak = A + k*(ldA);
      double s[M+2];
      double Xa2 = (Xpa[idx2 + 3*k]);
      double Xc2 = (Xpq[idx2 + 3*k]);
      double Xa1 = (Xpa[idx1 + 3*k]);
      double Xc1 = (Xpq[idx1 + 3*k]);
      for (int m = 0; m < M+2; ++m) {
        s[m] = Ak[m];
      }
      __syncwarp(active_mask);
      Ak += (ncartsum(L-2));
#pragma unroll
      for (int m = 0; m < M; ++m) {
        double p0 = s[m+0]*Xa2 - s[m+1]*Xc2;
        double p1 = s[m+1]*Xa2 - s[m+2]*Xc2;
        if (idx < ncart(L-1)) {
          Ak[idx] = p0;
        }
        Ak += ncart(L-1);
        double d = Xa1*p0 - Xc1*p1;
        d += ck*(s[m+0] - alpha_over_p[k]*s[m+1]);
        Ak[idx] = d;
        Ak += ncart(L);
      }
    }

  }


  template<int L>
  __device__ // __forceinline__
  void vrr1(
    int K,
    const double *Xpa, const double *one_over_2p,
    const double *Xpq, const double *alpha_over_p,
    double *A, int ldA)
  {
    if constexpr (L && L%2 == 1) {
      transfer_p<L>(K, Xpa, one_over_2p, Xpq, alpha_over_p, A, ldA);
    }
    if constexpr (L && L%2 == 0) {
      transfer_pd<L-2+1>(K, Xpa, one_over_2p, Xpq, alpha_over_p, A, ldA);
    }
    if constexpr (L > 2) {
      constexpr int Next = (L%2 == 1 ? 3 : 4);
      transfer<Next,L-Next+1>(K, Xpa, one_over_2p, Xpq, alpha_over_p, A, ldA);
    }
  }

}

#endif /* LIBINT_OS_VRR1_H */
