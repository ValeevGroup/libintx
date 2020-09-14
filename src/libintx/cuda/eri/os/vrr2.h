#ifndef LIBINT_OS_CUDA_VRR2_H
#define LIBINT_OS_CUDA_VRR2_H

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/recurrence.h"
#include "libintx/cuda/api/thread_group.h"
#include <utility>

namespace libintx::cuda::os::vrr2 {

  struct Index1 {

    uint16_t index:6;
    uint16_t coefficient:10;

    struct Generator {
      template<typename X, typename T>
      constexpr Index1 operator()(X x, T t) {
        recurrence::Index::Generator g;
        constexpr auto xt = g(t,x);
        if constexpr (xt.L <= (XMAX)) {
          constexpr auto Index = xt.index;
          std::integral_constant<size_t,Index> idx;
          constexpr auto L = cartesian::orbital(idx).L();
          constexpr auto index = xt.index - cartesian::index(L);
          constexpr auto coefficient = xt.multiplicity;
          static_assert(index < (1<<6));
          static_assert(coefficient < (1<<10));
          return { index, coefficient };
        }
        return { 0x3F, 0x3FF };
      }
    };

  };

  struct Index2 {

    uint32_t index:9;
    uint32_t coefficient:23;

    struct Generator {
      template<typename X, typename A>
      constexpr Index2 operator()(X x, A a) {
        recurrence::Index::Generator g;
        constexpr auto ax = g(x,a);
        if constexpr (ax.L <= (2*LMAX)) {
          static_assert(ax.index < (1<<9));
          static_assert(ax.coefficient < (1<<23));
          return {
            (uint32_t)ax.index,
            (uint32_t)ax.coefficient
          };
        }
        return { 0x1FF, 0x7FFFF };
      }
    };

  };

  __constant__
  const auto orbital_axis = make_array<uint16_t>(
    [](auto Idx) {
      return bitstring(cartesian::orbital(Idx));
    },
    cartesian::index_sequence_x
  );

  __constant__
  const auto index1_table = make_array<Index1>(
    Index1::Generator(),
    cartesian::index_sequence_x,
    cartesian::index_sequence_x
  );

  __device__
  const auto index2_table = make_array<Index2>(
    Index2::Generator(),
    cartesian::index_sequence_x,
    cartesian::index_sequence_ab
  );

  __device__
  inline auto index1(int i, int j) {
    return index1_table[i][j];
    // auto a = cartesian::orbital_list[i];
    // auto b = cartesian::orbital_list[j];
    // auto ab = a + b;
    // uint16_t m = 1;
    // for (size_t i = 0; i < 3; ++i) {
    //   m *= index1_coefficient_table[a[i]][b[i]];
    // }
    // return Index1{ (uint16_t)cartesian::index(ab), m };
  }

  __device__
  inline auto index2(int x, int ab) {
    assert(x < ncartsum(XMAX));
    assert(ab < ncartsum(2*LMAX));
    return index2_table[x][ab];
    // auto X = cartesian::orbital_list[x];
    // auto AB = cartesian::orbital_list[ab];
    // uint16_t m = 1;
    // for (size_t i = 0; i < 3; ++i) {
    //   m *= index2_coefficient_table[AB[i]][X[i]];
    // }
    // return Index2{ (uint16_t)cartesian::index<0>(AB+X), m };
  };

  template<int X>
  __device__ __forceinline__
  double Rt(const double *R, const uint16_t axis) {
    double Rt = R[axis & 0b11];
    //#pragma unroll
    for (int i = 1; i < X; ++i) {
      Rt *= R[(axis >> i*2) & 0b11];
      // int xy = (i >= t[0]); // 0 for x, 1 otherwise
      // int z  = (i >= X-t[2]); // 1 for z, 0 otherwise
      // Rt *= R[xy+z];
    }
    // for (int i = 0; i < 3; ++i) {
    //   Rt *= pow(R[i], t[i]);
    // }
    return Rt;
  }

  template<int AB, int X>
  __device__ __forceinline__
  void evaluate_terms(
    int A, int B,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, int LD1,
    double *V2,
    std::integral_constant<int,X>,
    std::integral_constant<int,X>)
  {
    if (X > AB) return;

    int thread_rank = this_thread_block().thread_rank();
    int num_threads = this_thread_block().size();

    // optimization trick, pre-contract V1
    __syncthreads();
    for (int i = cartesian::index(A-X)+thread_rank; i < cartesian::index(AB-X+1); i += num_threads) {
      double v2 = 0;
      for (int k = 0; k < K; ++k) {
        v2 += alpha_over_2pc[k]*V1[i + k*LD1];
      }
      V1[i] = v2;
    }

    static constexpr int NX = ncart(X);
    int AX = cartesian::index(A-X)*NX;

    __syncthreads();

    for (int ax = AX + thread_rank; ax < cartesian::index(AB-X+1)*NX; ax += num_threads) {
      int x = ax%NX;
      int a = ax/NX;
      double v2 = V1[a];
      auto idx = index2(x+cartesian::index(X),a);
      V2[x + idx.index*NX] += idx.coefficient*v2;
    //     //V2[cartesian::index<0>(a) + j*LD2] += c*v2;
    }


    // for (int i = cartesian::index(A-X); i < cartesian::index(AB-X+1); ++i) {
    //   double v2 = V1[i];
    //   for (int j = thread_rank; j < ncart(X); j += num_threads) {
    //     auto ab = index2(j+cartesian::index(X),i);
    //     V2[j + ab.index*ncart(X)] += ab.coefficient*v2;
    //     //V2[cartesian::index<0>(a) + j*LD2] += c*v2;
    //   }
    // }

  }

  // }
  template<int AB, int X, int T>
  __device__ //__forceinline__
  void evaluate_terms(
    int A, int B,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, int LD1, double *V2,
    std::integral_constant<int,X>,
    std::integral_constant<int,T>)
  {
    if (T > AB) return;

    int thread_rank = this_thread_block().thread_rank();
    int num_threads = this_thread_block().size();

    // optimization trick, pre-scale V1
    __syncthreads();
    for (int i = 0+thread_rank; i < cartesian::index(AB-T+1); i += num_threads) {
      for (int k = 0; k < K; ++k) {
        V1[i + k*LD1] *= alpha_over_2pc[k];
      }
    }

    static constexpr int NX = ncart(X-T);
    static constexpr int XT = cartesian::index(X-T);
    const int AT = cartesian::index(A-T);

    double* Xk = V2 + ncart(0,AB)*ncart(X);
    for (int kx = thread_rank; kx < NX*K; kx += num_threads) {
      int k = kx/NX;
      int x = kx%NX;
      Xk[kx] = vrr2::Rt<X-T>(Xpc+k*3, orbital_axis[x+XT]);
    }
    __syncthreads();

    for (int ax = thread_rank; ax < NX*ncart(A-T,AB-T); ax += num_threads) {
      int a = ax/NX + AT;
      int x = ax%NX;
      double v2 = 0;
      for (int k = 0; k < K; ++k) {
        double xk = Xk[x+k*NX];
        //double xk = vrr2::Rt<X-T>(Xpc+k*3, orbital_axis[x+XT]);
        double v1 = V1[a+k*LD1];
        v2 += xk*v1;
      }
      for (int t = cartesian::index(T); t < cartesian::index(T+1); ++t) {
        auto xt = index1(t,x+XT);
        auto at = index2(t,a);
        double c = (at.coefficient*xt.coefficient);
        // printf(
        //   "X=%i, T=%i, x,t=%i,%i, xt=%i, at=%i\n",
        //   X, T, x, t, xt.index, at.index
        // );
        //V2[xt.index + at.index*ncart(X)] += c*v2;
        atomicAdd(&V2[xt.index + at.index*ncart(X)], c*v2);
      }
    }
    //end:

    evaluate_terms<AB>(
      A, B, K, Xpc, alpha_over_2pc, V1, LD1, V2,
      std::integral_constant<int,X>{},
      std::integral_constant<int,T+1>{}
    );

  }

  // Term-0 + rest of terms
  template<int AB, int X>
  __device__ __forceinline__
  void evaluate_terms(
    int A, int B,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, int LD1,
    double *V2)
  {
    auto thread_block = this_thread_block();
    int thread_rank = thread_block.thread_rank();
    int num_threads = thread_block.size();

    static constexpr int NX = ncart(X);

    double* Xk = V2 + ncart(0,AB)*NX;
    for (int kx = thread_rank; kx < NX*K; kx += num_threads) {
      int k = kx/NX;
      int x = kx%NX;
      Xk[kx] = vrr2::Rt<X>(Xpc+k*3, orbital_axis[x+cartesian::index(X)]);
    }
    __syncthreads();

    int A0 = cartesian::index(A);
    for (int ax = thread_rank + A0*NX; ax < ncart(0,AB)*NX; ax += num_threads) {
      int x = ax%NX;
      int a = ax/NX;
      double v = 0;
      for (int k = 0; k < K; ++k) {
        double xk = Xk[x+k*NX];
        v += xk*V1[a + k*LD1];
      }
      V2[ax] += v;
    }

    evaluate_terms<AB>(
      A, B, K, Xpc, alpha_over_2pc, V1, LD1, V2,
      std::integral_constant<int,X>{},
      std::integral_constant<int,1>{}
    );

  }

  template<int AB, int X>
  __device__ __forceinline__
  void vrr2(
    int A, int B,
    int K, const double *Xpc, const double *alpha_over_2pc,
    double *V1, int LD1,
    double *V2)
  {
    V2 -= cartesian::index(A)*ncart(X);
    if constexpr (X == 0) {
      auto thread_block = this_thread_block();
      int thread_rank = thread_block.thread_rank();
      int num_threads = thread_block.size();
      for (int i = cartesian::index(A)+thread_rank; i < cartesian::index(AB+1); i += num_threads) {
        double s = 0;
        for (int k = 0; k < K; ++k) {
          s += V1[i + k*LD1];
        }
        //printf("vrr2[%i]: idx=%i, v=%f\n", i, i + ncartsum(A-1), s);
        V2[i] += s;
      }
    }
    else {
      evaluate_terms<AB,X>(A, B, K, Xpc, alpha_over_2pc, V1, LD1, V2);
    }
  }

}

#endif /* LIBINT_OS_CUDA_VRR2_H */
