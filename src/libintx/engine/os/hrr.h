#ifndef LIBINTX_ENGINE_OS_HRR1_H
#define LIBINTX_ENGINE_OS_HRR1_H

#include "libintx/array.h"
#include "libintx/orbital.h"
#include "libintx/simd.h"

#include <exception>

namespace libintx::os::hrr {

  // 540  1.838  1.775  1.790
  // 550  1.742  1.831  1.841
  // 640  1.874  1.877  1.852

  template<int A, int NX>
  void hrr_axis_transfer_yz(double X, const double *V1, const double *V2, double *H) {
    int Idx0 = ncart(A-1);
    int Idx1 = ncart(A)-1;
    simd::multiply_add_store(NX*(A+1), X, V1+Idx0*NX, V2+Idx1*NX, H+Idx0*NX);
    if constexpr (A > 0) {
      hrr_axis_transfer_yz<A-1,NX>(X, V1, V2, H);
    }
  }

  template<int A, int NX, int Axis>
  void hrr_axis_transfer(double X, const double *V1, const double *V2, double *H) {
    if (A <= 0) return;
    V2 += Axis*NX; // V2 pattern starts at either 0,1,2
    if constexpr (Axis == 0) {
      // x axis just single block operation
      simd::multiply_add_store(NX*ncart(A), X, V1, V2, H);
    }
    else {
      hrr_axis_transfer_yz<A,NX>(X, V1, V2, H);
    }
  }

  // H(A,T+1) = X*V1(A,T) + V2(A+1,T);
  template<int A, int T, int NX>
  void hrr_block_transfer(const double *X, const double *V1, const double *V2, double *H) {
#define V1(IDX) (V1+NX*(IDX*ncart(A+0)))
#define V2(IDX) (V2+NX*(IDX*ncart(A+1)))
    // X-component
    for (int j = 0; j < ncart(T); ++j) {
      hrr_axis_transfer<A,NX,0>(X[0], V1(j), V2(j), H);
      H += NX*ncart(A);
    }
    // Y-component
    for (int j = ncart(T-1); j < ncart(T-1)+T+1; ++j) {
      hrr_axis_transfer<A,NX,1>(X[1], V1(j), V2(j), H);
      H += NX*ncart(A);
    }
    // Z-component
    {
      constexpr int j = ncart(T)-1;
      hrr_axis_transfer<A,NX,2>(X[2], V1(j), V2(j), H);
    }
#undef V1
#undef V2
  }

  // (A:AB-1|T+1) = X*(A:AB-1|T) + (A+1:AB,T)
  template<int A, int AB, int T, int NX>
  void hrr_block_transfer(const double *X, const double *V, double *H) {
    if constexpr (A < AB) {
      auto *V1 = V;
      auto *V2 = V+NX*ncart(A)*ncart(T);
      hrr_block_transfer<A,T,NX>(X, V1, V2, H);
      // do (A+1) block
      V += NX*ncart(A)*ncart(T);
      H += NX*ncart(A)*ncart(T+1);
      hrr_block_transfer<A+1,AB,T,NX>(X, V, H);
    }
  }

  template<int A, int AB, int NX, int T = 0>
  auto hrr_transfer(const double *Xab, double *V, double *U) {
    if constexpr (A < AB) {
      // write to H if this transfer is last
      hrr_block_transfer<A,AB,T,NX>(Xab, V, U);
      // swap U/V, advance right
      hrr_transfer<A,AB-1,NX,T+1>(Xab, U, V);
    }
  }

  template<int AB, int NX>
  bool hrr1(int A, int B, const double *Xab, double *V, double *U) {
    if (B == 0 || A == 0) return 0;
    if (A > AB) {
      throw std::logic_error("Invalid parameter A/AB in HRR");
    }
    typedef void(*HRR)(const double*, double*, double*);
    //typedef std::function<void(const double*, double*, double*, double*)> Kernel;
    static constexpr auto hrr_table = make_array<HRR,AB+1>(
      [](auto Idx) {
        constexpr int A = Idx.value;
        return HRR(&hrr_transfer<A,AB,NX,0>);
      }
    );
    hrr_table[A](Xab, V, U);
    return (B%2);
  }

  template<int A, int B>
  void hrr1(const double *Xab, double *V, double *H) {
    assert(false);
  }



}

#endif /* LIBINTX_ENGINE_OS_HRR1_H */
