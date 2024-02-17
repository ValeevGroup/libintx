#ifndef LIBINTX_ENGINE_OS_OS3_H
#define LIBINTX_ENGINE_OS_OS3_H

#include "libintx/engine/os/vrr1.h"
#include "libintx/engine/os/vrr2.h"
#include "libintx/engine/os/hrr.h"

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/pure.h"
#include "libintx/utility.h"

#include <utility>
#include <functional>
#include <memory>
#include <type_traits>
#include <cassert>
#include <stdio.h>

namespace libintx::os {

using Vector3 = array<double,3>;

struct Stack {
  Stack() = default;
  explicit Stack(size_t n)
    : size_(n)
  {
    data_.reset(new double[n]);
  }
  double* base() {
    return data_.get();
  }
  void reset(size_t idx = 0) {
    offset_ = idx;
  }
  double* push(size_t n) {
    assert(offset_+n <= size_);
    double *ptr = this->base() + offset_;
    offset_ += n;
    return ptr;
  }
private:
  std::unique_ptr<double[]> data_;
  size_t size_ = 0;
  size_t offset_ = 0;
};


struct ObaraSaika3 {

  virtual ~ObaraSaika3() = default;

  virtual void compute(const Double<3> &rA, const Double<3> &rB, const Double<3> &rX, double*) const = 0;

  template<int AB, int X, class Boys>
  struct Kernel;

  template<class Boys>
  static auto kernel(const Gaussian &A, const Gaussian &B, const Gaussian &X, const Boys &boys) {
    return make_ab_x_kernel< std::unique_ptr<ObaraSaika3> >(
      [&](auto ab, auto x) {
        return std::make_unique< Kernel<ab.value,x.value,Boys> >(A,B,X,boys);
      },
      A.L+B.L, X.L
    );
  }

};

template<int _AB, int _X, class Boys>
struct ObaraSaika3::Kernel : ObaraSaika3 {

  static constexpr int L = (_AB + _X);
  static constexpr int N = (_AB + 1);

  const Shell A, B;

  struct AB {
    static constexpr int L = _AB;
    const int K;
    struct Primitive {
      double a, b, C;
    };
    std::vector<Primitive> prims;
  } AB;

  struct X {
    static constexpr int L = _X;
    const int K;
    std::vector<Gaussian::Primitive> prims;
    explicit X(const Gaussian &x) : K(x.K) {
      prims.resize(K);
      for (int k = 0; k < K; ++k) {
        this->prims[k] = x.prims[k];
      }
    }
  } X;

  const Boys boys;
  mutable Stack stack;

  int nbf() const {
    return ncart(A)*ncart(B)*ncart(X::L);
  }

  Kernel(const Gaussian &A, const Gaussian &B, const Gaussian &X, const Boys &boys)
    : A(A), B(B), AB{A.K*B.K}, X(X), boys(boys)
  {
    using libintx::nbf;
    int K = AB.K;
    AB.prims.resize(K);
    for (int j = 0; j < B.K; ++j) {
      for (int i = 0; i < A.K; ++i) {
        int k = i + j*A.K;
        AB.prims[k].a = A.prims[i].a;
        AB.prims[k].b = B.prims[j].a;
        double C = A.prims[i].C*B.prims[j].C;
        AB.prims[k].C = C*2*math::sqrt_4_pi5;
        //printf("AB[%i] = %f,%f,%f\n", k, AB.prims[k].a, AB.prims[k].b, AB.prims[k].C);
      }
    }
    const int NAB = ncartsum(AB.L)-ncartsum(A.L-1);
    using std::max;
    int size = 0;
    int hrr1 = (ncart(A.L)*ncart(B.L)*npure(X.L) + NAB*ncart(B.L-1)*npure(X.L));
    size = max(size, K*(ncartsum(AB::L) + 12)); // VRR1
    size = max(size, NAB*npure(X::L)); // VRR2 spherical transform
    size = max(size, hrr1); // HRR1
    size += max(
      nbf(B)*ncart(A)*nbf(X), // max cartesian_to_pure
      NAB*ncart(X::L) // VRR2
    );
    this->stack = Stack(size);
  }

  void compute(const Double<3> &rA, const Double<3> &rB, const Double<3> &rX, double *buffer) const override {

    using libintx::nbf;

    const int NAB = ncartsum(AB.L)-ncartsum(A.L-1);
    const int K = AB.K;

    stack.reset();
    double *V2 = stack.push(NAB*ncart(X::L));

    double *V1 = stack.push(ncartsum(AB::L)*K);
    double *alpha_over_p = stack.push(K);
    double *one_over_2p = stack.push(K);
    double *Xpa = stack.push(3*K);
    double *Xpq = stack.push(3*K);

    double *alpha_over_2pq = stack.push(K);
    double *X2 = stack.push(3*K);

    std::fill(V2, V2 + NAB*ncart(X::L), 0);

    for (int k = 0; k < K; ++k) {

      double *Vk = V1 + k*ncartsum(AB::L);

      double C = AB.prims[k].C*X.prims[0].C;
      double a = AB.prims[k].a;
      double b = AB.prims[k].b;
      double q = X.prims[0].a;
      double p = (a+b);
      double alpha = p*q/(p+q);

      Vector3 P = center_of_charge(a, rA, b, rB);
      const auto &Q = rX;

      double Fm[N];
      boys.compute(alpha*norm(P,Q), X::L, Fm);

      double Kab = std::exp(-(a*b)/(a+b)*norm(rA,rB));
      double Kcd = 1;
      double pq = p*q;
      C *= Kab*Kcd;
      C /= sqrt(pq*pq*(p+q));

      for (int m = 0; m < N; ++m) {
        Vk[m] = C*Fm[m];
      }

      one_over_2p[k] = 0.5/p;
      alpha_over_p[k] = alpha/p;
      for (int i = 0; i < 3; ++i) {
        Xpa[i+k*3] = P[i]-rA[i];
        Xpq[i+k*3] = alpha_over_p[k]*(P[i]-rX[i]);
        //printf("Xpa=%f, Xpq=%f\n", Xpa[i+3*k], Xpq[i+3*k]);
      }

      vrr1::vrr1<AB::L>(Xpa+k*3, Xpq+k*3, one_over_2p[k], alpha_over_p[k], Vk);

      if (!X::L) continue;

      alpha_over_2pq[k] = alpha/(2*pq);
      for (int i = 0; i < 3; ++i) {
        X2[i + 3*k] = (alpha/q)*(P[i]-rX[i]);
        //printf("Xpa=%f, Xpq=%f\n", Xpa[i+3*k], Xpq[i+3*k]);
      }

    }

    vrr2::vrr2<AB::L>(
      A.L, B.L, std::integral_constant<int,X::L>{},
      K, X2, alpha_over_2pq, V1, V2
    );

    stack.reset(ncart(X::L)*NAB); // V2

    constexpr int NX = npure(X::L);

    // (A+B,ncart(X)) -> (pure(X),A+B)
    if (X::L) {
      double *tmp = stack.push(NX*NAB);
      cartesian_to_pure(std::integral_constant<int,X::L>{}, NAB, V2, tmp);
      simd::transpose(NAB, NX, tmp, V2);
    }

    stack.reset();
    stack.push(
      max(
         NX*NAB, // VRR2
         NX*(NAB-ncart(AB.L))*ncart(B.L-1), // max HRR1
         NX*nbf(B)*ncart(A) // max cartesian_to_pure
      )
    );

    if (!B.L && !A.pure) {
      std::copy(V2, V2+NX*ncart(A), buffer);
      return;
    }

    double *ABX = nullptr;

    if (B.L) {
      int NA = ncart(A);
      auto rAB = rA-rB;
      double *tmp = stack.push(NX*(NAB-ncart(AB.L))*ncart(B.L-1));
      double *BAX = V2;
      auto swap = hrr::hrr1<AB::L, NX>(A.L, B.L, &rAB[0], BAX, tmp);
      if (swap) std::swap(BAX,tmp);
      if (B.pure) {
        // (B',A,X) -> (B,A,X)
        cartesian_to_pure(B.L, NX*NA, BAX, tmp);
        std::swap(BAX,tmp);
      }
      ABX = tmp;
      // (B,A,X) -> (A,B,X)
      simd::transpose<NX>(NA, libintx::nbf(B), BAX, (A.pure ? ABX : buffer));
    }
    else {
      ABX = V2;
    }

    if (A.pure) {
      // (A',B,X) -> (A,B,X)
      cartesian_to_pure(A.L, NX*libintx::nbf(B), ABX, buffer);
    }

  }

};

}

#endif /* LIBINTX_ENGINE_OS_OS3_H */
