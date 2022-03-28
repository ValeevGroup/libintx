#ifndef LIBINTX_MD_MD2_H
#define LIBINTX_MD_MD2_H

#include "libintx/array.h"
#include "libintx/tuple.h"
#include "libintx/shell.h"
#include "libintx/spherical.h"

#include <utility>
#include <functional>
#include <memory>
#include <type_traits>
#include <cassert>
#include <stdio.h>


namespace libintx {
namespace md {

using shell::cart;
using shell::cartsum;
using shell::spher;
using Vector3 = array<double,3>;


struct MD3 {

  virtual ~MD3() = default;

  virtual void compute(const Double<3> &rA, const Double<3> &rB, const Double<3> &rX, double*) const = 0;

  template<int AB, int X, class Boys>
  struct Kernel;

  template<int AB, int X, class Boys, class R = Kernel<AB,X,Boys> >
  static std::unique_ptr<R> kernel(const Gaussian &a, const Gaussian &b, const Gaussian &x, const Boys &boys) {
    return std::make_unique< Kernel<AB,X,Boys> >(a,b,x,boys);
  }

  template<class Boys, size_t ... ABs, size_t ... Xs>
  static auto kernel(
    const Gaussian &A, const Gaussian &B, const Gaussian &X, const Boys &boys,
    std::index_sequence<ABs...>, std::index_sequence<Xs...>)
  {
    using R = MD3;
    using Factory = std::function< std::unique_ptr<R>(const Gaussian&, const Gaussian&, const Gaussian&, const Boys&) >;
    static auto kernels = make_array<Factory>(
      [](auto AB, auto X) {
        return Factory(kernel<AB.value,X.value,Boys,R>);
      },
      std::index_sequence<ABs...>{},
      std::index_sequence<Xs...>{}
    );
    int AB = A.L + B.L;
    if ((size_t)AB < sizeof...(ABs) && (size_t)X.L < sizeof...(Xs)) {
      return kernels[AB][X.L](A,B,X,boys);
    }
    return std::unique_ptr<R>();
  }

};

template<int X, int Y, int Z, int N>
void R(const double (&s)[N]) {

}

template<int _AB, int _X, class Boys>
struct MD3::Kernel : MD3 {

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

  int nbf() const {
    return cart(A)*cart(B)*cart(X::L);
  }

  Kernel(const Gaussian &A, const Gaussian &B, const Gaussian &X, const Boys &boys)
    : A(A), B(B), AB{A.K*B.K}, X(X), boys(boys)
  {
    int K = AB.K;
    AB.prims.resize(K);
    for (int j = 0; j < B.K; ++j) {
      for (int i = 0; i < A.K; ++i) {
        int k = i + j*A.K;
        AB.prims[k].a = A.prims[i].a;
        AB.prims[k].b = B.prims[j].a;
        double C = A.prims[i].C*B.prims[j].C;
        AB.prims[k].C = C*2*pow(M_PI,2.5);
        //printf("AB[%i] = %f,%f,%f\n", k, AB.prims[k].a, AB.prims[k].b, AB.prims[k].C);
      }
    }
    const int NAB = cartsum(AB.L)-cartsum(A.L-1);
    using std::max;
    int size = 0;
    int hrr1 = (cart(A.L)*cart(B.L)*spher(X.L) + NAB*cart(B.L-1)*spher(X.L));
    size = max(size, K*(cartsum(AB::L) + 12)); // VRR1
    size = max(size, NAB*spher(X::L)); // VRR2 spherical transform
    size = max(size, hrr1); // HRR1
    size += NAB*cart(X::L); // VRR2
    //this->stack = Stack(size);
  }

  void compute(const Double<3> &rA, const Double<3> &rB, const Double<3> &rX, double *buffer) const override {

    const int NAB = cartsum(AB.L)-cartsum(A.L-1);
    const int K = AB.K;

    for (int k = 0; k < K; ++k) {

      double *Vk = V1 + k*cartsum(AB::L);

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

      double Kab = exp(-(a*b)/(a+b)*norm(rA,rB));
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

    stack.reset(cart(X::L)*NAB); // V2

    constexpr int NX = spher(X::L);

    // (A+B,cart(X)) -> (pure(X),A+B)
    if (X::L) {
      double *tmp = stack.push(NX*NAB);
      spherical::transform(NAB, std::integral_constant<int,X::L>{}, V2, tmp);
      simd::transpose(NAB, NX, tmp, V2);
    }

    stack.reset();
    stack.push(
      std::max(
        NX*NAB, // VRR2
        NX*(NAB-cart(AB.L))*cart(B.L-1) // max HRR1
      )
    );

    if (!B.L && !A.pure) {
      std::copy(V2, V2+NX*cart(A), buffer);
    }

    double *ABX = nullptr;

    if (B.L) {
      int NA = cart(A);
      auto rAB = rA-rB;
      double *tmp = stack.push(NX*(NAB-cart(AB.L))*cart(B.L-1));
      double *BAX = V2;
      auto swap = hrr::hrr1<AB::L, NX>(A.L, B.L, &rAB[0], BAX, tmp);
      if (swap) std::swap(BAX,tmp);
      if (B.pure) {
        spherical::transform(NX*NA, B.L, BAX, tmp);
        std::swap(BAX,tmp);
      }
      ABX = tmp;
      simd::transpose<NX>(NA, shell::nbf(B), BAX, (A.pure ? ABX : buffer));
    }
    else {
      ABX = V2;
    }

    if (A.pure) {
      spherical::transform(NX*shell::nbf(B), A.L, ABX, buffer);
    }

  }

};

}
}

#endif /* LIBINTX_MD_MD2_H */
