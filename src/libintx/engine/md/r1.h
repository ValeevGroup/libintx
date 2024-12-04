#ifndef LIBINTX_MD_R1_H
#define LIBINTX_MD_R1_H

#include "libintx/orbital.h"
#include <utility>
#include <tuple>

namespace libintx::md::r1 {

enum Order { DepthFirst = 1, BreadthFirst=2 };

using libintx::Orbital;

template<int X, int Y, int Z, int M, typename T = double>
struct R {
  static constexpr Orbital orbital = {
    uint8_t(X),
    uint8_t(Y),
    uint8_t(Z)
  };
  static constexpr int L = X+Y+Z;
  static constexpr int index = hermite::index2(orbital);
  T value;
};

template<int P, int Q>
struct visitor {

  template<typename F, class R>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr static void apply1(F &&f, const R &r) {
    //printf("%i,%i,%i\n", P, Q, r.L);
    apply<0>(f,r);
    apply<2>(f,r);
    apply<4>(f,r);
    apply<6>(f,r);
  }

  template<int K, typename F, class R>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr static void apply(F &&f, const R &r) {
    constexpr bool valid = (
      K <= Q &&
      Q-K <= R::L &&
      R::L-(Q-K) <= P
    );
    if constexpr (valid) apply<Q-K,0>(f,r);
  }

  template<int QK, int Idx, typename F, class R>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr static void apply(F &&f, const R &r) {
    constexpr auto q = cartesian::orbital<QK,Idx>();
    if constexpr (q <= R::orbital) {
      constexpr auto p = R::orbital-q;
      static_assert(p.L() <= P);
      static_assert(q.L() <= Q);
      constexpr int phase = (q.L()%2 ? -1 : +1);
      f(p, q, phase*r.value);
    }
    if constexpr (Idx+1 < ncart(QK)) apply<QK,Idx+1>(f,r);
  }

};


template<int Axis, int I, int J, int K, int M, typename ... Rs>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
auto r1_plus_axis(const auto& PQ, R<I,J,K,M> r1, std::tuple<Rs...> rs) {
  constexpr int X = (Axis == 0);
  constexpr int Y = (Axis == 1);
  constexpr int Z = (Axis == 2);
  constexpr int C = (
    (Axis == 0 ? I : 1)*
    (Axis == 1 ? J : 1)*
    (Axis == 2 ? K : 1)
  );
  R<I+X,J+Y,K+Z,M-1> r;
  r.value = PQ[Axis]*r1.value;
  if constexpr (C) {
    using R2 = R<I-X,J-Y,K-Z,M>;
    r.value += C*std::get<const R2&>(rs).value;
  }
  return r;
}

template<Order Order, int I, int J, int K, class F, typename ... R2, typename ... R1>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
void visit(F f, const auto& PQ,
           std::tuple<R2...> r2,
           const R<I,J,K,0> &r1_0,
           const R1& ... r1)
{
  static_assert(Order == Order::BreadthFirst || Order == Order::DepthFirst);
  if constexpr (Order == Order::BreadthFirst) {
    f(r1_0);
  }
  if constexpr (sizeof...(r1)) {
    {
      visit<Order,I,J,K+1>(f, PQ, std::tie(r1...), r1_plus_axis<2>(PQ, r1, r2)...);
    }
    if constexpr (!K) {
      visit<Order,I,J+1,K>(f, PQ, std::tie(r1...), r1_plus_axis<1>(PQ, r1, r2)...);
    }
    if constexpr (!J && !K) {
      visit<Order,I+1,J,K>(f, PQ, std::tie(r1...), r1_plus_axis<0>(PQ, r1, r2)...);
    }
  }
  if constexpr (Order == Order::DepthFirst) {
    f(r1_0);
  }
}

template<Order Order, class F, int N, std::size_t ... Idx>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
void visit(F f, const auto& PQ, const auto (&s)[N], std::index_sequence<Idx...>) {
  visit<Order,0,0,0>(f, PQ, std::tuple<>{}, R<0,0,0,Idx>{ s[Idx] }...);
}

template<int L, Order Order, int N, class F>
LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
void visit(F f, const auto& PQ, const auto (&s)[N]) {
  visit<Order>(f, PQ, s, std::make_index_sequence<L+1>());
}

template<int L, Order Order = Order::BreadthFirst>
LIBINTX_GPU_DEVICE LIBINTX_GPU_FORCEINLINE
void compute(const auto &PQ, const auto &s, auto* __restrict__ r1) {
  auto f = [&](auto r) constexpr {
    if constexpr (L >= r.L) r1[r.index] = r.value;
  };
  visit<L,Order>(f, PQ, s);
}

template<int L, Order Order = r1::Order::BreadthFirst>
LIBINTX_GPU_DEVICE LIBINTX_GPU_FORCEINLINE
void compute1(const auto &PQ, const auto &s, auto* __restrict__ r1) {
  auto f = [&](auto r) constexpr {
    if constexpr (L == r.L) r1[r.index] = r.value;
  };
  visit<L,Order>(f, PQ, s);
}

}

#endif /* LIBINTX_MD_R1_H */
