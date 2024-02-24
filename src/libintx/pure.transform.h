#ifndef LIBINTX_PURE_TRANSFORM_H
#define LIBINTX_PURE_TRANSFORM_H

#include "libintx/pure.h"
#include <tuple>

namespace libintx::pure {

  template<int ... Args>
  struct tuple;

  template<int L, int M>
  struct tuple<L,M> {
    std::integral_constant<int16_t,L> l;
    std::integral_constant<int16_t,M> m;
    //std::tuple< std::integral_constant<int,Args> ... >
  };

  template<int X, int Y, int Z>
  struct tuple<X,Y,Z> {
    std::integral_constant<int,X> x;
    std::integral_constant<int,Y> y;
    std::integral_constant<int,Z> z;
    //std::tuple< std::integral_constant<int,Args> ... >
  };

  template<int L, int M, int LX, int LY, int LZ>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  constexpr auto coefficient(tuple<L,M>, tuple<LX,LY,LZ>) {
    return Coefficient<L,M,LX,LY,LZ>{};
  }

#define Tuple(...) tuple<__VA_ARGS__>()

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<0>, F &&f) {
    f(Tuple(0,0), Tuple(0,0,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<1>, F &&f) {
    f(Tuple(1,-1), Tuple(0,1,0));
    f(Tuple(1,0), Tuple(0,0,1));
    f(Tuple(1,1), Tuple(1,0,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<2>, F &&f) {
    f(Tuple(2,-2), Tuple(1,1,0));
    f(Tuple(2,-1), Tuple(0,1,1));
    f(Tuple(2,0), Tuple(2,0,0), Tuple(0,2,0), Tuple(0,0,2));
    f(Tuple(2,1), Tuple(1,0,1));
    f(Tuple(2,2), Tuple(2,0,0), Tuple(0,2,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<3>, F &&f) {
    f(Tuple(3,-3), Tuple(2,1,0), Tuple(0,3,0));
    f(Tuple(3,-2), Tuple(1,1,1));
    f(Tuple(3,-1), Tuple(2,1,0), Tuple(0,3,0), Tuple(0,1,2));
    f(Tuple(3,0), Tuple(2,0,1), Tuple(0,2,1), Tuple(0,0,3));
    f(Tuple(3,1), Tuple(3,0,0), Tuple(1,2,0), Tuple(1,0,2));
    f(Tuple(3,2), Tuple(2,0,1), Tuple(0,2,1));
    f(Tuple(3,3), Tuple(3,0,0), Tuple(1,2,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<4>, F &&f) {
    f(Tuple(4,-4), Tuple(3,1,0), Tuple(1,3,0));
    f(Tuple(4,-3), Tuple(2,1,1), Tuple(0,3,1));
    f(Tuple(4,-2), Tuple(3,1,0), Tuple(1,3,0), Tuple(1,1,2));
    f(Tuple(4,-1), Tuple(2,1,1), Tuple(0,3,1), Tuple(0,1,3));
    f(Tuple(4,0), Tuple(4,0,0), Tuple(2,2,0), Tuple(2,0,2), Tuple(0,4,0), Tuple(0,2,2), Tuple(0,0,4));
    f(Tuple(4,1), Tuple(3,0,1), Tuple(1,2,1), Tuple(1,0,3));
    f(Tuple(4,2), Tuple(4,0,0), Tuple(2,0,2), Tuple(0,4,0), Tuple(0,2,2));
    f(Tuple(4,3), Tuple(3,0,1), Tuple(1,2,1));
    f(Tuple(4,4), Tuple(4,0,0), Tuple(2,2,0), Tuple(0,4,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<5>, F &&f) {
    f(Tuple(5,-5), Tuple(4,1,0), Tuple(2,3,0), Tuple(0,5,0));
    f(Tuple(5,-4), Tuple(3,1,1), Tuple(1,3,1));
    f(Tuple(5,-3), Tuple(4,1,0), Tuple(2,3,0), Tuple(2,1,2), Tuple(0,5,0), Tuple(0,3,2));
    f(Tuple(5,-2), Tuple(3,1,1), Tuple(1,3,1), Tuple(1,1,3));
    f(Tuple(5,-1), Tuple(4,1,0), Tuple(2,3,0), Tuple(2,1,2), Tuple(0,5,0), Tuple(0,3,2), Tuple(0,1,4));
    f(Tuple(5,0), Tuple(4,0,1), Tuple(2,2,1), Tuple(2,0,3), Tuple(0,4,1), Tuple(0,2,3), Tuple(0,0,5));
    f(Tuple(5,1), Tuple(5,0,0), Tuple(3,2,0), Tuple(3,0,2), Tuple(1,4,0), Tuple(1,2,2), Tuple(1,0,4));
    f(Tuple(5,2), Tuple(4,0,1), Tuple(2,0,3), Tuple(0,4,1), Tuple(0,2,3));
    f(Tuple(5,3), Tuple(5,0,0), Tuple(3,2,0), Tuple(3,0,2), Tuple(1,4,0), Tuple(1,2,2));
    f(Tuple(5,4), Tuple(4,0,1), Tuple(2,2,1), Tuple(0,4,1));
    f(Tuple(5,5), Tuple(5,0,0), Tuple(3,2,0), Tuple(1,4,0));
  }

  template<typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<6>, F &&f) {
    f(Tuple(6,-6), Tuple(5,1,0), Tuple(3,3,0), Tuple(1,5,0));
    f(Tuple(6,-5), Tuple(4,1,1), Tuple(2,3,1), Tuple(0,5,1));
    f(Tuple(6,-4), Tuple(5,1,0), Tuple(3,1,2), Tuple(1,5,0), Tuple(1,3,2));
    f(Tuple(6,-3), Tuple(4,1,1), Tuple(2,3,1), Tuple(2,1,3), Tuple(0,5,1), Tuple(0,3,3));
    f(Tuple(6,-2), Tuple(5,1,0), Tuple(3,3,0), Tuple(3,1,2), Tuple(1,5,0), Tuple(1,3,2), Tuple(1,1,4));
    f(Tuple(6,-1), Tuple(4,1,1), Tuple(2,3,1), Tuple(2,1,3), Tuple(0,5,1), Tuple(0,3,3), Tuple(0,1,5));
    f(Tuple(6,0), Tuple(6,0,0), Tuple(4,2,0), Tuple(4,0,2), Tuple(2,4,0), Tuple(2,2,2), Tuple(2,0,4), Tuple(0,6,0), Tuple(0,4,2), Tuple(0,2,4), Tuple(0,0,6));
    f(Tuple(6,1), Tuple(5,0,1), Tuple(3,2,1), Tuple(3,0,3), Tuple(1,4,1), Tuple(1,2,3), Tuple(1,0,5));
    f(Tuple(6,2), Tuple(6,0,0), Tuple(4,2,0), Tuple(4,0,2), Tuple(2,4,0), Tuple(2,0,4), Tuple(0,6,0), Tuple(0,4,2), Tuple(0,2,4));
    f(Tuple(6,3), Tuple(5,0,1), Tuple(3,2,1), Tuple(3,0,3), Tuple(1,4,1), Tuple(1,2,3));
    f(Tuple(6,4), Tuple(6,0,0), Tuple(4,2,0), Tuple(4,0,2), Tuple(2,4,0), Tuple(2,2,2), Tuple(0,6,0), Tuple(0,4,2));
    f(Tuple(6,5), Tuple(5,0,1), Tuple(3,2,1), Tuple(1,4,1));
    f(Tuple(6,6), Tuple(6,0,0), Tuple(4,2,0), Tuple(2,4,0), Tuple(0,6,0));
  }

  template<size_t L, typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<L>, F &&f) {
    constexpr auto C = cartesian::shell<L>();
    constexpr auto P = pure::shell<L>();
    auto ijk = std::apply(
      [](auto ... idx) {
        return std::tuple{
          Tuple(C[idx][0], C[idx][1], C[idx][2])...
        };
      },
      integer_sequence_tuple(std::make_index_sequence<C.size()>())
    );
    foreach(
      std::make_index_sequence<P.size()>(),
      [&](auto idx) {
        std::apply(
          [&](auto ... ijk) {
            f(Tuple(P[idx.value].l,P[idx.value].m), ijk...);
          },
          ijk
        );
      }
    );
  }

#undef Tuple

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  double eval(auto f, auto cart) {
    auto [i,j,k] = cart;
    return f(cartesian::Orbital{i,j,k});
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  double eval(auto f, auto i, auto j) {
    auto [xi,yi,zi] = i;
    auto [xj,yj,zj] = j;
    return f(
      cartesian::Orbital{xi,yi,zi},
      cartesian::Orbital{xj,yj,zj}
    );
  }

  template<size_t A, size_t B>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(std::index_sequence<A,B>, auto &&f, auto &&c) {
    using Pure = pure::Orbital;
    transform(
      std::index_sequence<A>{},
      [&](auto &&a, auto&& ... is) {
        transform(
          std::index_sequence<B>{},
          [=,is=std::tuple{is...}](auto &&b, auto&& ... js) {
            auto G = [=](auto &&j) {
              double bj = coefficient(b,j);
              return std::apply(
                [=](auto ... is) {
                  return bj*((coefficient(a,is)*eval(c,is,j)) + ... );
                },
                is
              );
            };
            double v = (G(js) + ...);
            f(Pure{a.l,a.m}, Pure{b.l,b.m}, v);
          }
        );
      }
    );
  }

  template<int A, typename F>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(F &&f, auto &&C) {
    using Pure = pure::Orbital;
    auto g = [&](auto &&lm, auto &&r) {
      auto c = coefficient(lm,r);
      if constexpr (c.value) return c.value*eval(C,r);
      else return 0;
    };
    transform(
      std::index_sequence<A>{},
      [&](auto &&lm, auto&& ... r) {
        auto [l,m] = lm;
        f(Pure{l,m}, (g(lm,r) + ...));
      }
    );
  }

  template<int A, int B, typename F, typename C>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(F &&f, C &&c) {
    transform(std::index_sequence<A,B>{}, f, c);
  }

  template<int A, typename T>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(T *v) {
    double u[npure(A)];
    auto c = [&](auto &&i) { return v[index(i)]; };
    auto f = [&](auto &&i, auto &&v) { u[index(i)] = v; };
    transform(std::index_sequence<A>{}, f, c);
    //#pragma unroll
    for (int i = 0; i < npure(A); ++i) {
      v[i] = u[i];
    }
  }

}
namespace libintx::pure::reference {

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(int A, int B, const auto &Cart, auto &&Pure) {
    for (int ib = 0; ib < npure(B); ++ib) {
      for (int ia = 0; ia < npure(A); ++ia) {
        auto a = pure::orbital(A,ia);
        auto b = pure::orbital(B,ib);
        double v = 0;
        for (auto q : cartesian::shell(B)) {
          for (auto p : cartesian::shell(A)) {
            auto ap = coefficient(a,p);
            auto bq = coefficient(b,q);
            v += ap*bq*Cart(index(p), index(q));
          }
        }
        //printf("c(%i,%i)=%f\n", index(a), index(b), Cart(index(a), index(b)));
        Pure(index(a), index(b)) = v;
      }
    }
  }

  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void transform(int A, int B, int C, int D, const auto &Cart, auto &&Pure) {
    for (int id = 0; id < npure(D); ++id) {
      for (int ic = 0; ic < npure(C); ++ic) {
        for (int ib = 0; ib < npure(B); ++ib) {
          for (int ia = 0; ia < npure(A); ++ia) {
            auto a = pure::orbital(A,ia);
            auto b = pure::orbital(B,ib);
            auto c = pure::orbital(C,ic);
            auto d = pure::orbital(D,id);
            double v = 0;
            for (auto s : cartesian::shell(D)) {
              for (auto r : cartesian::shell(C)) {
                auto cr = coefficient(c,r);
                auto ds = coefficient(d,s);
                for (auto q : cartesian::shell(B)) {
                  for (auto p : cartesian::shell(A)) {
                    auto ap = coefficient(a,p);
                    auto bq = coefficient(b,q);
                    v += ap*bq*cr*ds*Cart(index(p), index(q), index(r), index(s));
                  }
                }
                //printf("c(%i,%i)=%f\n", index(a), index(b), Cart(index(a), index(b)));
              }
            }
            //printf("v(%i,%i)=%f\n", i, j, v);
            Pure(index(a), index(b), index(c), index(d)) = v;
          }
        }
      }
    }
  }

}

#endif /* LIBINTX_PURE_TRANSFORM_H */
