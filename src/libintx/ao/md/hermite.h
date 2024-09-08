#ifndef LIBINTX_MD_HERMITE_H
#define LIBINTX_MD_HERMITE_H

#include "libintx/forward.h"
#include "libintx/orbital.h"
#include "libintx/utility.h"
#include "libintx/pure.transform.h"

namespace libintx::md {

  template<typename T, int A, int B, int P = A+B>
  struct E2 {

    alignas(T) T data[3][A+1][B+1][P+1];

    LIBINTX_ALWAYS_INLINE
    auto operator()(int i, int j, int k, int axis) const {
      return data[axis][i][j][k];
    }

    LIBINTX_ALWAYS_INLINE
    auto x(int i, int j) const {
      static_assert(P == 0);
      return data[0][i][j][0];
    }

    LIBINTX_ALWAYS_INLINE
    auto y(int i, int j) const {
      static_assert(P == 0);
      return data[1][i][j][0];
    }

    LIBINTX_ALWAYS_INLINE
    auto z(int i, int j) const {
      static_assert(P == 0);
      return data[2][i][j][0];
    }

    template<int Axis>
    LIBINTX_ALWAYS_INLINE
    const auto& p(Orbital a, Orbital b) const {
      return data[Axis][a[Axis]][b[Axis]];
    }

    LIBINTX_ALWAYS_INLINE
    const auto& px(Orbital a, Orbital b) const {
      return this->p<0>(a,b);
    }

    LIBINTX_ALWAYS_INLINE
    const auto& py(Orbital a, Orbital b) const {
      return this->p<1>(a,b);
    }

    LIBINTX_ALWAYS_INLINE
    const auto& pz(Orbital a, Orbital b) const {
      return this->p<2>(a,b);
    }

    template<typename Orbital>
    LIBINTX_ALWAYS_INLINE
    auto operator()(Orbital a, Orbital b, Orbital p) const {
      T e{1};
      for (int i = 0; i < 3; ++i) {
        e *= this->operator()(a[i], b[i], p[i], i);
      }
      return e;
    }

    LIBINTX_ALWAYS_INLINE
    E2(T a, T b, const array<T,3> &X) {
      constexpr bool Transpose = (B < A); // prefered init order
      auto E = [&](auto i, auto j, auto k, int x) ->auto& {
        if (Transpose) std::swap(i,j);
        return this->data[x][i][j][k];
      };
      if constexpr (Transpose) {
        init<B,A>(b,a,-X,E);
      }
      else {
        init<A,B>(a,b,X,E);
      }
      // for (int ia = 0; ia <= A; ++ia) {
      //   for (int ib = 0; ib <= B; ++ib) {
      //     for (int ip = 0; ip <= P; ++ip) {
      //       printf("E(%i,%i,%i) = %f\n", ia, ib, ip, this->data[0][ia][ib][ip]);
      //     }
      //   }
      // }
    }

  private:

    template<int First, int Second>
    LIBINTX_ALWAYS_INLINE
    static void init(T a, T b, const T* __restrict__ X, auto &&E) {
      // better performance if Second > First
      //static_assert(First <= Second);
      //static_assert(P <= First+Second);
      constexpr int L = First+Second;
      auto p = a + b;
      //assert(a && b);
      auto q = a*b/p;//((a ? a : 1)*(b ? b : 1))/p;
      //assert(q);
      T inv_2_p = 0.5/p;
libintx_unroll(3)
      for (size_t ix = 0; ix < 3; ++ix) {
        alignas(T) T qXa, qXb;
        alignas(T) T Eb[Second+1] = {};
        qXa = -(q*X[ix]/a);
        qXb = +(q*X[ix]/b);
        Eb[0] = T{1};
libintx_unroll(LIBINTX_MAX_L+3)
        for (int j = 0; j <= Second; ++j) {
          if (j != 0) init<L>(j,inv_2_p,qXb,Eb);
          alignas(T) T Eab[std::max(L+1,P+1)] = {};
libintx_unroll(100)
          for (int k = 0; k <= j; ++k) {
            if (j+k > L+P) break;
            Eab[k] = Eb[k];
          }
libintx_unroll(LIBINTX_MAX_L+3)
          for (int i = 0; i <= First; ++i) {
            if (i != 0) init<L+P>(i+j,inv_2_p,qXa,Eab);
libintx_unroll(2*LIBINTX_MAX_L+1)
            for (int k = 0; k <= P; ++k) {
              E(i,j,k,ix) = Eab[k];
              //printf("E(%i,%i,%i,%i)=%f\n", i, j, k, ix, E(i,j,k,ix));
            }
          }
        } // j
      }
    }

    template<int L, int N>
    LIBINTX_ALWAYS_INLINE
    static void init(int K, const T &inv_2_p, const T &X, T (&E)[N]) {
      if constexpr (L == 0) return;
      assert(K > 0);
      //assert(K < N);
      alignas(T) T Ep[N] = {};
      Ep[0] = X*E[0] + E[1];
libintx_unroll(2*LIBINTX_MAX_L+1)
      for (int k = 1; k < K; ++k) {
        Ep[k] = inv_2_p*E[k-1] + X*E[k];
        Ep[k] += (k+1)*E[k+1];
      }
      Ep[K] = inv_2_p*E[K-1];
libintx_unroll(2*LIBINTX_MAX_L+1)
      for (int k = 0; k <= K; ++k) {
        E[k] = Ep[k];
      }
    }

  };


//   template<int A, int B, typename T>
//   void hermite_to_cartesian(
//     const auto &ai, const auto &aj,
//     const auto &R,
//     const T &C, const T *H, T *G)
//   {
//     constexpr auto orbitals2 = hermite::orbitals2<A+B>;
//     constexpr int NP = nherm2(A+B);
//     md::E2<T,A,B,A+B> E(ai,aj,R);
//     //#pragma GCC unroll (28)
//     for (auto b : cartesian::orbitals<B>()) {
//       //#pragma GCC unroll (28)
//       for (auto a : cartesian::orbitals<A>()) {
//         int iab = index(a) + index(b)*ncart(A);
//         T g = 0;
// #pragma GCC unroll (455)
//         for (int ip = 0; ip < NP; ++ip) {
//           auto p = orbitals2[ip];
//           auto e = E(a,b,p);
//           g += e*H[ip];
//         }
//         G[iab] += C*g;
//       }
//     }
//   }


  template<typename T, int Ax, int Ay, int Az, int Px = 0, int Py = 0, int Pz = 0>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  auto hermite_to_cartesian(const auto &H, const T &inv_2p) {
    constexpr int Dx = (Ax != 0);
    constexpr int Dy = (Ay != 0 && !Dx);
    constexpr int Dz = (Az != 0 && !Dx && !Dy);
    if constexpr (Dx || Dy || Dz) {
      constexpr int ip = (Dx ? Px : (Dy ? Py : (Dz ? Pz : 0)));
      auto t = inv_2p*hermite_to_cartesian<T,Ax-Dx,Ay-Dy,Az-Dz,Px+Dx,Py+Dy,Pz+Dz>(H, inv_2p);
      if constexpr (ip) {
        auto t1 = hermite_to_cartesian<T,Ax-Dx,Ay-Dy,Az-Dz,Px-Dx,Py-Dy,Pz-Dz>(H, inv_2p);
        t += ip*t1;
      }
      return t;
    }
    else {
      //static constexpr auto p = Orbital{Px,Py,Pz};
      //return H(p);
      return H(
        std::integral_constant<uint8_t,Px>(),
        std::integral_constant<uint8_t,Py>(),
        std::integral_constant<uint8_t,Pz>()
      );
    }
  }

  template<int X, typename T>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void hermite_to_cartesian(const T &inv_2_p, auto &&H, auto &&V) {
    using hermite::index1;
    foreach(
      std::make_index_sequence<ncart(X)>(),
      [&](auto ix) {
        constexpr auto x = std::get<ix.value>(cartesian::orbitals<X>());
        auto v = hermite_to_cartesian<T,x[0],x[1],x[2]>(H, inv_2_p);
        V(x) = v;
      }
    );
  }

  template<int A, int B>
  struct pure_transform {
    constexpr pure_transform() {
      constexpr pure::Transform<A> pure_transform_a;
      constexpr pure::Transform<B> pure_transform_b;
      constexpr auto a = cartesian::orbitals<A>();
      constexpr auto b = cartesian::orbitals<B>();
      for (size_t jcart = 0; jcart < b.size(); ++jcart) {
        for (size_t icart = 0; icart < a.size(); ++icart) {
          int ip = cartesian::index(a[icart]+b[jcart]);
          for (int jpure = 0; jpure < npure(B); ++jpure) {
            for (int ipure = 0; ipure < npure(A); ++ipure) {
              auto C = (
                pure_transform_a.data[icart][ipure]*
                pure_transform_b.data[jcart][jpure]
              );
              this->data[ip][jpure][ipure] += C;
            }
          }
        }
      }
    }
    union {
      double data[ncart(A+B)][npure(B)][npure(A)] = {};
      double data2[ncart(A+B)][npure(A,B)];
    };
  };

  template<int A, int B>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void hermite_to_pure(const auto &S, const auto &T) {
    constexpr auto a = pure::orbitals<A>();
    constexpr auto b = pure::orbitals<B>();
    constexpr auto p = cartesian::orbitals<A+B>();
    constexpr auto ab_p = pure_transform<A,B>();
    constexpr auto ip = std::make_index_sequence<p.size()>();
    foreach2(
      std::make_index_sequence<a.size()>(),
      std::make_index_sequence<b.size()>(),
      [&](auto i, auto j) {
        //decltype(S(p[0])) v = {};
        decltype(S(std::integral_constant<int,0>{})) v = {};
        foreach(
          ip,
          [&](auto ip) {
            constexpr double c = ab_p.data[ip.value][j.value][i.value];
            if constexpr (c) {
              // NB static required for optimization of S(...)
              //static constexpr auto p_ip = p[ip.value];
              v += c*S(ip);
            }
          }
        );
        static constexpr auto a_i = a[i.value];
        static constexpr auto b_j = b[j.value];
        T(a_i,b_j,v);
      }
    );
  }

}

#endif /* LIBINTX_MD_HERMITE_H */
