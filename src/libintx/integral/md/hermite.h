#ifndef LIBINTX_MD_HERMITE_H
#define LIBINTX_MD_HERMITE_H

#include "libintx/orbital.h"
#include "libintx/utility.h"
#include "libintx/pure.transform.h"

namespace libintx::md {

  template<int L>
  struct E2 {

    double data[3][L+1][L+1][2*L+1] = {};

    auto operator()(int i, int j, int k, int axis) const {
      return data[axis][i][j][k];
    }

    template<typename Orbital>
    auto operator()(Orbital A, Orbital B, Orbital P) const {
      double e = 1;
      for (int i = 0; i < 3; ++i) {
        e *= data[i][A[i]][B[i]][P[i]];
      }
      return e;
    }

    E2(int A, int B, double a, double b, const double *R) {
      assert(A <= L);
      assert(B <= L);
      auto p = a + b;
      assert(p);
      auto q = ((a ? a : 1)*(b ? b : 1))/p;
      assert(q);
      for (int x = 0; x < 3; ++x) {
        auto &E = data[x];
        E[0][0][0] = 1;
        for (int i = 1; i <= L; ++i) {
          if (i > A) break;
          // k = 0
          E[i][0][0] = 0.0 - (q*R[x]/a)*E[i-1][0][0] + E[i-1][0][1];
          for (int k = 1; k < i; ++k) {
            double e = (
              (1/(2*p))*E[i-1][0][k-1] -
              (q*R[x]/a)*E[i-1][0][k] +
              (k+1)*E[i-1][0][k+1]
            );
            E[i][0][k] = e;
          }
          // k = i
          E[i][0][i] = (1/(2*p))*E[i-1][0][i-1] - 0.0 + 0.0;
        }
        // j
        for (int j = 1; j <= L; ++j) {
          if (j > B) break;
          for (int i = 0; i <= L; ++i) {
            if (i > A) break;
            E[i][j][0] = 0.0 + (q*R[x]/b)*E[i][j-1][0] + E[i][j-1][1];
            for (int k = 1; k < i+j; ++k) {
              double e = (
                (1/(2*p))*E[i][j-1][k-1] +
                (q*R[x]/b)*E[i][j-1][k] +
                (k+1)*E[i][j-1][k+1]
              );
              E[i][j][k] = e;
            }
            E[i][j][i+j] = (1/(2*p))*E[i][j-1][i+j-1] + 0.0 + 0.0;
          }
        }
      }
    }
  };

  template<int Ax, int Ay, int Az, int Px = 0, int Py = 0, int Pz = 0>
  LIBINTX_GPU_ENABLED
  double hermite_to_cartesian(auto &h, double inv_2p) {
    constexpr int Dx = (Ax != 0);
    constexpr int Dy = (Ay != 0 && !Dx);
    constexpr int Dz = (Az != 0 && !Dx && !Dy);
    if constexpr (Dx || Dy || Dz) {
      constexpr int ip = (Dx ? Px : (Dy ? Py : (Dz ? Pz : 0)));
      double t = 0;
      t += inv_2p*hermite_to_cartesian<Ax-Dx,Ay-Dy,Az-Dz,Px+Dx,Py+Dy,Pz+Dz>(h, inv_2p);
      if constexpr (ip) {
        t += ip*hermite_to_cartesian<Ax-Dx,Ay-Dy,Az-Dz,Px-Dx,Py-Dy,Pz-Dz>(h, inv_2p);
      }
      return t;
    }
    else {
      return h(
        std::integral_constant<int,Px>(),
        std::integral_constant<int,Py>(),
        std::integral_constant<int,Pz>()
      );
    }
  }

  template<int X>
  LIBINTX_GPU_ENABLED
  void hermite_to_cartesian(double inv_2_p, auto &&P, auto &&V) {
    using hermite::index1;
    foreach(
      std::make_index_sequence<ncart(X)>(),
      [&](auto ix) {
        constexpr auto x = std::get<ix.value>(cartesian::orbitals<X>());
        auto h = [&](auto&& ... p) {
          return P(Orbital{(uint8_t)p.value...});
        };
        auto v = hermite_to_cartesian<x[0],x[1],x[2]>(h, inv_2_p);
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
      for (int jcart = 0; jcart < b.size(); ++jcart) {
        for (int icart = 0; icart < a.size(); ++icart) {
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
    double data[ncart(A+B)][npure(B)][npure(A)] = {};
  };

  template<int A, int B>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void hermite_to_pure(auto &&S, auto &&T) {
    constexpr auto a = pure::orbitals<A>();
    constexpr auto b = pure::orbitals<B>();
    constexpr auto p = cartesian::orbitals<A+B>();
    constexpr auto ab_p = pure_transform<A,B>();
    constexpr auto ip = std::make_index_sequence<p.size()>();
    foreach2(
      std::make_index_sequence<a.size()>(),
      std::make_index_sequence<b.size()>(),
      [&](auto i, auto j) {
        double v = 0;
        foreach(
          ip,
          [&](auto ip) {
            constexpr double c = ab_p.data[ip][j][i];
            if constexpr (c) {
              v += c*S(p[ip]);
            }
          }
        );
        T(a[i],b[j],v);
      }
    );
  }

}

#endif /* LIBINTX_MD_HERMITE_H */
