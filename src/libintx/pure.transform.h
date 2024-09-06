#ifndef LIBINTX_PURE_TRANSFORM_H
#define LIBINTX_PURE_TRANSFORM_H

#include "libintx/pure.h"
#include "libintx/utility.h"
#include <tuple>

namespace libintx::pure {

  template<int ...>
  struct Transform;

  template<int _L>
  struct Transform<_L> {
    constexpr static std::integral_constant<int,_L> L = {};
    constexpr Transform() {
      constexpr auto p = pure::orbitals<L>();
      constexpr auto c = cartesian::orbitals<L>();
      for (int ipure = 0; ipure < npure(L); ++ipure) {
        for (int icart = 0; icart < ncart(L); ++icart) {
          this->data[icart][ipure] = pure::coefficient(p[ipure], c[icart]);
        }
      }
    }
    double data[ncart(L)][npure(L)] = {};
  public:
    LIBINTX_GPU_ENABLED
    double cartesian_to_pure(int ipure, auto *V) const {
      assert(ipure < npure(L));
      double v = 0;
      for (int i = 0; i < ncart(L); ++i) {
        v += V[i]*data[i][ipure];
      }
      return v;
    }
    LIBINTX_GPU_ENABLED
    double pure_to_cartesian(int icart, auto *V) const {
      assert(icart < ncart(L));
      double v = 0;
      for (int i = 0; i < npure(L); ++i) {
        v += V[i]*data[icart][i];
      }
      return v;
    }
    LIBINTX_GPU_ENABLED
    constexpr void apply(auto &&F, int l) const {
      assert(L == l);
      F(*this);
    }

  public:

    LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
    static void cartesian_to_pure(auto &&S, auto &&T) {
      constexpr auto p = pure::orbitals<L>();
      constexpr auto c = cartesian::orbitals<L>();
      constexpr auto p_c = Transform<L>();
      foreach(
        std::make_index_sequence<p.size()>(),
        [&](auto ip) {
          double v = 0;
          foreach(
            std::make_index_sequence<c.size()>(),
            [&](auto ic) {
              constexpr double coeff = p_c.data[ic.value][ip.value];
              if constexpr (coeff) {
                v += coeff*S(c[ic]);
              }
            }
          );
          T(p[ip],v);
        }
      );
    }

    LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
    static void pure_to_cartesian(auto &&S, auto &&T) {
      constexpr auto p = pure::orbitals<L>();
      constexpr auto c = cartesian::orbitals<L>();
      constexpr auto p_c = Transform<L>();
      foreach(
        std::make_index_sequence<c.size()>(),
        [&](auto ic) {
          double v = 0;
          foreach(
            std::make_index_sequence<p.size()>(),
            [&](auto ip) {
              constexpr double coeff = p_c.data[ic.value][ip.value];
              if constexpr (coeff) {
                v += coeff*S(p[ip]);
              }
            }
          );
          T(c[ic],v);
        }
      );
    }

  };


  template<int ... Ls>
  struct Transform : Transform<Ls>... {
    constexpr Transform() = default;
    constexpr void apply(auto &&F, int l) const {
      jump_table(
        std::index_sequence<Ls...>{},
        l,
        [&](auto L) {
          const auto *transform = static_cast<const Transform<L.value>*>(this);
          F(*transform);
        }
      );
    }
  };

  template<std::size_t ... Ls>
  constexpr auto make_transform(std::index_sequence<Ls...>) {
    return Transform<Ls...>{};
  }

  template<int L>
  LIBINTX_GPU_ENABLED LIBINTX_GPU_FORCEINLINE
  void cartesian_to_pure(auto &&C, auto &&P) {
    Transform<L>::cartesian_to_pure(C,P);
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
        for (auto q : cartesian::orbitals(B)) {
          for (auto p : cartesian::orbitals(A)) {
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
            for (auto s : cartesian::orbitals(D)) {
              for (auto r : cartesian::orbitals(C)) {
                auto cr = coefficient(c,r);
                auto ds = coefficient(d,s);
                for (auto q : cartesian::orbitals(B)) {
                  for (auto p : cartesian::orbitals(A)) {
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
