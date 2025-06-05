#ifndef LIBINTX_PURE_TRANSFORM_H
#define LIBINTX_PURE_TRANSFORM_H

#include "libintx/math.h"
#include "libintx/orbital.h"
#include "libintx/utility.h"
#include <math.h>


namespace libintx::pure {

  //
  // Computes coefficient of a cartesian Gaussian in a real solid harmonic Gaussian
  // See IJQC 54, 83 (1995), eqn (15).
  // If m is negative, imaginary part is computed, whereas a positive m indicates
  // that the real part of spherical harmonic Ylm is requested.
  // copied from libint2
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  constexpr double coefficient(int l, int m, int lx, int ly, int lz) {

    using namespace libintx::math;
    //auto sqrt = [](auto x) { return x; };//&math::root<2>;
    auto factorial1 = &math::factorial<1,double>;
    auto factorial2 = &math::factorial<2,double>;

    auto sqrt = [](double y) {
      auto fabs = [](auto v) {
        return v < 0 ? -v : v;
      };
      double x = 1;
      while (fabs(x*x - y)/std::max(x*x,y) > 1e-15) {
        x = (x + y/x)/2;
      }
      //printf ("%e %e\n", y, x);
      return x;
    };

    if (l != (lx+ly+lz)) return 0;

    auto abs_m = abs(m);
    if ((lx + ly - abs_m)%2)
      return 0.0;

    auto j = (lx + ly - abs_m)/2;
    if (j < 0)
      return 0.0;

    // Checking whether the cartesian polynomial contributes to the requested component of Ylm
    auto comp = (m >= 0) ? 1 : -1;
    /*  if (comp != ((abs_m-lx)%2 ? -1 : 1))*/
    auto i = abs_m-lx;
    if (comp != parity(abs(i)))
      return 0.0;

    double pfac = sqrt(
      (factorial1(2*lx)*factorial1(2*ly)*factorial1(2*lz))/factorial1(2*l) *
      (factorial1(l-abs_m)/factorial1(l)) *
      (1.0)/factorial1(l+abs_m) *
      (1.0)/(factorial1(lx)*factorial1(ly)*factorial1(lz))
    );
    /*  pfac = sqrt(fac[l-abs_m]/(fac[l]*fac[l]*fac[l+abs_m]));*/
    pfac /= (1L << l);
    if (m < 0)
      pfac *= parity((i-1)/2);
    else
      pfac *= parity(i/2);

    auto i_min = j;
    auto i_max = (l-abs_m)/2;
    double sum = 0;
    for(auto i=i_min;i<=i_max;i++) {
      double pfac1 = binomial(l,i)*binomial(i,j);
      pfac1 *= ((parity(i)*factorial1(2*(l-i)))/factorial1(l-abs_m-2*i));
      double sum1 = 0.0;
      const int k_min = std::max((lx-abs_m)/2,0);
      const int k_max = std::min(j,lx/2);
      for(int k=k_min;k<=k_max;k++) {
        if (lx-2*k <= abs_m)
          sum1 += binomial(j,k)*binomial(abs_m,lx-2*k)*parity(k);
      }
      sum += pfac1*sum1;
    }

    sum *= sqrt(
      factorial2(2*l-1)/
      (factorial2(2*lx-1)*factorial2(2*ly-1)*factorial2(2*lz-1))
    );

    double result = (m == 0) ? pfac*sum : M_SQRT2*pfac*sum;
    return result;

  }

  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  constexpr double coefficient(const pure::Orbital &p, const cartesian::Orbital &c) {
    auto [l,m] = p;
    auto [x,y,z] = c;
    return coefficient(l, m, x, y, z);
  }

  template<int L, int M, int LX, int LY, int LZ>
  struct Coefficient {
    static constexpr double value = coefficient(L,M,LX,LY,LZ);
    constexpr operator double() const { return value; }
  };

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

    // LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
    // constexpr static auto coefficients() {
    //   constexpr size_t N = [&](){
    //     size_t nnzero = 0;
    //     constexpr Transform transform;
    //     for (size_t i = 0; i < npure(L); ++i) {
    //       for (size_t j = 0; j < ncart(L); ++j) {
    //         if (transform.data[j][i]) ++nnzero;
    //       }
    //     }
    //     return nnzero;
    //   }();
    //   using tuple = std::tuple<pure::Orbital, cartesian::Orbital, double>;
    //   return array<tuple,N>{};
    // }

    LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
    static void cartesian_to_pure(auto &&S, auto &&T) {
      constexpr auto p = pure::orbitals<L>();
      constexpr auto c = cartesian::orbitals<L>();
      constexpr auto p_c = Transform<L>();
      foreach(
        std::make_index_sequence<p.size()>(),
        [&](auto ip) {
           auto v = std::remove_cvref_t<decltype(S(c[0]))>(0);
          foreach(
            std::make_index_sequence<c.size()>(),
            [&](auto ic) {
              constexpr double coeff = p_c.data[ic.value][ip.value];
              constexpr auto c_ic = c[ic.value];
              if constexpr (coeff) {
                v += coeff*S(c_ic);
              }
            }
          );
          T(p[ip],v);
        }
      );
    }

    LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
    static void pure_to_cartesian(auto &&S, auto &&T) {
      constexpr auto p = pure::orbitals<L>();
      constexpr auto c = cartesian::orbitals<L>();
      constexpr auto p_c = Transform<L>();
      foreach(
        std::make_index_sequence<c.size()>(),
        [&](auto ic) {
          auto v = decltype(S(c[0]))(0);
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
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void cartesian_to_pure(auto &&S, auto &&T) {
    Transform<L>::cartesian_to_pure(S,T);
  }

  template<int L>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void pure_to_cartesian(auto &&S, auto &&T) {
    Transform<L>::pure_to_cartesian(S,T);
  }

  template<int A, int B, typename S, typename T>
  void cartesian_to_pure(const S* __restrict__ src, T* __restrict__ dst) {
    constexpr int NB = ncart(B);
    T U[NB][npure(A)];
libintx_unroll(28)
    for (int j = 0; j < NB; ++j) {
      pure::cartesian_to_pure<A>(
        [&](auto a) { return src[index(a) + j*ncart(A)]; },
        [&](auto a, auto v) { U[j][index(a)] = v; }
      );
    }
    constexpr int NA = npure(A);
libintx_unroll(13)
    for (int i = 0; i < NA; ++i) {
      pure::cartesian_to_pure<B>(
        [&](auto b) { return U[index(b)][i]; },
        [&](auto b, auto v) { dst[i+index(b)*NA] += v; }
      );
    }
  }

  template<int A, int B, typename T>
  void pure_to_cartesian(const T *src, T *dst) {
    constexpr int NB = npure(B);
    T U[NB][ncart(A)];
libintx_unroll(13)
    for (int j = 0; j < NB; ++j) {
      pure::pure_to_cartesian<A>(
        [&](auto a) { return src[index(a) + j*npure(A)]; },
        [&](auto a, auto v) { U[j][index(a)] = v; }
      );
    }
    constexpr int NA = ncart(A);
libintx_unroll(28)
    for (int i = 0; i < NA; ++i) {
      pure::pure_to_cartesian<B>(
        [&](auto b) { return U[index(b)][i]; },
        [&](auto b, auto v) { dst[i + index(b)*NA] += v; }
      );
    }
  }

}

#endif /* LIBINTX_PURE_TRANSFORM_H */
