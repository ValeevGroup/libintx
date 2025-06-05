#ifndef LIBINTX_PURE_REFERENCE_H
#define LIBINTX_PURE_REFERENCE_H

#include <cmath>

namespace libintx::pure::reference {

  /// fac[k] = k!
  static constexpr std::array<int64_t,21> fac = {{1LL, 1LL, 2LL, 6LL, 24LL, 120LL, 720LL, 5040LL, 40320LL, 362880LL, 3628800LL, 39916800LL,
      479001600LL, 6227020800LL, 87178291200LL, 1307674368000LL, 20922789888000LL,
      355687428096000LL, 6402373705728000LL, 121645100408832000LL,
      2432902008176640000LL}};

  /// df_Kminus1[k] = (k-1)!!
  static constexpr std::array<int64_t,31> df_Kminus1 = {{1LL, 1LL, 1LL, 2LL, 3LL, 8LL, 15LL, 48LL, 105LL, 384LL, 945LL, 3840LL, 10395LL, 46080LL, 135135LL,
      645120LL, 2027025LL, 10321920LL, 34459425LL, 185794560LL, 654729075LL,
      3715891200LL, 13749310575LL, 81749606400LL, 316234143225LL, 1961990553600LL,
      7905853580625LL, 51011754393600LL, 213458046676875LL, 1428329123020800LL,
      6190283353629375LL}};

  /// bc(i,j) = binomial coefficient, i! / (j! (i-j)!)
  template <typename Int> int64_t bc(Int i, Int j) {
    assert(i < Int(fac.size()));
    assert(j < Int(fac.size()));
    assert(i >= j);
    return fac[i] / (fac[j] * fac[i-j]);
  }

  template <typename Int>
  signed char parity(Int i) {
    return i%2 ? -1 : 1;
  }

  template<typename Real = double>
  Real coefficient(int l, int m, int lx, int ly, int lz) {

    auto abs_m = std::abs(m);
    if ((lx + ly - abs_m)%2)
      return 0.0;

    auto j = (lx + ly - abs_m)/2;
    if (j < 0)
      return 0.0;

    /*----------------------------------------------------------------------------------------
      Checking whether the cartesian polynomial contributes to the requested component of Ylm
      ----------------------------------------------------------------------------------------*/
    auto comp = (m >= 0) ? 1 : -1;
    /*  if (comp != ((abs_m-lx)%2 ? -1 : 1))*/
    auto i = abs_m-lx;
    if (comp != parity(abs(i)))
      return 0.0;

    assert(l <= 10); // libint2::math::fac[] is only defined up to 20
    Real pfac = sqrt( ((Real(fac[2*lx])*Real(fac[2*ly])*Real(fac[2*lz]))/fac[2*l]) *
                      ((Real(fac[l-abs_m]))/(fac[l])) *
                      (Real(1)/fac[l+abs_m]) *
                      (Real(1)/(fac[lx]*fac[ly]*fac[lz]))
    );
    /*  pfac = sqrt(fac[l-abs_m]/(fac[l]*fac[l]*fac[l+abs_m]));*/
    pfac /= (1L << l);
    if (m < 0)
      pfac *= parity((i-1)/2);
    else
      pfac *= parity(i/2);

    auto i_min = j;
    auto i_max = (l-abs_m)/2;
    Real sum = 0;
    for(auto i=i_min;i<=i_max;i++) {
      Real pfac1 = bc(l,i)*bc(i,j);
      pfac1 *= (Real(parity(i)*fac[2*(l-i)])/fac[l-abs_m-2*i]);
      Real sum1 = 0.0;
      const int k_min = std::max((lx-abs_m)/2,0);
      const int k_max = std::min(j,lx/2);
      for(int k=k_min;k<=k_max;k++) {
        if (lx-2*k <= abs_m)
          sum1 += bc(j,k)*bc(abs_m,lx-2*k)*parity(k);
      }
      sum += pfac1*sum1;
    }
    sum *= sqrt(Real(df_Kminus1[2*l])/(df_Kminus1[2*lx]*df_Kminus1[2*ly]*df_Kminus1[2*lz]));

    Real result = (m == 0) ? pfac*sum : M_SQRT2*pfac*sum;
    return result;
  }

  template<typename Real = double>
  Real coefficient(auto pure, auto cart) {
    auto [lx,ly,lz] = cart;
    auto [l,m] = pure;
    return coefficient(l,m,lx,ly,lz);
  }

  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
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

  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
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

#endif /* LIBINTX_PURE_REFERENCE_H */
