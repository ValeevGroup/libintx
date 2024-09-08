#ifndef LIBINTX_PURE_H
#define LIBINTX_PURE_H

#include "libintx/math.h"
#include "libintx/orbital.h"

namespace libintx::pure {

//
// Computes coefficient of a cartesian Gaussian in a real solid harmonic Gaussian
// See IJQC 54, 83 (1995), eqn (15).
// If m is negative, imaginary part is computed, whereas a positive m indicates
// that the real part of spherical harmonic Ylm is requested.
// copied from libint2
constexpr inline double coefficient(int l, int m, int lx, int ly, int lz) {

  using namespace libintx::math;
  //auto sqrt = [](auto x) { return x; };//&math::root<2>;
  auto sqrt = &math::root<2>;
  auto factorial1 = &math::factorial<1,double>;
  auto factorial2 = &math::factorial<2,double>;

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

}

#endif /* LIBINTX_PURE_H */
