#ifndef LIBINTX_MATH_H
#define LIBINTX_MATH_H

#include <cmath>
#include <cassert>
#include <utility>

namespace libintx::math {

  using std::pow;
  using std::sqrt;

  // from mpmath import mp; mp.dps=50; mp.sqrt(4*mp.power(mp.pi,5))
  constexpr double sqrt_4_pi5 = double{34.986836655249725692525643359743107557510223563488};
  constexpr double sqrt_pi3 = double{5.56832799683170784528481798212};

  /// df_Kminus1[k] = (k-1)!!
  static constexpr int64_t factorial2_Kminus1[31] = {
    1LL, 1LL, 1LL, 2LL, 3LL, 8LL, 15LL, 48LL, 105LL, 384LL, 945LL, 3840LL, 10395LL, 46080LL, 135135LL,
    645120LL, 2027025LL, 10321920LL, 34459425LL, 185794560LL, 654729075LL,
    3715891200LL, 13749310575LL, 81749606400LL, 316234143225LL, 1961990553600LL,
    7905853580625LL, 51011754393600LL, 213458046676875LL, 1428329123020800LL,
    6190283353629375LL
  };


  template<typename T>
  constexpr inline T abs(T v) {
    return (v < T(0) ? -v : v);
  }

  inline double ipow(double r, int n) {
    switch (n) {
    case 0: return 1;
    case 1: return r;
    case 2: return r*r;
    case 3: return r*r*r;
    case 4: return r*r*r*r;
    case 5: return r*r*r*r*r;
    case 6: return r*r*r*r*r*r;
    case 7: return r*r*r*r*r*r*r;
    case 8: return r*r*r*r*r*r*r*r;
    case 9: return r*r*r*r*r*r*r*r*r;
    }
    return pow(r,n);
  }

  template<int N, typename T>
  constexpr T pow(T x) {
    T y = 1;
    for (int i = 1; i <= N; ++i) {
      y *= x;
    }
    return y;
  }

  template<int N>
  constexpr double root(double y) {
    long double e = std::numeric_limits<double>::epsilon()*10;
    long double x = 1;
    // //return __ieee754_sqrt(v);
    while (abs(pow<N>(x) - y) > e) {
      x = ((N-1)*x + y/pow<N-1>(x))/N;
    }
    return x;
  }

  template<int N = 1, typename T = int64_t>
  constexpr T factorial(int n) {
    // /// fac[k] = k!
    // static constexpr int64_t factorial[21] = {
    //   1LL, 1LL, 2LL, 6LL, 24LL, 120LL, 720LL, 5040LL, 40320LL, 362880LL, 3628800LL, 39916800LL,
    //   479001600LL, 6227020800LL, 87178291200LL, 1307674368000LL, 20922789888000LL,
    //   355687428096000LL, 6402373705728000LL, 121645100408832000LL,
    //   2432902008176640000LL
    // };
    T f = 1;
    for (int i = n; i > 0; i -= N) {
      f *= i;
    }
    return f;
  }

  template<typename T = int64_t>
  constexpr inline T binomial(int n, int k) {
    assert(n >= k);
    return T(factorial(n)/(factorial(k)*factorial(n-k)));
  }

  constexpr inline auto parity(int i) {
    return (i%2 ? -1 : 1);
  }

  template<int K, typename T = int64_t>
  constexpr inline auto figurate(int n) {
    int64_t nk = 1;
    for (size_t k = 0; k < K; ++k) {
      nk *= (n+k);
    }
    return T(nk/factorial(K));
  };

  template<int K, std::size_t ... Idx>
  constexpr auto make_figurate_numbers(std::index_sequence<Idx...>) {
    return std::index_sequence< figurate<K>(Idx)... >{};
  }

  template<int N>
  constexpr auto triangular_numbers = make_figurate_numbers<2>(std::make_index_sequence<N>{});

  template<int N>
  constexpr auto tetrahedral_numbers = make_figurate_numbers<3>(std::make_index_sequence<N>{});

}

#endif /* LIBINTX_MATH_H */
