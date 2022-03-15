#undef NDEBUG

#include <vector>
#include <array>
#include <cassert>
#include <math.h>

template<int N = 1>
constexpr inline double factorial(int n) {
  int64_t f = 1;
  while (n > 1) { f *= n; n -= N; }
  return f;
}

constexpr inline double binomial(int n, int k) {
  //assert(n >= k);
  return factorial(n)/(factorial(k)*factorial(n-k));
}

constexpr inline auto parity(int i) {
  return (i%2 ? -1 : 1);
};

//
// Computes coefficient of a cartesian Gaussian in a real solid harmonic Gaussian
// See IJQC 54, 83 (1995), eqn (15).
// If m is negative, imaginary part is computed, whereas a positive m indicates
// that the real part of spherical harmonic Ylm is requested.
// copied from libint2
constexpr inline double coefficient(int l, int m, int lx, int ly, int lz) {

  auto abs_m = std::abs(m);
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
    (factorial(2*lx)*factorial(2*ly)*factorial(2*lz))/factorial(2*l) *
    (factorial(l-abs_m)/factorial(l)) *
    (1.0)/factorial(l+abs_m) *
    (1.0)/(factorial(lx)*factorial(ly)*factorial(lz))
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
    pfac1 *= (double(parity(i)*factorial(2*(l-i)))/factorial(l-abs_m-2*i));
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
    (factorial<2>(2*l-1))/
    (factorial<2>(2*lx-1)*factorial<2>(2*ly-1)*factorial<2>(2*lz-1))
  );

  double result = (m == 0) ? pfac*sum : M_SQRT2*pfac*sum;
  return result;

}

inline auto cartesian_list(int L) {
  std::vector< std::array<int,3> > s;
  for (int x = L; x >= 0; --x) {
    for (int y = L-x; y >= 0; --y) {
      int z = (L-(x+y));
      assert((x+y+z) == L);
      s.push_back({x,y,z});
    }
  }
  return s;
}

inline auto spherical_list(int L) {
  std::vector<int> Js;
  for (int l = -L; l <= L; ++l) {
    Js.push_back(l);
  }
  return Js;
}

inline void table(int L, std::vector<int> spherical_list) {
  printf("// L=%i\n", L);
  for (int j : spherical_list) {
    assert(j <= L);
    for (auto f : cartesian_list(L)) {
      double c = coefficient(L, j, f[0], f[1], f[2]);
      if (c == 0) continue;
      printf(
        "{%i, %i, %i, %i, %i, %.18f},\n",
        L, j, f[0], f[1], f[2], c
      );
    }
  }
  printf("\n");
};

int main(int argc, char **argv) {
  assert(argc == 2);
  int L = atoi(argv[1]);
  for (int i = 0; i <= L; ++i) {
    table(i, spherical_list(i));
  }
}
