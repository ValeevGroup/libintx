#ifndef LIBINTX_MD_REFERENCE_H
#define LIBINTX_MD_REFERENCE_H

#include <utility>
#include <tuple>

namespace libintx::md::reference {

  using double3 = double[3];
  using libintx::Orbital;

  LIBINTX_GPU_ENABLED
  inline double E(int i, int j, int k, double a, double b, double R) {
    auto p = a + b;
    assert(p);
    auto q = ((a ? a : 1)*(b ? b : 1))/p;
    assert(q);
    if ((k < 0) or (k > (i + j))) return 0;
    if (i == 0 && j == 0 && k == 0) {
      //printf("R=%f E(0)=%f\n", R, std::exp(-q*R*R));
      return 1;//std::exp(-q*R*R); // K_AB
    }
    if (i) {
      // decrement index i
      assert(i && a);
      return (
        (1/(2*p))*E(i-1,j,k-1,a,b,R) -
        (q*R/a)*E(i-1,j,k,a,b,R)    +
        (k+1)*E(i-1,j,k+1,a,b,R)
      );
    }
    else {
      // decrement index j
      assert(j && b);
      return (
        (1/(2*p))*E(i,j-1,k-1,a,b,R) +
        (q*R/b)*E(i,j-1,k,a,b,R) +
        (k+1)*E(i,j-1,k+1,a,b,R)
      );
    }
  }

  template<size_t N>
  double R(int t, int u, int v, int n, double (&s)[N], double3 r) {
    if (!t && !u && !v) return s[n];
    if (!t && !u) {
      double value = 0.0;
      if (v > 1)
        value += (v-1)*R(t,u,v-2,n+1,s,r);
      value += r[2]*R(t,u,v-1,n+1,s,r);
      return value;
    }
    if (!t) {
      double value = 0.0;
      if (u > 1)
        value += (u-1)*R(t,u-2,v,n+1,s,r);
      value += r[1]*R(t,u-1,v,n+1,s,r);
      return value;
    }
    {
      double value = 0.0;
      if (t > 1)
        value += (t-1)*R(t-2,u,v,n+1,s,r);
      value += r[0]*R(t-1,u,v,n+1,s,r);
      return value;
    }
  }

}

#endif /* LIBINTX_MD_REFERENCE_H */
