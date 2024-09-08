#ifndef LIBINTX_MD_REFERENCE_H
#define LIBINTX_MD_REFERENCE_H

#include "libintx/boys/reference.h"

#include <utility>
#include <tuple>
#include <cmath>

namespace libintx::md::reference {

  const double sqrt_4_pi5 = std::sqrt(4*std::pow(M_PI,5));

  using double3 = double[3];
  using libintx::Orbital;

  constexpr auto orbital(int idx) {
    constexpr auto orbitals = hermite::orbitals2<max(XMAX,2*LMAX)+2*LMAX>;
    assert(idx < orbitals.size());
    return orbitals[idx];
  }

  LIBINTX_GPU_ENABLED
  inline double E(int i, int j, int k, double a, double b, double R) {
    auto p = a + b;
    assert(p);
    auto q = ((a ? a : 1)*(b ? b : 1))/p;
    assert(q);
    if (!a || !b) R = 0;
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

  double R(int t, int u, int v, int n, const auto &s, const double3 &r) {
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

  void compute_r1(int L, double alpha, const auto &PQ, auto &r1) {

    std::vector<double> s(L+1);

    for (int m = 0; m <= L; ++m) {
      double Fm = boys::Reference().compute(alpha*norm(PQ), m);
      //printf("md.reference F[%i]=%f\n", m, Fm);
      s[m] = Fm*pow(-2*alpha,m);
    }

    for (int i = 0; i < nherm2(L); ++i) {
      auto [x,y,z] = reference::orbital(i);
      r1[i] = md::reference::R(x, y, z, 0, s.data(), PQ.data);
    }

  }

  void compute_p_q(int Bra, int Ket, const auto &r1, auto &pq) {
    for (int ip = 0; ip < nherm2(Bra); ++ip) {
      for (int iq = 0; iq < nherm2(Ket); ++iq) {
        auto p = reference::orbital(ip);
        auto q = reference::orbital(iq);
        pq(ip,iq) = r1[hermite::index2(p+q)];
      }
    }
  }

  void compute_p_cd(const auto &Ak, const auto &Bk, const auto &C, const auto &D, auto &pCD) {

    auto &[A,ka] = Ak;
    auto &[B,kb] = Bk;

    auto prim = [](const auto &g, int k) {
      return g.prims[k];
    };

    int Bra = A.L + B.L;
    int Ket = C.L + D.L;

    auto AB = center(A) - center(B);
    auto CD = center(C) - center(D);

    for (int kd = 0; kd < D.K; ++kd) {
      for (int kc = 0; kc < C.K; ++kc) {

        std::vector<double> r1(nherm2(Bra+Ket));
        {
          double a = exp(A,ka);
          double b = exp(B,kb);
          double c = exp(C,kc);
          double d = exp(D,kd);
          auto P = center_of_charge(a, center(A), b, center(B));
          auto Q = center_of_charge(c, center(C), d, center(D));
          double p = a+b;
          double q = c+d;
          double pq = p*q;
          double K = 1;
          K *= 1/sqrt(pq*pq*(p+q));
          K *= prim(A,ka).C*prim(B,kb).C;
          K *= prim(C,kc).C*prim(D,kd).C;
          K *= std::exp(-(a*b)/p*norm(AB));
          K *= std::exp(-(c*d)/q*norm(CD));
          //K *= 2*std::pow(M_PI,2.5);
          double alpha = (p*q)/(p+q);
          compute_r1(Bra+Ket, alpha, P-Q, r1);
          for (size_t i = 0; i < r1.size(); ++i) {
            r1[i] *= K;
            //printf("ref: r[%i]=%f\n", i, r1[i]);
          }
        }

        for (auto d : orbitals(D.L)) {
          for (auto c : orbitals(C.L)) {
            for (int iq = 0; iq < nherm2(Ket); ++iq) {
              auto q =  reference::orbital(iq);
              double e = 1;
              for (int i = 0; i < 3; ++i) {
                e *= E(c[i], d[i], q[i], exp(C,kc), exp(D,kd), CD[i]);
              }
              for (int ip = 0; ip < nherm2(Bra); ++ip) {
                auto p = reference::orbital(ip);
                auto q = reference::orbital(iq);
                double phase = (q.L()%2 == 0 ? +1 : -1);
                double r = r1[hermite::index2(p+q)];
                //printf("ref: r=%f\n", K*r);
                pCD(ip,index(c),index(d)) += e*r*phase;
              }
            }
          } // c
        } // d

      } // kc
    } // kd

  }

  void compute(const auto &A, const auto &B, const auto &C, const auto &D, auto &ABCD) {

    auto prim = [](const auto &g, int k) {
      return g.prims[k];
    };

    int Bra = A.L + B.L;
    int Ket = C.L + D.L;

    auto AB = center(A) - center(B);
    auto CD = center(C) - center(D);

    for (int kd = 0; kd < D.K; ++kd) {
      for (int kc = 0; kc < C.K; ++kc) {
        for (int kb = 0; kb < B.K; ++kb) {
          for (int ka = 0; ka < A.K; ++ka) {

            std::vector<double> r1(nherm2(Bra+Ket));
            {
              double a = exp(A,ka);
              double b = exp(B,kb);
              double c = exp(C,kc);
              double d = exp(D,kd);
              auto P = center_of_charge(a, center(A), b, center(B));
              auto Q = center_of_charge(c, center(C), d, center(D));
              double p = a+b;
              double q = c+d;
              double pq = p*q;
              double K = 1;
              K *= 1/sqrt(pq*pq*(p+q));
              K *= prim(A,ka).C*prim(B,kb).C;
              K *= prim(C,kc).C*prim(D,kd).C;
              K *= std::exp(-(a*b)/p*norm(AB));
              K *= std::exp(-(c*d)/q*norm(CD));
              //K *= 2*std::pow(M_PI,2.5);
              double alpha = (p*q)/(p+q);
              compute_r1(Bra+Ket, alpha, P-Q, r1);
              for (size_t i = 0; i < r1.size(); ++i) {
                r1[i] *= K;
                //printf("ref: r[%i]=%f\n", i, r1[i]);
              }
            }

            for (auto d : orbitals(D)) {
              for (auto c : orbitals(C)) {
                for (auto b : orbitals(B)) {
                  for (auto a : orbitals(A)) {
                    double v = 0;
                    for (int iq = 0; iq < nherm2(Ket); ++iq) {
                      auto q =  reference::orbital(iq);
                      double Eq = 1;
                      for (int ix = 0; ix < 3; ++ix) {
                        Eq *= E(c[ix], d[ix], q[ix], exp(C,kc), exp(D,kd), CD[ix]);
                      }
                      double phase = (q.L()%2 == 0 ? +1 : -1);
                      for (int ip = 0; ip < nherm2(Bra); ++ip) {
                        auto p = reference::orbital(ip);
                        double Ep = 1;
                        for (int ix = 0; ix < 3; ++ix) {
                          Ep *= E(a[ix], b[ix], p[ix], exp(A,ka), exp(B,kb), AB[ix]);
                        }
                        double r = r1[hermite::index2(p+q)];
                        v += Ep*Eq*r*phase;
                      } // p
                    } // q
                    ABCD(index(a),index(b),index(c),index(d)) += sqrt_4_pi5*v;
                  } // a
                } // b
              } // c
            } // d

          } // ka
        }// kb
      } // kc
    } // kd

  }

}

#endif /* LIBINTX_MD_REFERENCE_H */
