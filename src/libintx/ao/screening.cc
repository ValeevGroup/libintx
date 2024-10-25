#include "libintx/ao/screening.h"
#include "libintx/simd.h"
#include "libintx/utility.h"

#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/pure.transform.h"

#define LIBINTX_AO_MD_R1_COMPUTE_INLINE LIBINTX_ALWAYS_INLINE
#include "libintx/ao/md/r1.h"

#include <cmath>

namespace libintx::ao::screening {

  template<int First, int Second>
  void compute(const Gaussian& A, const Gaussian& B, double *V) {

    assert(A.L == First);
    assert(B.L == Second);

    libintx_assert(A.pure && B.pure);
    //constexpr bool pure = true;

    constexpr int L = 2*(First+Second);
    constexpr auto orbitals2 = hermite::orbitals2<First+Second>;
    constexpr int NR = orbitals2.size();
    constexpr int NA = npure(First);
    constexpr int NB = npure(Second);

    const auto &boys = boys::chebyshev<4*LMAX+1>();

    auto AB = center(A) - center(B);

    struct Hermite2 {
      double exps[2];
      double C;
      array<double,3> r;
      double E[NR][NA*NB] = {};
    };

    std::vector<Hermite2> Hs;
    Hs.reserve(A.K*B.K);

    for (int kb = 0; kb < B.K; ++kb) {
      for (int ka = 0; ka < A.K; ++ka) {

        double C = 1;
        double a = exp(A,ka);
        double b = exp(B,kb);
        double p = a+b;
        C *= A.prims[ka].C;
        C *= B.prims[kb].C;
        C *= expf(-(a*b)/p*norm(AB));
        if (C == 0) continue;
        auto r = center_of_charge<double>(a, center(A), b, center(B));
        Hermite2 h = {
          {a,b}, C,
          { r[0], r[1], r[2] }
        };

        libintx::md::E2<double,First,Second> e2(a, b, AB);

        for (auto p : orbitals2) {
          constexpr pure::Transform<First> first_transform;
          constexpr pure::Transform<Second> second_transform;
          double V[NA*NB] = {};
          for (auto b : cartesian::orbitals<Second>()) {
            double U[NA] = {};
            for (auto a : cartesian::orbitals<First>()) {
              auto e = e2(a,b,p);
              //printf("<%i,%i> H(%i,%i,%i) = %e #ei=%f,a2=%f\n", A, B, ia, ib, ip, e, ei, a2);
              for (int lm = 0; lm < NA; ++lm) {
                auto c = first_transform.data[index(a)][lm];
                U[lm] += e*c;
              }
            }
            for (int lm = 0; lm < NB; ++lm) {
              auto c = second_transform.data[index(b)][lm];
              for (int ia = 0; ia < NA; ++ia) {
                V[ia + lm*npure(A)] += c*U[ia];
              }
            }
          } // ib
          std::copy(V, V + NA*NB, h.E[hermite::index2(p)]);
        }
        Hs.push_back(h);
      }
    }

    for (size_t ij = 0; ij < Hs.size(); ++ij) {
      double p_cd[orbitals2.size()][ncart(First)*ncart(Second)] = {};
      const auto &Eij = Hs[ij].E;
      for (size_t kl = 0; kl < Hs.size(); ++kl) {
        const auto &Ekl = Hs[kl].E;
        auto [a,b] = Hs[ij].exps;
        auto [c,d] = Hs[kl].exps;
        auto C = Hs[ij].C*Hs[kl].C;
        //C *= (ij == kl ? 1 : 2);
        double r1[nherm2(L)] = { };
        md::r1::compute<L>(a+b, c+d, Hs[ij].r-Hs[kl].r, C, boys, r1);

        //double inv_2_q = 1/math::pow<L/2>(2*q);

        for (size_t ip = 0; ip < orbitals2.size(); ++ip) {
          auto p = orbitals2[ip];
          for (size_t iq = 0; iq < orbitals2.size(); ++iq) {
            auto q = orbitals2[iq];
            auto r = hermite::phase(q)*r1[hermite::index2(p+q)];
            for (int icd = 0; icd < NA*NB; ++icd) {
              p_cd[ip][icd] += Ekl[iq][icd]*r;
            }
          }
          // md::hermite_to_pure<First,Second>(
          //   [&](auto q) {
          //     return r1[hermite::index2(p+q)];
          //   },
          //   [&](auto c, auto d, auto v) {
          //     int phase = ((First+Second)%2 == 0 ? +1 : -1);
          //     p_cd[ip][index(c) + index(d)*NA] += phase*v*inv_2_q;
          //   }
          // );
        }
      }
      for (int icd = 0, i = 0; icd < NA*NB; ++icd) {
        for (int iab = 0; iab < NA*NB; ++iab) {
          double v = 0;
          for (size_t ip = 0; ip < orbitals2.size(); ++ip) {
            v += Eij[ip][iab]*p_cd[ip][icd];
          }
          V[i++] += double(v);
        }
      }
    }

    // printf("ab|ab %i,%i\n", First, Second);
    // for (int icd = 0, i = 0; icd < NA*NB; ++icd) {
    //   for (int iab = 0; iab < NA*NB; ++iab) {
    //     printf("  (%i,%i) = %e\n", iab, icd, V[i++]);
    //   }
    // }

  }

  double compute(Norm<double> f, const Gaussian& a, const Gaussian& b) {
    int na = nbf(a);
    int nb = nbf(b);
    std::vector<double> V(na*nb*na*nb);

    using Kernel = std::function<void(const Gaussian& a, const Gaussian& b, double *V)>;

    Kernel kernel_table[LMAX+1][LMAX+1] = {};
    foreach2(
	     std::make_index_sequence<LMAX+1>{},
	     std::make_index_sequence<LMAX+1>{},
      [&](auto A, auto B) {
        kernel_table[A][B] = Kernel(&screening::compute<A,B>);
      }
    );
    auto compute = kernel_table[a.L][b.L];
    libintx_assert(compute);
    compute(a, b, V.data());
    return f(V.size(), V.data(), 1);
  }

  template<int N, typename T>
  T norm(size_t n, const T *v, size_t inc) {
    T norm = T(0);
    for (size_t i = 0; i < n; ++i) {
      using std::abs;
      auto u = abs(*v);
      norm += math::pow<N>(u);
      v += inc;
    }
    if constexpr (N == 1) return norm;
    if constexpr (N == 2) {
      using std::sqrt;
      return sqrt(norm);
    }
    else {
      using std::pow;
      return pow(norm, 1.0/N);
    }
  }

  template<typename T>
  T absmax(size_t n, const T *v, size_t inc) {
    T norm = T(0);
    for (size_t i = 0; i < n; ++i) {
      using std::abs;
      auto u = abs(*v);
      using std::max;
      norm = max(u, norm);
    }
    return norm;
  }

  template<typename T>
  std::vector< std::tuple<Index2,T> >
  pairs(const Basis<Gaussian> &basis, Norm<double> norm, T min, libintx::num_threads num_threads) {
    md::IntegralEngine<2> ao(basis, basis);
    std::vector< std::tuple<Index2,T> > pairs;
    size_t nij = 0;
#pragma omp parallel for schedule(dynamic,1) num_threads(num_threads.value)
    for (size_t j = 0; j < basis.size(); ++j) {
      for (size_t i = 0; i < basis.size(); ++i) {
        Index2 ij = { (int)i, (int)j };
        if (j > i) continue;
        ++nij;
        if (basis[i].r != basis[j].r) {
          std::vector<double> V(nbf(basis[i])*nbf(basis[j]));
          ao.overlap({ij}, V.data());
          float vij = norm(V.size(), V.data(), 1);
          //if (vij < min) printf("|ij| %i %i %e\n ", i, j, vij);
          if (vij < min) continue;
        }
        float n = (float)compute(norm, basis[i], basis[j]);
        //if (n < min) continue;
#pragma omp critical(libintx_ao_screening)
        pairs.push_back({ij,sqrt(n)});
      }
    }
    // printf(
    //   "ao::screening:\n%lu pairs > %e out of %lu, %f density\n",
    //   pairs.size(), min, nij, (double)pairs.size()/nij
    // );
    return pairs;
  }

  template
  std::vector< std::tuple<Index2,float> >
  pairs(const Basis<Gaussian> &basis, Norm<double> norm, float min, libintx::num_threads);

  template
  double norm<2,double>(size_t, const double*, size_t);

  template
  double absmax<double>(size_t, const double*, size_t);

#ifdef LIBINTX_SIMD_DOUBLE

  template
  LIBINTX_SIMD_DOUBLE norm<2,LIBINTX_SIMD_DOUBLE>(size_t, const LIBINTX_SIMD_DOUBLE*, size_t);

  template
  LIBINTX_SIMD_DOUBLE absmax<LIBINTX_SIMD_DOUBLE>(size_t, const LIBINTX_SIMD_DOUBLE*, size_t);

#endif

}
