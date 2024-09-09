#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/ao/md/r1.h"
#include "libintx/boys/chebyshev.h"
#include "libintx/math.h"
#include "libintx/simd.h"

#include "libintx/config.h"
#include "libintx/utility.h"

namespace libintx::md {

  template<typename S>
  auto params2(const Nuclear::Operator::Parameters &params) {
    const auto &v = params.centers;
    libintx_assert(v.size());
    if constexpr (std::is_scalar_v<S>) {
      return v;
    }
    else {
      constexpr int N = S::size();
      std::vector< std::tuple<S, std::array<S,3> > > s((v.size() + N - 1)/N);
      s.back() = { S(0), { S(0), S(0), S(0) } };
      for (size_t i = 0; i < v.size(); ++i) {
        std::get<0>(s[i/N])[i%N] = std::get<0>(v[i]);
        for (int k = 0; k < 3; ++k) {
          std::get<1>(s[i/N])[k][i%N] = std::get<1>(v[i])[k];
        }
      }
      return s;
    }
  }

  template<int A, int B, typename T>
  void cartesian_to_pure(const T (&V)[ncart(A)][ncart(B)], auto &&AB) {
    constexpr pure::Transform<A> pure_transform_a;
    constexpr pure::Transform<B> pure_transform_b;
    constexpr int NA = npure(A);
    T U[ncart(B)][NA] = {};
libintx_unroll(13)
    for (int lm = 0; lm < NA; ++lm) {
      constexpr int NA = ncart(A);
      constexpr int NB = ncart(B);
      T Ub[NB] = {};
libintx_unroll(28)
      for (int ia = 0; ia < NA; ++ia) {
        auto c = pure_transform_a.data[ia][lm];
        if (!c) continue;
        for (int ib = 0; ib < NB; ++ib) {
          Ub[ib] += c*V[ia][ib];
        }
      }
      for (int ib = 0; ib < NB; ++ib) {
        U[ib][lm] = Ub[ib];
      }
    }
    constexpr int NB = npure(B);
libintx_unroll(13)
    for (int lm = 0; lm < NB; ++lm) {
      T V[NA] = {};
      constexpr int NB = ncart(B);
libintx_unroll(28)
      for (int ib = 0; ib < NB; ++ib) {
        auto c = pure_transform_b.data[ib][lm];
        if (!c) continue;
        for (int ia = 0; ia < NA; ++ia) {
          V[ia] += c*U[ib][ia];
        }
      }
      for (int ia = 0; ia < NA; ++ia) {
        AB(ia,lm) = V[ia];
      }
    } // ib
  }


  template<typename T, int A, int B>
  void overlap(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A,B,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto &a : orbitals<A>()) {
libintx_unroll(28)
      for (auto &b : orbitals<B>()) {
        U[index(a)][index(b)] += C*E(a, b, Orbital{{0,0,0}});

      }
    }
  }

  template<typename T, int A, int B>
  void kinetic(T a1, T a2, auto &&R, T C, auto &&U) {
    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;
    E2<T,A,B+2,0> E(a1,a2,R);
    C *= sqrt(math::pow<3>(math::pi/(a1+a2)));
libintx_unroll(28)
    for (auto [l1,m1,n1] : orbitals<A>()) {
libintx_unroll(28)
      for (auto [l2,m2,n2] : orbitals<B>()) {
        T t0 = E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2);
        T t1 = (
          E.x(l1,l2+2)*E.y(m1,m2)*E.z(n1,n2) +
          E.x(l1,l2)*E.y(m1,m2+2)*E.z(n1,n2) +
          E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2+2)
        );
        T t2 = {};
        int l = l2*(l2-1);
        int m = m2*(m2-1);
        int n = n2*(n2-1);
        if (l) t2 += l*E.x(l1,l2-2)*E.y(m1,m2)*E.z(n1,n2);
        if (m) t2 += m*E.x(l1,l2)*E.y(m1,m2-2)*E.z(n1,n2);
        if (n) t2 += n*E.x(l1,l2)*E.y(m1,m2)*E.z(n1,n2-2);
        U(index(l1,m1,n1),index(l2,m2,n2)) += C*(a2*(2*B+3)*t0 - 2*a2*a2*t1 - 0.5*t2);
      }
    }
  }

  template<int A, int B, typename T, typename Z>
  void nuclear(
    double a1, auto &&r1, double a2, auto &&r2,
    const std::vector< std::tuple<Z,std::array<T,3> > > &Cs,
    double C, auto &&U)
  {

    using std::sqrt;
    using cartesian::orbitals;
    using cartesian::index;

    auto &boys = libintx::boys::chebyshev<2*LMAX,A+B>();

    assert(!Cs.empty());

    constexpr int NP = nherm2(A+B);

    T p = a1 + a2;
    array<T,3> P;
    for (int i = 0; i < 3; ++i) {
      P[i] = center_of_charge(a1, r1, a2, r2)[i];
    }
    T R[NP] = {};
    for (size_t i = 0; i < Cs.size(); ++i) {

      auto& [Zi,Ci] = Cs[i];
      auto PC = P-Ci;

      if constexpr (!is_simd_v<T>) {
        if (!Zi) continue;
      }

      T s[A+B+1] = { };//, a2, p, a1*a2, 1, 1 };
      auto x = p*norm(PC);
      boys.template compute<A+B>(x, s);
      T pi = 1;
      libintx_unroll(13)
      for (int i = 0; i <= A+B; ++i) {
        s[i] *= pi;
        //printf("F_[%i](%f) = %f\n", i, (double)x[3], (double)s[i][3]);
        pi *= -2*p;
      }

      auto V = [&,Zi=Zi](auto &&r) {
        R[r.index] += -Zi*r.value;
      };
      namespace r1 = libintx::md::r1;
      r1::visit<A+B>(V, PC, s);
    }

    double R1[NP] = {};
    if constexpr (std::is_scalar_v<T>) {
      for (size_t i = 0; i < NP; ++i) {
        R1[i] = R[i];
      }
    }
    else {
      for (size_t i = 0; i < NP; ++i) {
        for (size_t k = 0; k < T::size(); ++k) {
          R1[i] += R[i][k];
        }
        //printf("R[i] = %f\n", double(R1[i]));
      }
    }

    E2<double,A,B,A+B> E(a1,a2,r1-r2);
    C *= 2*math::pi/(a1+a2);

    foreach2(
      std::make_index_sequence<ncart(A)>{},
      std::make_index_sequence<ncart(B)>{},
      [&](auto ia, auto ib) {
        constexpr auto a = orbitals<A>()[ia];
        constexpr auto b = orbitals<B>()[ib];
        constexpr auto ab = a+b;
        constexpr auto PX = ab.lmn[0];
        constexpr auto PY = ab.lmn[1];
        constexpr auto PZ = ab.lmn[2];
        double u = 0;
        auto *Ex = E.px(a,b);
        auto *Ey = E.py(a,b);
        auto *Ez = E.pz(a,b);
        libintx_unroll(13)
        for (uint8_t iz = 0; iz <= PZ; ++iz) {
          libintx_unroll(13)
          for (uint8_t iy = 0; iy <= PY; ++iy) {
            double Eyz = Ey[iy]*Ez[iz];
            libintx_unroll(13)
            for (uint8_t ix = 0; ix <= PX; ++ix) {
              auto p = Orbital{ix,iy,iz};
              u += R1[hermite::index2(p)]*Ex[ix]*Eyz;
            }
          }
        }
        U[ia][ib] += C*u;
      }
    );

  }

  template<int A, int B, Operator Op, typename T, int KMAX>
  void compute2(
    const auto &params,
    const array<T,3> &r1,
    const array<T,3> &r2,
    std::pair<int,int> K,
    const gto::Primitive<T> (&g1)[KMAX],
    const gto::Primitive<T> (&g2)[KMAX],
    auto &&V)
  {
    T U[ncart(A)][ncart(B)] = {};
    array<T,3> R = r1-r2;
    T r = norm(R);
    for (int k1 = 0; k1 < K.first; ++k1) {
      for (int k2 = 0; k2 < K.second; ++k2) {
        auto [ei,Ci] = g1[k1];
        auto [ej,Cj] = g2[k2];
        using std::exp;
        T Kab = exp(-ei*ej/(ei+ej)*r);
        auto C = Kab*Ci*Cj;
        if constexpr (Op == Operator::Overlap) {
          overlap<T,A,B>(ei, ej, R, C, U);
        }
        if constexpr (Op == Operator::Kinetic) {
          constexpr bool Transpose = (B > A);
          auto Uij = [&](auto i, auto j) ->auto& {
            return (Transpose ? U[j][i] : U[i][j]);;
          };
          if (!Transpose) {
            kinetic<T,A,B>(ei, ej, R, C, Uij);
          }
          else {
            kinetic<T,B,A>(ej, ei, -R, C, Uij);
          }
        }
        if constexpr (Op == Operator::Nuclear) {
          static_assert(!is_simd_v<T>);
          nuclear<A,B>(ei, r1, ej, r2, params, C, U);
        }
      }
    }
    cartesian_to_pure<A,B>(U,V);
  }

  //using Gaussian2 = std::tuple<Gaussian,Gaussian>;
  using Gaussian2 = std::tuple<const Gaussian&, const Gaussian&>;

  template<int A, int B, Operator Op, typename Params, typename T>
  void compute2(
    const Params& params,
    const std::vector<Gaussian2> &abs,
    T* __restrict__ V, int ldV)
  {

    constexpr size_t N = simd::size<T,1>;

    for (size_t i = 0; i < abs.size(); i += N) {

      size_t Ni = std::min(N, abs.size()-i);

      const auto a = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<0>(abs[i + idx]);
      };

      const auto b = [&](size_t idx) -> const Gaussian* {
        if (idx >= Ni) return nullptr;
        return &std::get<1>(abs[i + idx]);
      };

      std::pair<int,int> K = { 0, 0 };
      for (size_t j = 0; j < N; ++j) {
        K.first = std::max(K.first, (a(j) ? a(j)->K : 0));
        K.second = std::max(K.second, (b(j) ? b(j)->K : 0));
      }
      assert(K.first && K.second);

      const auto &r1 = gto::pack_centers<T>(a);
      const auto &r2 = gto::pack_centers<T>(b);

      const auto &g1 = gto::pack_primitives<T>(a);
      const auto &g2 = gto::pack_primitives<T>(b);

      auto f = [i,V,ldV](auto ia, auto ib) ->auto& {
        return V[i/N + (ia + ib*npure(A))*ldV];
      };

      compute2<A,B,Op,T>(params,r1,r2,K,g1.data,g2.data,f);

    }
  }

  template<typename T, Operator Op, typename Params>
  void IntegralEngine<2>::compute(const Params &params, const std::vector<Index2> &ijs, const Visitor &V) {

    using Kernel = std::function<void(
      const Params&,
      const std::vector<Gaussian2>&,
      T* __restrict__, int ldV
    )>;

    static auto kernel_array = make_array<Kernel,LMAX+1,LMAX+1>(
      [](auto a, auto b) {
        return Kernel(&md::compute2<a,b,Op,Params,T>);
      }
    );

    auto [i,j] = ijs.front();
    const auto &a = this->bra_[i];
    const auto &b = this->ket_[j];
    auto kernel = kernel_array[a.L][b.L];

    constexpr int N = simd::size<T,1>;
    int Batch = (this->Batch ? this->Batch : N);

#pragma omp parallel num_threads(this->num_threads)
    {

      // int K = nprim(a)*nprim(b);
      int ldV = (Batch+N-1)/N;
      auto V_batch = std::make_unique<T[]>(npure(a.L,b.L)*ldV);

      std::vector<Gaussian2> batch;
      batch.reserve(Batch);

#pragma omp for schedule(dynamic,1)
      for (size_t ij = 0; ij < ijs.size(); ij += Batch) {
        size_t nij = std::min<size_t>(ijs.size()-ij,Batch);
        batch.clear();
        for (size_t k = 0; k < nij; ++k) {
          const auto& [i,j] = ijs[ij+k];
          batch.emplace_back(std::tie(bra_[i], ket_[j]));
        }
        kernel(params, batch, V_batch.get(), ldV);
        V(nij,ij,reinterpret_cast<double*>(V_batch.get()),N*ldV);
      }

    }

  }

  void IntegralEngine<2>::compute(Operator op, const std::vector<Index2> &ij, const Visitor &V) {
#ifdef LIBINTX_SIMD_DOUBLE
    using S = LIBINTX_SIMD_DOUBLE;
#else
    using S = double;
#endif
    if (op == Overlap) {
      this->compute<S,Overlap>(Overlap::Operator::Parameters{},ij,V);
    }
    if (op == Kinetic) {
      this->compute<S,Kinetic>(Kinetic::Operator::Parameters{},ij,V);
    }
    if (op == Nuclear) {
      const auto &params = params2<S>(std::get<Nuclear::Operator::Parameters>(this->params_));
      this->compute<double,Nuclear>(params,ij,V);
    }
  }

  void IntegralEngine<2>::compute(Operator op, const std::vector<Index2> &ijs, double *V) {
    int na = nbf(bra_[ijs[0].first]);
    int nb = nbf(bra_[ijs[0].second]);
    size_t ldV = ijs.size();
    auto v = [&](size_t batch, size_t idx, const double *U, size_t ldU) {
      for (int iab = 0; iab < na*nb; ++iab) {
        auto *dst = V + iab*ldV + idx;
        auto *src = U + iab*ldU;
        std::copy_n(src, batch, dst);
      }
    };
    this->compute(op, ijs, v);
  }

  IntegralEngine<2>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket)
    : bra_(bra), ket_(ket)
  {
    libintx_assert(!bra_.empty());
    libintx_assert(!ket_.empty());
  }

  IntegralEngine<2>::~IntegralEngine() {}

} // libintx::md

template<>
std::unique_ptr< libintx::ao::IntegralEngine<2> > libintx::ao::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket)
{
  return std::make_unique< libintx::md::IntegralEngine<2> >(bra,ket);
}
