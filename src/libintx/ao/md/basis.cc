#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/pure.transform.h"
#include "libintx/config.h"
#include "libintx/simd.h"

#include <iostream>

namespace libintx::md {

  template<int A, int B, typename T>
  void make_basis(
    const gto::Primitive<T> &g1,
    const array<T,3> &r1,
    const gto::Primitive<T> &g2,
    const array<T,3> &r2,
    Phase<int> phase,
    Hermite<T> *h, T *hermite_to_ao)
  {

    //constexpr int NP = nherm2(A+B);
    constexpr int NAB = npure(A)*npure(B);

   // constexpr pure::Transform<A> pure_transform_a;
    // constexpr pure::Transform<B> pure_transform_b;

    const auto &[a1,c1] = g1;
    const auto &[a2,c2] = g2;

    using std::exp;
    T Kab = exp(-(a1*a2)/(a1+a2)*norm(r1,r2));
    *h = Hermite<T>{
      .exp = (a1+a2),
      .C = c1*c2*Kab,
      .r = center_of_charge(a1,r1,a2,r2),
      .inv_2_exp = 1/math::pow<A+B>(phase.value*2*(a1 + a2))
    };

    E2<T,A,B,A+B> E(a1, a2, r1-r2);

    T phases[A+B+1] = { 1 };
    for (int i = 1; i <= A+B; ++i) {
      phases[i] = phases[i-1]*phase.value;
    }
    //libintx_unroll(28)
    for (auto p : hermite::orbitals2<A+B>) {
      T U[ncart(B)][ncart(A)] = {};
      T phase_p = phases[p.L()];
      //T phase_p = std::pow(phase.value, p.L());
      libintx_unroll(28)
      for (auto b : cartesian::orbitals<B>()) {
        libintx_unroll(28)
        for (auto a : cartesian::orbitals<A>()) {
          U[index(b)][index(a)] = phase_p*E(a,b,p);
        }
      }
      auto *H = hermite_to_ao + hermite::index2(p)*NAB;
      std::fill_n(H, npure(A,B), T(0));
      pure::cartesian_to_pure<A,B>(&U[0][0], H);
    }
  }



  template<int A, int B, typename T>
  HermiteBasis<2,T> make_basis(
    const Basis<Gaussian> &As,
    const Basis<Gaussian> &Bs,
    const std::vector<Index2> &pairs,
    Phase<int> phase,
    std::vector<T> &allocator)
  {

    // constexpr int NP = nherm2(A+B);
    // constexpr int NAB = npure(A)*npure(B);
    // constexpr pure::Transform<A> pure_transform_a;
    // constexpr pure::Transform<B> pure_transform_b;

    constexpr int Lanes = []() {
      if constexpr (std::is_scalar_v<T>) return 1;
      else return T::size();
    }();

    auto [first,second] = pairs.front();
    const auto &a0 = As[first];
    const auto &b0 = Bs[second];

    int N = pairs.size();
    int K = a0.K*b0.K;

    HermiteBasis<2,T> basis;
    basis.first = a0;
    basis.second = b0;
    basis.K = K;
    basis.N = N;
    basis.extent_ = (sizeof(Hermite<T>)/sizeof(T) + npure(A,B)*nherm2(A+B));
    allocator.resize(basis.extent_*K*((N+Lanes-1)/Lanes));
    basis.data_ = allocator.data();

    T Inf = math::infinity<double>;

    for (int ij = 0; ij < N; ij += Lanes) {
      array<T,3> r1 = { Inf, Inf, Inf };
      array<T,3> r2 = { Inf, Inf, Inf };
      for (int l = 0; l < Lanes; ++l) {
        if (ij+l >= N) break;
        auto [i,j] = pairs[ij+l];
        const auto &a = As[i];
        const auto &b = Bs[j];
        libintx_assert(a.L == A);
        libintx_assert(b.L == B);
        libintx_assert(a.K*b.K == K);
        if constexpr (std::is_scalar_v<T>) {
          r1 = center(a);
          r2 = center(b);
        }
        else {
          for (int k = 0; k < 3; ++k) {
            r1[k][l] = center(a)[k];
            r2[k][l] = center(b)[k];
          }
        }
      }
      for (int k = 0; k < K; ++k) {
        gto::Primitive<T> g1 = { 1, 0 };
        gto::Primitive<T> g2 = { 1, 0 };
        for (int l = 0; l < Lanes; ++l) {
          if (ij+l >= N) break;
          auto [i,j] = pairs[ij+l];
          const auto &a = As[i];
          const auto &b = Bs[j];
          const auto &[a1,c1] = gto::primitive(a,k%a.K);
          const auto &[a2,c2] = gto::primitive(b,k/a.K);
          if constexpr (std::is_scalar_v<T>)  {
            g1 = { a1, c1 };
            g2 = { a2, c2 };
          }
          else {
            g1.a[l] = a1;
            g1.C[l] = c1;
            g2.a[l] = a2;
            g2.C[l] = c2;
          }
        }
        make_basis<A,B>(
          g1, r1,
          g2, r2,
          phase,
          const_cast< Hermite<T>* >(basis.hermite(ij/Lanes,k)),
          const_cast<T*>(basis.hermite_to_ao(ij/Lanes,k))
        );
      }
    }

    return basis;

  }


  template<typename T>
  HermiteBasis<2,T> make_basis(
    const Basis<Gaussian> &As,
    const Basis<Gaussian> &Bs,
    const std::vector<Index2> &pairs,
    const double *norms,
    Phase<int> phase,
    int Batch,
    std::vector<T> &allocator)
  {
    if (pairs.empty()) return HermiteBasis<2,T>{};
    auto [first,second] = pairs.front();
    const auto &a = As[first];
    const auto &b = Bs[second];
    HermiteBasis<2,T> p;
    jump_table(
      std::make_index_sequence<(LMAX+1)*(LMAX+1)>{},
      a.L + b.L*(LMAX+1),
      [&](auto AB) {
        constexpr int A = AB%(LMAX+1);
        constexpr int B = AB/(LMAX+1);
        p = make_basis<A,B>(As, Bs, pairs, phase, allocator);
      }
    );
    return p;
  }


  template<typename T>
  HermiteBasis<1,T> make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    int Batch,
    std::vector< Hermite<T> > &allocator)
  {

    libintx_assert(!A.empty());
    libintx_assert(!idx.empty());

    int L = A[idx.front()].L;
    int K = A[idx.front()].K;

    for (auto i : idx) {
      libintx_assert(A[i].K == K);
      libintx_assert(A[i].L == L);
    }

    auto Inf = math::infinity<double>;

    if constexpr (std::is_scalar_v<T>) {
      size_t N = idx.size();
      size_t nbatch = (N + Batch - 1)/Batch;
      allocator.resize(K*nbatch*Batch);
      std::fill(
        allocator.begin()+K*N,
        allocator.end(),
        Hermite<T>{ 0, 0, { Inf, Inf, Inf }, 0 }
      );
      for (size_t i = 0; i < N; ++i) {
        auto &Ai = A[idx[i]];
        for (int k = 0; k < K; ++k) {
          auto &r = center(Ai);
          auto [e,C] = primitive(Ai,k);
          allocator[i] = { e, C, r, 1.0/(2*e) };
        }
      }
    }
    else {
      constexpr int Lanes = T::size();
      size_t N = Lanes*(idx.size() + Lanes - 1);
      size_t nbatch = (N + Batch - 1)/Batch;
      allocator.resize(K*nbatch*Batch);
      std::fill(
        allocator.begin() + K*(nbatch - 1),
        allocator.end(),
        Hermite<T>{ T(1), T(0), { T(Inf), T(Inf), T(Inf) }, T(0) }
      );
      for (size_t i = 0; i < idx.size(); ++i) {
        auto &Ai = A[idx[i]];
        for (int k = 0; k < K; ++k) {
          auto &r = center(Ai);
          auto [e,C] = primitive(Ai,k);
          auto& h = allocator[k + K*(i/Lanes)];
          h.exp[i%Lanes] =  e;//, C, r, 1.0/(2*e) };
          h.C[i%Lanes] =  C ; //, r, 1.0/(2*e) };
          h.r[0][i%Lanes] = r[0];
          h.r[1][i%Lanes] = r[1];
          h.r[2][i%Lanes] = r[2];
          h.inv_2_exp[i%Lanes] = 1.0/(2*e);
        }
      }
    }

    return HermiteBasis<1,T>{L,K,(int)idx.size(),Batch,allocator.data()};

  }

  template
  HermiteBasis<1,double> make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    int Batch,
    std::vector< Hermite<double> > &allocator
  );

  template
  HermiteBasis<2,double> make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    const double *norms,
    Phase<int> phase,
    int Batch,
    std::vector<double> &allocator
   );

#ifdef LIBINTX_SIMD_DOUBLE

  template
  HermiteBasis<1,LIBINTX_SIMD_DOUBLE> make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    int Batch,
    std::vector< Hermite<LIBINTX_SIMD_DOUBLE> > &allocator
  );

  template
  HermiteBasis<2,LIBINTX_SIMD_DOUBLE> make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    const double *norms,
    Phase<int> phase,
    int Batch,
    std::vector<LIBINTX_SIMD_DOUBLE> &allocator
   );

#endif

}
