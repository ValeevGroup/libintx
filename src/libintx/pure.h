#ifndef LIBINTX_PURE_H
#define LIBINTX_PURE_H

#include "libintx/math.h"
#include "libintx/simd.h"
#include "libintx/orbital.h"
#include "libintx/config.h"

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

template<typename Orbitals>
constexpr static size_t nnzero(const Orbitals& orbitals) {
  size_t n = 0;
  for (auto o : orbitals) {
    auto [i,j,k] = o.lmn;
    int l = i+j+k;
    for (int m = -l; m <= l; ++m) {
      if (coefficient(l,m,i,j,k)) ++n;
    }
  }
  return n;
  //return 100;
}

template<int LMAX>
struct OrbitalTransform {

  constexpr static int maxp = npure(LMAX);

  using Pure = pure::Orbital;
  using Cartesian = cartesian::Orbital;

  struct Index {
    uint16_t offset = 0;
    array<uint8_t,14> orbital = {};
  };

  constexpr static auto cart() {
    return cartesian::orbitals(cartesian::index_sequence<LMAX>);
  }

  constexpr static auto pure() {
    array<Pure,(LMAX+1)*(LMAX+1)> s = {};
    for (int16_t l = 0; l <= LMAX; ++l) {
      for (int16_t m = -l; m <= l; ++m) {
        Pure o = {l,m};
        s[l*l+index(o)] = o;
      }
    }
    return s;
  }

  constexpr static auto coefficient(Pure s, Cartesian t) {
    auto [l,m] = s;
    auto [i,j,k] = t;
    return pure::coefficient(l,m,i,j,k);
  }

  template<typename F>
  LIBINTX_GPU_ENABLED
  constexpr void apply(F &&f, int idx) const {
    auto [k,bf] = this->index_[idx];
    f(bf[0], this->coefficients_[k]);
    for (size_t i = 1; bf[i] && i < std::size(bf); ++i) {
      f(bf[i], this->coefficients_[k+i]);
    }
  }

  constexpr OrbitalTransform() {
    constexpr auto pure = this->pure();
    constexpr auto cart = this->cart();
    for (size_t j = 0, k = 0; j < pure.size(); ++j) {
      assert(k <= std::numeric_limits<uint16_t>::max());
      index_[j].offset = k;
      for (size_t i = 0, ij = 0; i < cart.size(); ++i) {
        auto c = coefficient(pure[j],cart[i]);
        if (!c) continue;
        coefficients_[k++] = c;
        index_[j].orbital[ij++] = index(cart[i]);
      }
    }
  }

private:
  constexpr static size_t N = nnzero(cart());
  array<double,N> coefficients_ = {};
  array<Index,std::size(pure())> index_ = {};

};

LIBINTX_GPU_CONSTANT
const auto orbital_transform = pure::OrbitalTransform<std::max(LMAX,XMAX)>();

}

namespace libintx {

template<typename Integer>
void cartesian_to_pure(Integer L, double *T, size_t stride = 1) {
  double p[pure::orbital_transform.maxp] = {};
  for (int i = 0; i < npure(L); ++i) {
    auto f = [&,i](auto k, auto c) {
      p[i] += c*T[k*stride];
    };
    pure::orbital_transform.apply(f, pure::index(L)+i);
  }
  for (int i = 0; i < npure(L); ++i) {
    T[i*stride] = p[i];
  }
}

// T[L,N] -> T[L',N]
template<typename Integer>
void cartesian_to_pure(Integer L, int N, const double *T, double *U) {
  std::fill(U, U+npure(L)*N, 0.0);
  for (int i = 0; i < npure(L); ++i) {
    auto f = [&,i](auto k, auto c) {
      simd::axpy(N, c, T+k*N, U+i*N);
    };
    pure::orbital_transform.apply(f, pure::index(L)+i);
  }
}

}

#ifdef __CUDACC__

#include <cooperative_groups.h>

namespace libintx {

using cuda_thread_group = ::cooperative_groups::thread_group;

// T[N,L] -> T[N,L']
template<typename Integer>
__device__
inline void cartesian_to_pure(Integer L, int N, double *T, cuda_thread_group g) {
  auto rank = g.thread_rank();
  for (size_t j = rank; j < N; j += g.size()) {
    auto *U = T+j*ncart(L);
    double p[npure(L)] = {};
#pragma unroll
    for (size_t i = 0; i < npure(L); ++i) {
      auto f = [&](auto k, auto c) {  p[i] += c*U[k]; };
      pure::orbital_transform.apply(f, pure::index(L)+i);
    }
    ::cooperative_groups::coalesced_threads().sync();
#pragma unroll
    for (size_t i = 0; i < npure(L); ++i) {
      T[i+j*npure(L)] = p[i];
    }
  }
  g.sync();
}

// T[A,B,N] -> T[A',B',N]
template<size_t N = 1> // inner stride N
__device__
inline void cartesian_to_pure(
  std::pair<bool,bool> convert,
  int A, int B, double *T,
  cuda_thread_group g)
{
  auto rank = g.thread_rank();
  int na = ncart(A);
  int nb = ncart(B);
  if (!convert.first) goto B;
  // T[A,B,N] -> T[A',B,N]
  na = npure(A);
  for (int j = rank; j < N*nb; j += g.size()) {
    auto *U = T+j;
    double p[pure::orbital_transform.maxp] = {};
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == na) break;
      auto f = [&](auto k, auto c) {  p[i] += c*U[k*nb*N]; };
      pure::orbital_transform.apply(f, pure::index(A)+i);
    }
    ::cooperative_groups::coalesced_threads().sync();
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == na) break;
      U[i*nb*N] = p[i];
    }
  }
  g.sync();
 B:
  if (!convert.second) return;
  // T[A',B,N] -> T[A',B',N]
  g.sync();
  nb = npure(B);
  for (int ik = rank; ik < N*na; ik += g.size()) {
    int i = ik/N;
    int k = ik%N;
    auto *U = T+k+i*ncart(B)*N;
    double p[pure::orbital_transform.maxp] = {};
    assert(nb <= std::size(p));
#pragma unroll
    for (size_t j = 0; j < std::size(p); ++j) {
      if (j == nb) break;
      auto f = [&](auto kcart, auto c) { p[j] += c*U[kcart*N]; };
      pure::orbital_transform.apply(f, pure::index(B)+j);
    }
    ::cooperative_groups::coalesced_threads().sync();
#pragma unroll
    for (size_t j = 0; j < std::size(p); ++j) {
      if (j == nb) break;
      T[k+N*(j+i*nb)] = p[j];
    }
  }
  g.sync();
  return;
}

__device__
inline void pure_to_cartesian(
  std::pair<bool,bool> convert,
  int A, int B, double *T,
  cuda_thread_group g)
{
  auto rank = g.thread_rank();
  double p[pure::orbital_transform.maxp];
  if (!convert.first) goto B;
  if (rank < npure(B)) {
#define T(i,j) T[i*npure(B)+j]
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == npure(A)) break;
      p[i] = T(i,rank);
    }
    ::cooperative_groups::coalesced_threads().sync();
    for (size_t k = 0; k < ncart(A); ++k) {
      T(k,rank) = 0;
    }
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == npure(A)) break;
      auto t = [&](auto k, auto c) { T(k,rank) += c*p[i]; };
      pure::orbital_transform.apply(t, pure::index(A)+i);
    }
#undef T
  }
  g.sync();
 B:
  if (!convert.second) return;
  if (rank < ncart(A)) {
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == npure(B)) break;
      p[i] = T[i+rank*npure(B)];
    }
    ::cooperative_groups::coalesced_threads().sync();
#define T(i,j) T[i*ncart(B)+j]
    for (size_t k = 0; k < ncart(B); ++k) {
      T(rank,k) = 0;
    }
#pragma unroll
    for (size_t i = 0; i < std::size(p); ++i) {
      if (i == npure(B)) break;
      auto t = [&](auto k, auto c) { T(rank,k) += c*p[i]; };
      pure::orbital_transform.apply(t, pure::index(B)+i);
    }
#undef T
  }
  g.sync();
  return;
}

__device__
inline void pure_to_cartesian(int L, double *A, cuda_thread_group g) {
  pure_to_cartesian({true,false}, L, 0, A, g);
  // auto rank = g.thread_rank();
  // double s = 0;
  // if (rank < npure(L)) s = A[rank];
  // if (rank < ncart(L)) A[rank] = 0;
  // g.sync();
  // if (rank < npure(L)) {
  //   auto t = [&](auto k, auto c) { atomicAdd(A+k, c*s); };
  //   pure::orbital_transform.apply(t, pure::index(L)+rank);
  // }
}

__device__
inline void cartesian_to_pure(int L, double *A, cuda_thread_group g) {
  cartesian_to_pure({true,false}, L, 0, A, g);
  // auto rank = g.thread_rank();
  // double s = 0;
  // if (rank < npure(L)) {
  //   auto t = [&](auto k, auto c) { s += c*A[k]; };
  //   pure::orbital_transform.apply(t, pure::index(L)+rank);
  // }
  // g.sync();
  // if (rank < npure(L)) A[rank] = s;
}

}

#endif

#endif /* LIBINTX_PURE_H */
