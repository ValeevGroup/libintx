#ifndef LIBINTX_AO_MD_R1_H
#define LIBINTX_AO_MD_R1_H

#include "libintx/orbital.h"
#include "libintx/math.h"

namespace libintx::md::r1 {

  template<int L, typename T, int I = 0, int J = 0, int K = 0>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void visit(auto &&V, const auto &pq, const T* __restrict__ A1, const T* __restrict__ A2 = nullptr) {
    constexpr int M = I+J+K;
    std::integral_constant<int,hermite::index2(Orbital{I,J,K})> ijk;
    V(ijk, A1[0]);
    //V[ijk] = A1[0];
    if constexpr (L > M) {
      T A0[L-M] = {};
      if constexpr (!J && !K) {
        libintx_unroll(24)
        for (int l = 0; l < L-M; ++l) {
          A0[l] = pq[0]*A1[l+1];
          if constexpr (I) A0[l] += I*A2[l+1];
        }
        visit<L,T,I+1,J,K>(V,pq,A0,A1);
      }
      if constexpr (!K) {
        T A0[L-M] = {};
        libintx_unroll(24)
        for (int l = 0; l < L-M; ++l) {
          A0[l] = pq[1]*A1[l+1];
          if constexpr (J) A0[l] += J*A2[l+1];
        }
        visit<L,T,I,J+1,K>(V,pq,A0,A1);
      }
      {
        T A0[L-M] = {};
        libintx_unroll(24)
        for (int l = 0; l < L-M; ++l) {
          A0[l] = pq[2]*A1[l+1];
          if constexpr (K) A0[l] += K*A2[l+1];
        }
        visit<L,T,I,J,K+1>(V,pq,A0,A1);
      }
    }
  }

  template<int L, typename T, size_t N>
  LIBINTX_GPU_ENABLED LIBINTX_ALWAYS_INLINE
  void compute(auto &&PQ, const T (&s)[L+1], T (&R1)[N]) {
    auto V = [&](auto &&idx, auto &&v) { return R1[idx] = v; };
    r1::visit<L>(V, PQ, s);
  }

  template<int L, typename T, size_t N>
#ifdef LIBINTX_AO_MD_R1_COMPUTE_INLINE
  LIBINTX_AO_MD_R1_COMPUTE_INLINE
#endif
  void compute(const T &p, const T &q, auto &&PQ, const T &C, auto &&boys, T (&R1)[N]) {
    T pq = p*q;
    T alpha = pq/(p+q);
    T x = alpha*norm(PQ);
    T s[L+1] = {};
    boys.template compute<L>(x, 0, s);
    using std::sqrt;
    auto Si = C/sqrt(pq*pq*(p+q))*math::sqrt_4_pi5;
    //T Kab = exp(-(a*b)/p*norm(P));
    //T Kcd = 1;//exp(-norm(Q));
    //C *= Kcd;
libintx_unroll(25)
    for (int i = 0; i <= L; ++i) {
      s[i] *= Si;
      Si *= -2*alpha;
    }
    //printf("p,q,PQ = %f,%f,%f, s[0] = %f\n", p, q, norm(PQ), s[0]);
    auto V = [&](auto &&idx, auto &&v) { return R1[idx] = v; };
    r1::visit<L>(V, PQ, s);
    // for (int i = 0; i < N; ++i) {
    //   //printf("r1[%i] = %f\n", i, R1[i]);
    // }
  }

}

#endif /* LIBINTX_AO_MD_R1_H */
