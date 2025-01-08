#ifndef LIBINTX_AO_MD_R1_KERNEL_H
#define LIBINTX_AO_MD_R1_KERNEL_H

#include "libintx/forward.h"
#include "libintx/ao/md/r1.h"
#include "libintx/ao/md/r1/recurrence.h"
#include "libintx/math.h"

namespace libintx::md::r1 {

  template<int L, typename T, size_t N>
#ifdef LIBINTX_AO_MD_R1_KERNEL_INLINE
  LIBINTX_AO_MD_R1_KERNEL_INLINE
#endif
  void compute(const T &p, const T &q, auto &&PQ, const T &C, auto &&boys, T (&R1)[N]) {
    T pq = p*q;
    T alpha = pq/(p+q);
    T x = alpha*norm(PQ);
    T s[L+1] = {};
    boys.template compute<L>(x, 0, s);
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
    namespace r1 = libintx::md::r1;
    if constexpr (L <= 12) {
      r1::compute<L>(PQ, s, R1);
    }
    else {
      constexpr r1::Recurrence<L> recurrence;
      for (int i = 0; i <= L; ++i) {
        R1[i] = s[i];
      }
      r1::compute<L>(recurrence, PQ, R1);
    }
    // for (int i = 0; i < N; ++i) {
    //   //printf("r1[%i] = %f\n", i, R1[i]);
    // }
  }

}

#endif /* LIBINTX_AO_MD_R1_KERNEL_H */
