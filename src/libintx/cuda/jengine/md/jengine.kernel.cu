#include "libintx/cuda/forward.h"
#include "libintx/cuda/jengine/md/forward.h"
#include "libintx/boys/cuda/chebyshev.h"
#include "libintx/engine/md/r1.h"

#include "libintx/cuda/api/kernel.h"
#include "libintx/cuda/api/stream.h"
#include "libintx/cuda/api/thread_group.h"

namespace libintx::cuda::jengine::md {
namespace {

  // __inline__ __device__
  // static double warp_sum(double v) {
  //   for (int offset = warpSize/2; offset > 0; offset /= 2)  {
  //     v += __shfl_down_sync(0xffffffff, v, offset);
  //   }
  //   return v;
  // }

  namespace r1 = libintx::md::r1;

  template<int Bra, int Ket, class Boys>
  struct DFJ {

    static constexpr int M = Bra+Ket;
    static constexpr int NW = 32;

    template<int Step>
    static void kernel(
      const Boys &boys,
      int NP, const Primitive2 *P,
      int NQ, const Primitive2 *Q,
      const double* input,
      double* output,
      float cutoff,
      Stream &stream)
    {
      if (!NP || !NQ) return;
      dim3 grid = { (unsigned int)NP };
      dim3 block = { NW };
      //shmem += nbra()*8;
      //printf("DFJ<%i,%i>\n", Bra, Ket);
      cuda::kernel::launch<32,2><<<grid,block,0,stream>>>(
        DFJ{},
        std::integral_constant<int,Step>{},
        boys, NQ, P, Q,
        input, output, cutoff
      );
    }

    template<class Visitor>
    __device__
    static void compute1(
      const Boys &boys,
      double p, double q,
      const Center &PQ,
      Visitor &&v)
    {

      double alpha = (p*q)/(p+q);

      double s[M+1];
      boys.template compute<M>(alpha*norm(PQ), 0, s);

      // printf("p=%e, q=%e\n", p, q);

      {
        double pq = p*q;
        double C = rsqrt(pq*pq*(p+q));
        // double Kab = exp(-(a*b)/p*norm(P));
        // double Kcd = 1;//exp(-norm(Q));
        // C *= Kab*Kcd;
        for (size_t m = 0; m <= M; ++m) {
          s[m] *= C;
          C *= -2*alpha;
          //if (m < Ket) s[m] = 0;
        }
      }

      constexpr auto Order = r1::DepthFirst;
      r1::visit<M,Order>(
        [&](auto &r) constexpr {
          r1::visitor<Bra,Ket>::apply1(v,r);
        },
        PQ, s
      );

    }

    __device__
    void operator()(
      std::integral_constant<int,1>,
      const Boys &boys, int NQ,
      const Primitive2 *P1, const Primitive2 *Q,
      const double* __restrict__ Dp,
      double* __restrict__ Xq,
      float cutoff) const
    {

      __shared__ Primitive2 p;
      memcpy1(&P1[blockIdx.x], &p, this_thread_block());
      __syncthreads();

      __shared__ Center P;
      if (threadIdx.x == 0) {
        P = center_of_charge(p.exp[0], p.r[0], p.exp[1], p.r[1]);
      }

      constexpr int nbra = nherm2(Bra);
      constexpr int nket = nherm1(Ket);

      __shared__ double shared_Xq[NW*nket];
      //double local_Xq[nket];
      __shared__ double shared_Dp[nbra];

      // printf("blockIdx=%i, Bra=%i, Ket=%i, %p\n", blockIdx.x, Bra, Ket, Dp);

      for (size_t i = threadIdx.x; i < nbra; i += NW) {
        shared_Dp[i] = Dp[i+blockIdx.x*nbra];
      }

      __shared__ struct {
        double exponent;
        Center r;
      } qk[NW*2];

      __shared__ int32_t ik[NW*2];

      for (int K = 0, kb = 0; K < NQ; K += NW) {

        //int kb = NQ-k;
        //printf("NQ=%i, k=%i\n", NQ, k);

        if (kb < NW) {
          bool keep = false;
          double exponent = 0;
          Center center;
          if (threadIdx.x+K < NQ) {
            auto q = Q[threadIdx.x+K];
            exponent = q.exp[0];
            center = q.r[0];
            keep = (p.norm*q.norm > cutoff);
            //printf("%i: keep=%i\n", threadIdx.x, keep);
          }
          //kb += min(NQ-K,NW);
          uint32_t mask = __ballot_sync(0xFFFFFFFF, keep);
          if (keep) {
            auto before = (uint64_t(mask) << (NW-threadIdx.x));
            auto idx = __popc(uint32_t(before))+kb;
            qk[idx].exponent = exponent;
            qk[idx].r = center;
            ik[idx] = K+threadIdx.x;
          }
          kb += __popc(mask);
          // printf("%i: keep=%i mask=0x%x popc=%i kb=%i k=%i\n",
          //        threadIdx.x, keep, mask, __popc(mask), kb, k);
        }

        if (kb < NW && K+NW < NQ) continue;

        __syncthreads();

        fill(NW*nket, shared_Xq, 0.0, this_thread_block());

        // for (size_t i = 0; i < nket; ++i) {
        //   local_Xq[i] = 0;
        // }

        __syncthreads();

        if (threadIdx.x < kb) {

          auto q = qk[threadIdx.x];
          auto PQ = P - q.r;

          // printf("a=%f, b=%f, P=%f,%f,%f CD=%f,%f,%f\n",
          //        p.exponent, q.exponent,
          //        AB[0], AB[1], AB[2],
          //        CD[0], CD[1], CD[2]
          // );

          compute1(
            boys, p.exp[0]+p.exp[1], q.exponent, PQ,
            [&](auto p, auto q, auto v) {
              auto kp = hermite::index2(p);
              auto kq = hermite::index1(q);
              shared_Xq[kq+threadIdx.x*nket] += v*shared_Dp[kp];
            }
          );

        }

        __syncthreads();

        // __shared__ double shared_Xq[NW*nket];
        // for (int i = 0; i < nket; ++i) {
        //   shared_Xq[i+threadIdx.x*nket] = local_Xq[i];
        // }
        // __syncthreads();

        int nk = min(NW,kb);
        //printf("k=%i nket=%i nq=%i\n", k, nket, nq);
        for (int i = threadIdx.x; i < nket*nk; i += NW) {
          auto l = nket*ik[i/nket] + i%nket;
          // auto l = k+threadIdx.x + i%nket;
          // auto l = k*nket + i;
          atomicAdd(Xq+l, shared_Xq[i]);
          // printf(
          //   "Xq'[%i]=%e j[%i]=%e\n",
          //   i, sh_Xq[i], k*nket+i, Xq[k*nket+i]
          // );
        }
        kb -= nk;

        __syncthreads();

      }

    }

    __device__
    void operator()(
      std::integral_constant<int,2>,
      const Boys &boys, int NQ,
      const Primitive2 *P2, const Primitive2 *Q,
      const double* __restrict__ Xq,
      double* __restrict__ Jp,
      float cutoff) const
    {

      __shared__ Primitive2 p;
      memcpy1(&P2[blockIdx.x], &p, this_thread_block());
      __syncthreads();

      __shared__ Center P;
      if (threadIdx.x == 0) {
        P = center_of_charge(p.exp[0], p.r[0], p.exp[1], p.r[1]);
      }

      constexpr int nbra = nherm2(Bra);
      constexpr int nket = nherm1(Ket);

      __shared__ double sh_Xq[NW*nket];
      double local_Jp[nbra] = {};

      // printf("blockIdx=%i, Bra=%i, Ket=%i, %p\n", blockIdx.x, Bra, Ket, Dp);

      // for (size_t i = threadIdx.x; i < nbra; i += NW) {
      //   sh_Jp[i] = 0;
      // }

      __shared__ struct {
        double exponent;
        Center r;
      } qk[NW*2];

      __shared__ int32_t ik[NW*2];

      for (int K = 0, kb = 0; K < NQ; K += NW) {

        //int kb = NQ-k;
        //printf("NQ=%i, k=%i\n", NQ, k);

        if (kb < NW) {
          bool keep = false;
          double exponent = 0;
          Center center;
          if (threadIdx.x+K < NQ) {
            auto q = Q[threadIdx.x+K];
            exponent = q.exp[0];
            center = q.r[0];
            keep = true;//(p.norm*q.norm > cutoff);
            //printf("%i: keep=%i\n", threadIdx.x, keep);
          }
          //kb += min(NQ-K,NW);
          uint32_t mask = __ballot_sync(0xFFFFFFFF, keep);
          if (keep) {
            auto before = (uint64_t(mask) << (NW-threadIdx.x));
            auto idx = __popc(uint32_t(before))+kb;
            qk[idx].exponent = exponent;
            qk[idx].r = center;
            ik[idx] = K+threadIdx.x;
          }
          kb += __popc(mask);
          // printf("%i: keep=%i mask=0x%x popc=%i kb=%i k=%i\n",
          //        threadIdx.x, keep, mask, __popc(mask), kb, k);
        }

        if (kb < NW && K+NW < NQ) continue;

        __syncthreads();

        int nk = min(NW,kb);
        //printf("k=%i nket=%i nq=%i\n", k, nket, nq);
        for (int i = threadIdx.x; i < nket*nk; i += NW) {
          auto l = nket*ik[i/nket] + i%nket;
          // auto l = k+threadIdx.x + i%nket;
          // auto l = k*nket + i;
          sh_Xq[i] = Xq[l];
          // printf(
          //   "Jq'[%i]=%e j[%i]=%e\n",
          //   i, sh_Jq[i], k*nket+i, Jq[k*nket+i]
          // );
        }

        __syncthreads();

        if (threadIdx.x < kb) {

          auto q = qk[threadIdx.x];
          auto PQ = P - q.r;

          // printf("a=%f, b=%f, P=%f,%f,%f CD=%f,%f,%f\n",
          //        p.exponent, q.exponent,
          //        AB[0], AB[1], AB[2],
          //        CD[0], CD[1], CD[2]
          // );

          compute1(
            boys, p.exp[0]+p.exp[1], q.exponent, PQ,
            [&](auto p, auto q, auto v) {
              auto kp = hermite::index2(p);
              auto kq = hermite::index1(q);
              local_Jp[kp] += v*sh_Xq[kq+threadIdx.x*nket];
            }
          );

        }

        kb -= nk;

      }

      //fill(nbra, Jp+blockIdx.x*nbra, 34.2, this_thread_block());

      for (size_t i = 0; i < nbra; ++i) {
        atomicAdd(&Jp[i+blockIdx.x*nbra], local_Jp[i]);
        // double v = warp_sum(local_Jp[i]);
        // //v = local_Jp[i];
        // if (threadIdx.x == 0) {
        //   Jp[i+blockIdx.x*nbra] = v;
        // }
      }

    } // transform(...)

  };

}
}

//#define LIBINTX_CUDA_MD_JENGINE_KERNEL_BRA_KET 0,0

template<int Bra, int Ket, int Step, class Boys>
void libintx::cuda::jengine::md::df_jengine_kernel(
  const Boys &boys,
  int NP, const Primitive2 *P,
  int NQ, const Primitive2 *Q,
  const double* input,
  double* output,
  float cutoff,
  Stream &stream
)
{
  auto kernel = &DFJ<Bra,Ket,Boys>::template kernel<Step>;
  kernel(boys, NP, P, NQ, Q, input, output, cutoff, stream);
}

#define LIBINTX_CUDA_MD_JENGINE_KERNEL(...)                             \
  template                                                              \
  void libintx::cuda::jengine::md::df_jengine_kernel<__VA_ARGS__>(      \
    const Boys &boys,                                                   \
    int NP, const Primitive2 *P,                                        \
    int NQ, const Primitive2 *Q,                                        \
    const double* input,                                                \
    double* output,                                                     \
    float cutoff,                                                       \
    Stream&                                                             \
  );

LIBINTX_CUDA_MD_JENGINE_KERNEL(
  LIBINTX_CUDA_MD_JENGINE_KERNEL_BRA_KET,1,libintx::cuda::Boys);
LIBINTX_CUDA_MD_JENGINE_KERNEL(
  LIBINTX_CUDA_MD_JENGINE_KERNEL_BRA_KET,2,libintx::cuda::Boys);
