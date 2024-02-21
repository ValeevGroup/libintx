#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/eri4.h"
#include "libintx/cuda/md/eri4.kernel.h"

#include "libintx/cuda/api/kernel.h"
#include "libintx/boys/cuda/chebyshev.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"
#include "libintx/tuple.h"

namespace libintx::cuda::md {

  template
  void ERI4::compute<LIBINTX_CUDA_MD_ERI4_KERNEL_BRA_KET>(
    ERI4 &eri,
    const Basis2&,
    const Basis2&,
    double *V, size_t ldV,
    cudaStream_t stream
  );

  template<int Bra, int Ket>
  void ERI4::compute(
    ERI4 &eri,
    const Basis2& bra,
    const Basis2& ket,
    double *V, size_t ldV,
    cudaStream_t stream)
  {
    //printf("ERI4::compute<%i,%i>\n", Bra, Ket);
    eri4::Basis2<Bra> ab(bra.first, bra.second, bra.K, bra.N, bra.data);
    eri4::Basis2<Ket> cd(ket.first, ket.second, ket.K, ket.N, ket.data);
    //assert(cd.nbf*cd.N <= ldV);
    dim3 grid = { (uint)ab.N, (uint)cd.N };
    using Block = cuda::thread_block<32,4>;
    auto &pq = eri.pq_;
    pq.resize((grid.x*grid.y)*(ab.nherm*cd.nherm));
    auto &abq = eri.abq_;
    abq.resize((grid.x*grid.y)*(ab.nbf*cd.nherm));
    for (int kcd = 0; kcd < ket.K; ++kcd) {
      for (int kab = 0; kab < bra.K; ++kab) {
        double C = math::sqrt_4_pi5;
        double Ck = (kab == 0 ? 0 : 1.0);
        eri4::eri4_pq<Block,2><<<grid,Block{},0,stream>>>(ab, cd, {kab,kcd}, cuda::boys(), pq.data());
        // cuda::stream::synchronize();
        // for (int iq = 0; iq < cd.nherm*cd.N; ++iq) {
        //   for (int ip = 0; ip < ab.nherm*ab.N; ++ip) {
        //     //printf("(p|q)[%i,%i] = %f\n", ip, iq, pq.data()[iq + ip*cd.nherm*cd.N]);
        //   }
        // }
        eri4::eri4_ap_px<0>(ab, cd.N*cd.nherm, ab.gdata(0,kab), pq.data(), C, Ck, abq.data(), stream);
        // cuda::stream::synchronize();
        // for (int iq = 0; iq < cd.N*cd.nherm; ++iq) {
        //   for (int i = 0; i < ab.N*ab.nbf; ++i) {
        //     //printf("(ab|q)[%i,%i] = %f\n", i, iq, abq.data()[iq + i*cd.nherm*cd.N]);
        //   }
        // }
      }
      double Ck = (kcd == 0 ? 0 : 1.0);
      eri4::eri4_ap_px<1>(cd, ab.N*ab.nbf, cd.gdata(0,kcd), abq.data(), 1.0, Ck, V, stream);
    }
    // cuda::stream::synchronize();
    // for (int k = 0; k < cd.nbf*cd.N; ++k) {
    //   for (int i = 0; i < ab.nbf*ab.N; ++i) {
    //     //printf("V[%i,%i] = %f\n", i, k, V[k + i*cd.nbf*cd.N]);
    //   }
    // }
    //printf("\n\n\n");
  }

}
