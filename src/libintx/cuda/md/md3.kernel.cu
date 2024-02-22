#include "libintx/cuda/md/md3.h"
#include "libintx/cuda/md/md3.kernel.h"
#include "libintx/utility.h"

#pragma nv_diag_suppress 2361

namespace libintx::cuda::md {

#ifdef LIBINTX_CUDA_MD_MD3_KERNEL_X_KET

  template
  void ERI3::compute<LIBINTX_CUDA_MD_MD3_KERNEL_X_KET>(
    const Basis1&,
    const Basis2&,
    TensorRef<double,2>,
    cudaStream_t stream
  );

#endif

  template<int X, int C, int D>
  auto ERI3::compute_v2(
    const Basis1& bra,
    const Basis2& ket,
    TensorRef<double,2> XCD,
    cudaStream_t stream)
  {
    //printf("ERI3::compute_v2<%i,%i,%i>\n", X,C,D);

    kernel::Basis1<X> x{bra.K, bra.N, bra.data};
    kernel::Basis2<C+D> cd(ket.first, ket.second, ket.K, ket.N, ket.data);

    constexpr int L = x.L+cd.L;
    constexpr int NP = x.nherm;
    constexpr int NQ = cd.nherm;
    const int NX = x.nbf;
    const int NCD = cd.nbf;

    dim3 grid = { (uint)x.N, (uint)cd.N };
    using Block = cuda::thread_block<std::clamp(NQ,32,128)>;

    TensorRef<double,4> qx {
      this->buffer<0>(NQ*x.N*NX*cd.N),
      { NQ, x.N, NX, cd.N }
    };

    for (int kcd = 0; kcd < ket.K; ++kcd) {
      for (int kx = 0; kx < bra.K; ++kx) {
        //double C = 2*std::pow(M_PI,2.5);
        //double Ck = (kx == 0 ? 0 : 1.0);
        // [q,i,x,kl]
        kernel::compute_q_x<Block,2><<<grid,Block{},0,stream>>>(
          x, cd, {kx,kcd},
          cuda::boys(),
          qx
        );
      }
      // [i,x,cd,kl] = [q,i,x,kl]'*H(cd,q,kl)'
      batch_gemm<RowMajor, RowMajor, ColumnMajor>(
        NX*x.N, NCD, cd.nherm,
        math::sqrt_4_pi5, // alpha
        qx.data(), NQ, NQ*NX*x.N,
        cd.gdata(0,kcd), NCD, cd.stride*cd.K,
        (kcd == 0 ? 0.0 : 1.0), // beta
        XCD.data(), NX*x.N, NX*x.N*NCD,
        cd.N, // batches
        stream
      );
    } // kcd
  }


  template<int X, int Ket>
  void ERI3::compute(
    const Basis1& x,
    const Basis2& ket,
    TensorRef<double,2> XCD,
    cudaStream_t stream)
  {


    foreach(
      std::make_index_sequence<Ket+1>{},
      [&](auto C) {
        constexpr int D = Ket-C;
        if constexpr (std::max<int>({C,D}) <= LMAX) {
          if (C != ket.first.L || D != ket.second.L) return;
          this->compute_v2<X,C,D>(x, ket, XCD, stream);
        }
      }
    );

  }

}
