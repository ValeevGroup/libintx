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
  auto ERI3::compute_v0(
    const Basis1& bra,
    const Basis2& ket,
    TensorRef<double,2> XCD,
    cudaStream_t stream)
  {

    constexpr int shmem = 0;

    using kernel::Basis1;
    using kernel::Basis2;

    //printf("ERI4::compute<%i,%i> bra.K=%i, ket.K=%i \n", Bra, Ket, bra.K, ket.K);

    Basis1<X> x{bra.K, bra.N, bra.data};
    Basis2<C,D> cd(ket.K, ket.N, ket.data, ket.k_stride, nullptr);

    using kernel_x = kernel::md_v0_kernel<Basis1<X>, Basis2<C,D>, 128,1,1, MaxShmem>;

    using kernel_xy = typename kernel::find_if<
      800, MaxShmem,
      kernel::md_v0_kernel<Basis1<X>, Basis2<C,D>, 32,4,1, MaxShmem>,
      kernel::md_v0_kernel<Basis1<X>, Basis2<C,D>, 16,8,1, MaxShmem>
      //kernel::md_v0_kernel<Basis1<X>, Basis2<C,D>, 8,16,1, MaxShmem>
      >::type;

    if constexpr (kernel::test<kernel_x>(900,MaxShmem)) {
      typename kernel_x::ThreadBlock thread_block;
      dim3 grid = {
        (uint)(x.N+thread_block.x-1)/thread_block.x,
        (uint)(cd.N+thread_block.z-1)/thread_block.z
      };
      launch<<<grid,thread_block,shmem,stream>>>(
        kernel_x(), x, cd, cuda::boys(), std::tuple{}, XCD
      );
      //printf("v0:xz\n");
      return std::true_type();
    }
    else if constexpr (!std::is_same_v<kernel_xy,void>) {
      typename kernel_xy::ThreadBlock thread_block;
      dim3 grid = {
        (uint)(x.N+thread_block.x-1)/thread_block.x,
        (uint)(cd.N)
      };
      for (int kx = 0; kx < bra.K; ++kx) {
        launch<<<grid,thread_block,shmem,stream>>>(
          kernel_xy(), x, kx, cd, cuda::boys(), std::tuple{}, XCD
        );
      }
      //printf("v0:xz\n");
      return std::true_type();
    }
    else {
      return std::false_type();
    }

  }

  template<int X, int C, int D>
  auto ERI3::compute_v2(
    const Basis1& bra,
    const Basis2& ket,
    TensorRef<double,2> XCD,
    cudaStream_t stream)
  {
    //printf("ERI3::compute_v2<%i,%i,%i>\n", X,C,D);

    kernel::Basis1<X> x{bra.K, bra.N, bra.data};
    kernel::Basis2<C+D> cd(ket.first, ket.second, ket.K, ket.N, ket.data, ket.k_stride);

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
        cd.gdata(0,kcd), NCD, cd.stride,
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
          constexpr auto v2 = (
            (X == 3 && (npure(C)*npure(D) > 5*7)) ||
            (X > 3 && (npure(C)*npure(D) > 5*5))
          );
          if constexpr (v2) {
            this->compute_v2<X,C,D>(x, ket, XCD, stream);
          }
          else {
            auto v0 = this->compute_v0<X,C,D>(x, ket, XCD, stream);
            if constexpr (!v0) {
              this->compute_v2<X,C,D>(x, ket, XCD, stream);
            }
          }
        }
      }
    );

  }

}
