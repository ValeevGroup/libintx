// -*-c++-*-

#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/md4.h"
#include "libintx/cuda/md/md4.kernel.h"

#include "libintx/cuda/api/kernel.h"
#include "libintx/cuda/blas.h"
#include "libintx/boys/cuda/chebyshev.h"
#include "libintx/utility.h"

#include "libintx/config.h"

#pragma nv_diag_suppress 2361

namespace libintx::cuda::md {

#ifdef LIBINTX_CUDA_MD_MD4_KERNEL_BRA_KET

  template
  void ERI4::compute<LIBINTX_CUDA_MD_MD4_KERNEL_BRA_KET>(
    const Basis2&,
    const Basis2&,
    TensorRef<double,2>,
    cudaStream_t stream
  );

#endif
  template<int A, int B, int C, int D>
  auto ERI4::compute_v0(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    cudaStream_t stream)
  {

    using Bra = kernel::Basis2<A,B>;
    using Ket = kernel::Basis2<C,D>;
    constexpr int shmem = 0;
    constexpr int NAB = npure(A,B);

    //printf("ERI4::compute<%i,%i> bra.K=%i, ket.K=%i \n", Bra, Ket, bra.K, ket.K);
    Bra ab(bra.K, bra.N, bra.data, bra.k_stride, bra.pure_transform);
    Ket cd(ket.K, ket.N, ket.data, ket.k_stride, ket.pure_transform);

    using kernel_xz = kernel::md_v0_kernel<Bra,Ket,16,1,4,MaxShmem>;

    using kernel_xy = typename kernel::find_if<
      800,MaxShmem,
      kernel::md_v0_kernel<Bra,Ket,32,4,1,MaxShmem>,
      kernel::md_v0_kernel<Bra,Ket,16,8,1,MaxShmem>,
      kernel::md_v0_kernel<Bra,Ket,8,16,1,MaxShmem>
      >::type;

    if constexpr (kernel::test<kernel_xz>(900,MaxShmem)) {
      typename kernel_xz::ThreadBlock thread_block;
      static_assert(bra.alignment%thread_block.x == 0);
      dim3 grid = {
        (uint)(ab.N+thread_block.x-1)/thread_block.x,
        (uint)(cd.N+thread_block.z-1)/thread_block.z
      };
      TensorRef<double,5> ab_p(
        this->allocate<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x*bra.K),
        { thread_block.x, NAB, nherm2(A+B-1), grid.x, (size_t)bra.K }
      );
      for (int kab = 0; kab < bra.K; ++kab) {
        cuda::batch_transpose(
          NAB*nherm2(A+B-1), thread_block.x,
          ab.gdata(0,kab), ab.stride,
          &ab_p(0,0,0,0,kab), thread_block.x,
          grid.x,
          stream
        );
      }
      launch<<<grid,thread_block,shmem,stream>>>(
        kernel_xz(), ab, cd, cuda::boys(), std::tuple{ab_p}, ABCD
      );
      //printf("v0:xz\n");
      return std::true_type();
    }
    // performs poorly when Bra > 6
    else if constexpr (!std::is_same_v<kernel_xy,void> && (A+B <= 6)) {
      typename kernel_xy::ThreadBlock thread_block;
      dim3 grid = {
        (uint)(ab.N+thread_block.x-1)/thread_block.x,
        (uint)cd.N
      };
      TensorRef<double,4> ab_p(
        this->allocate<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x),
        { thread_block.x, NAB, nherm2(A+B-1), grid.x }
      );
      for (int kab = 0; kab < ab.K; ++kab) {
        cuda::batch_transpose(
          NAB*nherm2(A+B-1), thread_block.x,
          ab.gdata(0,kab), ab.stride,
          ab_p.data(), thread_block.x,
          grid.x,
          stream
        );
        launch<<<grid,thread_block,shmem,stream>>>(
          kernel_xy(), ab, kab, cd, cuda::boys(), std::tuple{ab_p}, ABCD
        );
      }
      //printf("v0:xy ncd_batch=%i\n", kernel_xy::ncd_batch);
      return std::true_type();
    }
    else {
      return std::false_type();
    }

  }



  template<int A, int B, int C, int D>
  auto ERI4::compute_v2(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    cudaStream_t stream)
  {
    //printf("ERI4::compute_v2<%i,%i,%i,%i>\n", A,B,C,D);
    using kernel::Basis2;

    Basis2<A+B> ab(bra.first, bra.second, bra.K, bra.N, bra.data, bra.k_stride);
    Basis2<C+D> cd(ket.first, ket.second, ket.K, ket.N, ket.data, ket.k_stride);

    constexpr uint NP = ab.nherm;
    constexpr uint NQ = cd.nherm;
    const uint NAB = ab.nbf;
    const uint NCD = cd.nbf;

    //assert(cd.nbf*cd.N <= ldV);
    dim3 grid = { (uint)ab.N, (uint)cd.N };

    using md_v2_p_cd_kernel = kernel::md_v2_p_cd_kernel<
      Basis2<A+B>, Basis2<C,D>, 128, MaxShmem>;

    if constexpr (kernel::test<md_v2_p_cd_kernel>(800,MaxShmem)) {

      auto *buffer0 = this->allocate<0>(NP*NCD*(grid.x*grid.y));
      auto *buffer1 = this->allocate<1>(NAB*NCD*(grid.x*grid.y));

      for (int kab = 0; kab < bra.K; ++kab) {
        int kcd_batch = ket.K;
        size_t dynamic_shmem = sizeof(typename md_v2_p_cd_kernel::Shmem::Dynamic);
        size_t static_shmem = sizeof(typename md_v2_p_cd_kernel::Shmem::Static);
        while (static_shmem + kcd_batch*dynamic_shmem > MaxShmem) {
          --kcd_batch;
        }
        // [p,cd,kl,ij]
        typename md_v2_p_cd_kernel::ThreadBlock thread_block;
        kernel::launch<<<grid,thread_block,kcd_batch*dynamic_shmem,stream>>>(
          md_v2_p_cd_kernel(),
          ab, kab, Basis2<C,D>(cd), kcd_batch, cuda::boys(),
          TensorRef<double,4>{ buffer0, { NP, NCD, grid.y, grid.x } }
        );
        // H(ab,p,ij)*[p,cd,kl,ij] -> [ab,cd,kl,ij]
        batch_gemm<ColumnMajor, ColumnMajor, ColumnMajor>(
          NAB, NCD*cd.N, NP,
          1.0,
          ab.gdata(0,kab), NAB, ab.stride,
          buffer0, NP, NP*NCD*cd.N,
          (kab == 0 ? 0.0 : 1.0), // beta
          buffer1, NAB, NAB*NCD*cd.N,
          ab.N, // batches
          stream
        );

      }

      // [ij,ab,cd,kl] = [ab,cd,kl,ij]
      cuda::transpose(
        NAB*NCD*cd.N, ab.N,
        buffer1, NAB*NCD*cd.N,
        ABCD.data(), ab.N,
        stream
      );
    }

    else {

      using Block = cuda::thread_block<32,4>;

      auto *pq = this->allocate<0>(
        std::max(
          (grid.x*grid.y)*(NQ*NP),
          (grid.x*grid.y)*(NQ*NAB)
        )
      );
      auto *p_cd = this->allocate<1>((grid.x*grid.y)*(NP*NCD));
      auto *p_cd_transpose = pq;
      auto *ab_cd_transpose = this->allocate<2>((grid.x*grid.y)*(NAB*NCD));

      for (int kab = 0; kab < bra.K; ++kab) {
        for (int kcd = 0; kcd < ket.K; ++kcd) {
          // [p,ij,q,kl]
          kernel::compute_p_q<Block,2><<<grid,Block{},0,stream>>>(
            ab, cd, {kab,kcd},
            cuda::boys(),
            TensorRef<double,4>{pq, { NP, grid.x, NQ, grid.y}}
          );
          // [p,ij,q,kl]*H(cd,q,ij)' -> [p,ij,cd,kl]
          batch_gemm<ColumnMajor, RowMajor, ColumnMajor>(
            NP*ab.N, NCD, NQ,
            1.0, // alpha
            pq, NP*ab.N, NP*ab.N*NQ,
            cd.gdata(0,kcd), NCD, cd.stride,
            (kcd == 0 ? 0.0 : 1.0), // beta
            p_cd, NP*ab.N, NP*ab.N*NCD,
            cd.N, // batches
            stream
          );
        }
        // [p,ij,cd,kl] -> [cd,kl,p,ij]
        cuda::transpose(
          NP*ab.N, NCD*cd.N,
          p_cd, NP*ab.N,
          p_cd_transpose, NCD*cd.N,
          stream
        );
        // H(ab,p,ij)*[cd,kl,p,ij]' -> [ab,cd,kl,ij]
        batch_gemm<ColumnMajor, RowMajor, ColumnMajor>(
          NAB, NCD*cd.N, NP,
          1.0,
          ab.gdata(0,kab), NAB, ab.stride,
          p_cd_transpose, NCD*cd.N, NCD*cd.N*NP,
          (kab == 0 ? 0.0 : 1.0), // beta
          ab_cd_transpose, NAB, NAB*NCD*cd.N,
          ab.N, // batches
          stream
        );
      } // kcd

      // [ab,cd,kl,ij] -> [ij,ab,cd,kl]
      cuda::transpose(
        NAB*NCD*cd.N, ab.N,
        ab_cd_transpose, NAB*NCD*cd.N,
        ABCD.data(), ab.N,
        stream
      );

    }

  }

  template<int Bra, int Ket>
  void ERI4::compute(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    cudaStream_t stream)
  {

    constexpr size_t NP = nherm2(Bra);
    constexpr size_t NQ = Ket ? nherm2(Ket-1) : 0;

    foreach2(
      std::make_index_sequence<Bra+1>{},
      std::make_index_sequence<Ket+1>{},
      [&](auto A, auto C) {

        constexpr int B = Bra-A;
        constexpr int D = Ket-C;

        if constexpr (std::max<int>({A,B,C,D}) <= LMAX) {

          if (A != bra.first.L || B != bra.second.L) return;
          if (C != ket.first.L || D != ket.second.L) return;

          auto v0 = this->compute_v0<A,B,C,D>(bra, ket, ABCD, stream);
          if constexpr (!v0) {
            this->compute_v2<A,B,C,D>(bra, ket, ABCD, stream);
          }

        }
      }
    );

  }

}
