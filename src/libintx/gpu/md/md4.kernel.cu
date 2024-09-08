// -*-c++-*-

#if !(defined(LIBINTX_GPU_MD_MD4_KERNEL_BRA) && defined(LIBINTX_GPU_MD_MD4_KERNEL_KET))
#error LIBINTX_GPU_MD_MD4_KERNEL_BRA/KET undefined
#endif

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/md4.kernel.h"
#include "libintx/gpu/boys.h"

#include "libintx/gpu/blas.h"
#include "libintx/utility.h"

#include "libintx/config.h"

#pragma nv_diag_suppress 2361

namespace libintx::gpu::md {

  constexpr int MaxShmem = LIBINTX_GPU_MAX_SHMEM;

  template
  void IntegralEngine<4>::compute<LIBINTX_GPU_MD_MD4_KERNEL_BRA,LIBINTX_GPU_MD_MD4_KERNEL_KET>(
    const Basis2&,
    const Basis2&,
    TensorRef<double,2>,
    gpuStream_t stream
  );

  template<int A, int B, int C, int D>
  auto IntegralEngine<4>::compute_v0(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    gpuStream_t stream)
  {

    using Bra = kernel::Basis2<A,B>;
    using Ket = kernel::Basis2<C,D>;

    constexpr int shmem = 0;
    constexpr int NAB = npure(A,B);

    //printf("IntegralEngine<4>::compute<%i,%i> bra.K=%i, ket.K=%i \n", Bra, Ket, bra.K, ket.K);
    Bra ab(bra.K, bra.N, bra.data, bra.k_stride, bra.pure_transform);
    Ket cd(ket.K, ket.N, ket.data, ket.k_stride, ket.pure_transform);

    using kernel_xy = typename kernel::find_if<
      (Bra::L+Ket::L <= 4 ? 900 : 800), MaxShmem, // 900 catches (pp|pp), (ds|pp)
      kernel::md4_ab_cd_kernel<Bra,Ket,16,4,1,MaxShmem/2>,
      kernel::md4_ab_cd_kernel<Bra,Ket,128,1,1,MaxShmem>
      >::type;

    using kernel_xz = typename kernel::find_if<
      800,MaxShmem,
      kernel::md4_ab_cd_kernel<Bra,Ket,32,1,4,MaxShmem>,
      kernel::md4_ab_cd_kernel<Bra,Ket,16,1,8,MaxShmem>,
      kernel::md4_ab_cd_kernel<Bra,Ket,8,1,16,MaxShmem>
      >::type;

    if constexpr (!std::is_same_v<kernel_xy,void>) {
      typename kernel_xy::ThreadBlock thread_block;
      static_assert(md::Basis2::alignment%thread_block.x == 0);
      dim3 grid = {
        (uint)(ab.N+thread_block.x-1)/thread_block.x,
        (uint)(cd.N+thread_block.z-1)/thread_block.z
      };
      TensorRef<double,5> ab_p(
        this->allocate<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x*bra.K),
        { thread_block.x, NAB, nherm2(A+B-1), grid.x, (size_t)bra.K }
      );
      // E(ab,p,ij) -> (ij,ab,p)
      for (int kab = 0; kab < bra.K; ++kab) {
        gpu::batch_transpose(
          NAB*nherm2(A+B-1), thread_block.x,
          ab.gdata(0,kab), ab.stride,
          &ab_p(0,0,0,0,kab), thread_block.x,
          grid.x,
          stream
        );
      }
      kernel::launch<<<grid,thread_block,shmem,stream>>>(
        kernel_xy(), ab, cd, gpu::boys(), std::tuple{ab_p}, ABCD
      );
      return std::true_type();
    }
    // performs poorly when Bra > 6
    else if constexpr (!std::is_same_v<kernel_xz,void> && Bra::L <= 6) {
      typename kernel_xz::ThreadBlock thread_block;
      dim3 grid = {
        (uint)(ab.N+thread_block.x-1)/thread_block.x,
        (uint)cd.N
      };
      TensorRef<double,4> ab_p(
        this->allocate<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x),
        { thread_block.x, NAB, nherm2(A+B-1), grid.x }
      );
      for (int kab = 0; kab < ab.K; ++kab) {
        gpu::batch_transpose(
          NAB*nherm2(A+B-1), thread_block.x,
          ab.gdata(0,kab), ab.stride,
          ab_p.data(), thread_block.x,
          grid.x,
          stream
        );
        kernel::launch<<<grid,thread_block,shmem,stream>>>(
          kernel_xz(), ab, kab, cd, gpu::boys(), std::tuple{ab_p}, ABCD
        );
      }
      return std::true_type();
    }
    else {
      return std::false_type();
    }

  }




  template<int A, int B, int C, int D>
  auto IntegralEngine<4>::compute_v1(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    gpuStream_t stream)
  {

    //return std::false_type();

    using kernel::Basis2;

    constexpr int Bra = A+B;
    constexpr int Ket = C+D;
    constexpr size_t NP = nherm2(Bra);
    constexpr size_t NQ = Ket ? nherm2(Ket-1) : 0;
    constexpr size_t NAB = npure(A,B);
    constexpr size_t NCD = npure(C,D);

    //printf("IntegralEngine<4>::compute<%i,%i> bra.K=%i, ket.K=%i \n", Bra, Ket, bra.K, ket.K);
    Basis2<Bra> ab(bra);
    Basis2<Ket> cd(ket);

    constexpr int DimX = 16;
    constexpr int DimY = 128/DimX;
    auto thread_block = gpu::thread_block<DimX,DimY>();

    using kernel0 = kernel::md4_v1_r1_p_cd_kernel<Basis2<A+B>, Basis2<C,D>, DimX,DimY, MaxShmem>;
    using kernel1 = kernel::md4_v1_ab_cd_kernel<Basis2<A,B>, void, DimX,DimY, MaxShmem>;

    constexpr bool viable = {
      kernel::test<kernel0>(800,MaxShmem) &&
      kernel::test<kernel1>(800,MaxShmem)
    };

    // printf(
    //   "IntegralEngine<4>::compute_v1 <%i,%i,%i,%i>: viable=(%i)\n",
    //   A, B, C, D, viable
    // );

    if constexpr (viable) {

      dim3 grid0 = {
        (uint)(ab.N+DimX-1)/DimX,
        (uint)(cd.N)
      };

      dim3 grid1 = {
        (uint)(ab.N+DimX-1)/DimX,
        (uint)(NCD*cd.N+DimY-1)/DimY
      };

      static_assert(md::Basis2::alignment%DimY == 0);
      size_t nkl_aligned = cd.N + DimY-cd.N%DimY;
      size_t p_cd_size = DimX*NP*NCD*nkl_aligned*grid0.x;

      int k_batch = ket.K;
      size_t r1_size = (grid0.x*grid0.y)*(DimX*nherm2(Bra+Ket));

      // printf(
      //   "IntegralEngine<4>::compute_v1<%i,%i,%i,%i>: K=%i, maxk=%i mem=%fGB\n",
      //   A, B, C, D,
      //   ket.K, maxk,
      //   (r1_size*maxk + p_cd_buffer.size())*(sizeof(double)/1e9)
      // );

      using p_cd_kernel = kernel::md4_v1_p_cd_kernel<Basis2<A+B>,Basis2<C,D>,DimX,DimY,1,MaxShmem>;

      if (kernel::test<p_cd_kernel>(800,MaxShmem)) {
        k_batch = 0;
      }
      else {
        if (Ket) {
          k_batch = std::min<int>(MaxShmem/sizeof(typename kernel0::Shmem), ket.K);
        }
        if (this->max_memory) {
          while (k_batch > 1) {
            size_t m = (r1_size*k_batch + p_cd_size);
            if (m*sizeof(double) <= this->max_memory) break;
            --k_batch;
          }
        }
      }

      size_t buffer0_size = p_cd_size;
      size_t buffer1_size = std::max<size_t>(
        {
          DimX*grid0.x*(1+nherm2(Bra-1)*NAB),
          r1_size*k_batch,
          bra.N*NP*NCD*ket.N,
        }
      );

      size_t min_memory = (buffer0_size + buffer1_size)*sizeof(double);

      if (this->max_memory && (min_memory > this->max_memory)) {
        throw gpu::runtime_error(
          str(
            "libintx.gpu.md4 requires at least ", min_memory, " bytes, "
            "max_memory=", this->max_memory, " bytes"
          )
        );
      }

      TensorRef<double,5> pCD{
        this->allocate<0>(buffer0_size),
        { DimX, NP, NCD, nkl_aligned, grid0.x }
      };

      double *buffer1 = this->allocate<1>(buffer1_size);

      for (int kp = 0; kp < bra.K; ++kp) {

        if constexpr (kernel::test<p_cd_kernel>(800,MaxShmem)) {
          kernel::launch<<<grid0,thread_block,0,stream>>>(
            p_cd_kernel(), ab, kp, Basis2<C,D>(cd), gpu::boys(), std::tuple{}, pCD
          );
        }
        else {
          for (int kq = 0; kq < ket.K; kq += k_batch) {
            int nk = std::min(ket.K-kq, k_batch);
            for (int k = 0; k < nk; ++k) {
              int ldR = (grid0.x*grid0.y)*(DimX*nherm2(Bra+Ket));
              kernel::compute_r1_kernel<DimX,DimY,0><<<grid0,thread_block,0,stream>>>(
                ab, cd, {kp,kq+k},
                gpu::boys(),
                TensorRef<double,4>{
                  buffer1+k*ldR,
                  { DimX, nherm2(Bra+Ket), grid0.x, grid0.y }
                }
              );
            }
            int shmem = nk*sizeof(typename kernel0::Shmem);
            kernel::launch<<<grid0,thread_block,shmem,stream>>>(
              kernel0(),
              ab,
              kernel::Basis2<C,D>(cd), kq, nk,
              TensorRef<double,5>{
                buffer1,
                { DimX, nherm2(Bra+Ket), grid0.x, grid0.y, (uint)nk }
              },
              pCD
            );
          }
        } // else

        TensorRef<double,3> batched_ab_p{
          buffer1,
          { DimX, 1+nherm2(Bra-1)*NAB, grid1.x }
        };
        static_assert(md::Basis2::alignment%DimX == 0);
        gpu::batch_transpose(
          (1+nherm2(Bra-1)*NAB), DimX,
          ab.gdata(0,kp)-1, ab.stride,
          batched_ab_p.data(), DimX,
          grid1.x,
          stream
        );

        kernel::launch<<<grid1,thread_block,0,stream>>>(
          kernel1(),
          bra.N, NCD*cd.N,
          batched_ab_p,
          pCD.reshape(DimX, NP, NCD*nkl_aligned, grid1.x),
          (kp == 0 ? 0.0 : 1.0),
          ABCD,
          bra.pure_transform
        );

      } // kp

      return std::true_type{};

    } // viable
    else {
      return std::false_type{};
    }

  }


  template<int A, int B, int C, int D>
  auto IntegralEngine<4>::compute_v2(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    gpuStream_t stream)
  {
    //printf("IntegralEngine<4>::compute_v2<%i,%i,%i,%i>\n", A,B,C,D);
    using kernel::Basis2;

    Basis2<A+B> ab(bra);
    Basis2<C+D> cd(ket);

    constexpr uint NP = ab.nherm;
    constexpr uint NQ = cd.nherm;
    const uint NAB = ab.nbf;
    const uint NCD = cd.nbf;

    //assert(cd.nbf*cd.N <= ldV);
    dim3 grid = { (uint)ab.N, (uint)cd.N };

    using md_v2_p_cd_kernel = kernel::md4_v2_p_cd_kernel<
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
        assert(kcd_batch);
        // [p,cd,kl,ij]
        typename md_v2_p_cd_kernel::ThreadBlock thread_block;
        kernel::launch<<<grid,thread_block,kcd_batch*dynamic_shmem,stream>>>(
          md_v2_p_cd_kernel(),
          ab, kab, Basis2<C,D>(cd), kcd_batch, gpu::boys(),
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
      gpu::transpose(
        NAB*NCD*cd.N, ab.N,
        buffer1, NAB*NCD*cd.N,
        ABCD.data(), ab.N,
        stream
      );
    }

    else {


      using Block = gpu::thread_block<32,4>;

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
          kernel::compute_p_q_kernel<Block::x,Block::y,2><<<grid,Block{},0,stream>>>(
            ab, cd, {kab,kcd},
            gpu::boys(),
            TensorRef<double,4>{pq, { NP, grid.x, NQ, grid.y}}
          );
          // [p,ij,q,kl]*H(cd,q,kl)' -> [p,ij,cd,kl]
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
        gpu::transpose(
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
      gpu::transpose(
        NAB*NCD*cd.N, ab.N,
        ab_cd_transpose, NAB*NCD*cd.N,
        ABCD.data(), ab.N,
        stream
      );

    }

  }

  template<int Bra, int Ket>
  void IntegralEngine<4>::compute(
    const Basis2& bra,
    const Basis2& ket,
    TensorRef<double,2> ABCD,
    gpuStream_t stream)
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
            auto v1 = this->compute_v1<A,B,C,D>(bra, ket, ABCD, stream);
            if constexpr (!v1) {
              this->compute_v2<A,B,C,D>(bra, ket, ABCD, stream);
            }
          }

        }
      }
    );

  }

}
