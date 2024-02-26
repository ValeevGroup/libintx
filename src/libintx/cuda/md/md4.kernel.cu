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
        this->buffer<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x*bra.K),
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
        this->buffer<0>(thread_block.x*NAB*nherm2(A+B-1)*grid.x),
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

    kernel::Basis2<A+B> ab(bra.first, bra.second, bra.K, bra.N, bra.data, bra.k_stride);
    kernel::Basis2<C+D> cd(ket.first, ket.second, ket.K, ket.N, ket.data, ket.k_stride);

    constexpr int NP = ab.nherm;
    constexpr int NQ = cd.nherm;
    const int NAB = ab.nbf;
    const int NCD = cd.nbf;

    //assert(cd.nbf*cd.N <= ldV);
    dim3 grid = { (uint)ab.N, (uint)cd.N };
    using Block = cuda::thread_block<32,4>;

    auto *pq = this->buffer<0>(
      std::max(
        (grid.x*grid.y)*(ab.nherm*cd.nherm),
        (grid.x*grid.y)*(ab.nbf*cd.nherm)
      )
    );
    auto *abq = this->buffer<1>((grid.x*grid.y)*(ab.nbf*cd.nherm));
    auto *abq_transpose = pq;

    for (int kcd = 0; kcd < ket.K; ++kcd) {
      for (int kab = 0; kab < bra.K; ++kab) {
        //double C = 2*std::pow(M_PI,2.5);
        //double Ck = (kab == 0 ? 0 : 1.0);
        // [p,q,kl,ij]
        kernel::compute_p_q<Block,2><<<grid,Block{},0,stream>>>(
          ab, cd, {kab,kcd},
          cuda::boys(),
          TensorRef<double,4>{pq, { NP, NQ, grid.y, grid.x}}
        );
        // [ab,q,kl,ij] = H(ab,p,ij)*[p,q,kl,ij]
        batch_gemm<ColumnMajor, ColumnMajor, ColumnMajor>(
          NAB, cd.N*cd.nherm, NP,
          1.0,
          ab.gdata(0,kab), NAB, ab.stride,
          pq, NP, NP*NQ*cd.N,
          (kab == 0 ? 0.0 : 1.0), // beta
          abq, NAB, NAB*NQ*cd.N,
          ab.N, // batches
          stream
        );
        // cuda::stream::synchronize(stream);
        // for (size_t i = 0; i < NAB; ++i) {
        //   printf("%f\n", ABCD(i,0));
        // }
        // return;
      } // kab
      // [ij,ab,q,kl] = [ab,q,kl,ij]
      cuda::transpose(
        ab.nbf*cd.nherm*cd.N, ab.N,
        abq, ab.nbf*cd.nherm*cd.N,
        abq_transpose, ab.N,
        stream
      );
      // [ij,ab|cd,kl] = [ij,ab|q,kl]*H(cd,q,kl)'
      batch_gemm<ColumnMajor, RowMajor, ColumnMajor>(
        NAB*ab.N, NCD, cd.nherm,
        1.0, // alpha
        abq_transpose, NAB*ab.N, NAB*ab.N*NQ,
        cd.gdata(0,kcd), NCD, cd.stride,
        (kcd == 0 ? 0.0 : 1.0), // beta
        ABCD.data(), NAB*ab.N, NAB*ab.N*NCD,
        cd.N, // batches
        stream
      );
    } // kcd
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
