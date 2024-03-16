#ifndef LIBINTX_CUDA_MD_MD_KERNEL_H
#define LIBINTX_CUDA_MD_MD_KERNEL_H

#include "libintx/cuda/forward.h"
#include "libintx/cuda/md/basis.h"
#include "libintx/engine/md/r1.h"
#include "libintx/engine/md/hermite.h"

#include "libintx/cuda/api/thread_group.h"

#include "libintx/config.h"
#include "libintx/math.h"
#include "libintx/utility.h"

namespace libintx::cuda::md::kernel {

  namespace cart = libintx::cartesian;
  namespace herm = libintx::hermite;

  using libintx::md::hermite_to_cartesian;
  using libintx::md::hermite_to_pure;
  using libintx::pure::cartesian_to_pure;

  template<int X>
  struct Basis1 {
    static constexpr int Centers = 1;
    static constexpr int L = X;
    static constexpr int nherm = nherm1(L);
    static constexpr int nbf = npure(L);
    const int K, N;
    const Hermite *data;
    LIBINTX_GPU_ENABLED
    const Hermite* hdata(int idx, int k) const {
      return data + idx + k*N;
    }
  };

  template<int ... Args>
  struct Basis2;

  template<int AB>
  struct Basis2<AB> {
    static constexpr int Centers = 2;
    static constexpr int L = AB;
    static constexpr int nherm = nherm2(L);
    const Shell first, second;
    const int nbf;
    const int K;
    const int N;
    const double *data;
    const int stride;
    const size_t k_stride;
    const double *pure_transform;
    explicit Basis2(const md::Basis2 &basis)
      : first(basis.first), second(basis.second),
        nbf(libintx::nbf(first)*libintx::nbf(second)),
        K(basis.K), N(basis.N),
        data(basis.data),
        stride(sizeof(Hermite)/sizeof(double) + nherm*nbf),
        k_stride(basis.k_stride),
        pure_transform(basis.pure_transform)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int ij, int k) const {
      return reinterpret_cast<const Hermite*>(data + ij*stride + k*k_stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int ij, int k) const {
      return reinterpret_cast<const double*>(hdata(ij,k)+1);
    }
  };

  template<int _A, int _B>
  struct Basis2<_A,_B> {
    static constexpr int Centers = 2;
    static constexpr int First = _A;
    static constexpr int Second = _B;
    static constexpr int L = _A + _B;
    static constexpr int nherm = nherm2(L);
    static constexpr int nbf = npure(_A)*npure(_B);
    static constexpr int stride = sizeof(Hermite)/sizeof(double) + nherm*nbf;
    const int K;
    const int N;
    const double *data;
    const size_t k_stride;
    const double *pure_transform;
    Basis2(int K, int N, const double *H, size_t k_stride, const double *pure_transform)
      : K(K), N(N), data(H), k_stride(k_stride),
        pure_transform(pure_transform)
    {
    }
    explicit Basis2(const Basis2<L> &basis)
      : K(basis.K), N(basis.N), data(basis.data), k_stride(basis.k_stride)
    {
    }
    LIBINTX_GPU_ENABLED
    auto hdata(int ij, int k = 0) const {
      return reinterpret_cast<const Hermite*>(data + ij*stride + k*k_stride);
    }
    LIBINTX_GPU_ENABLED
    auto gdata(int ij, int k = 0) const {
      return reinterpret_cast<const double*>(hdata(ij,k)+1);
    }
  };


  LIBINTX_GPU_DEVICE
  constexpr auto orbitals2 = hermite::orbitals<2*LMAX>;

  LIBINTX_GPU_DEVICE
  constexpr auto orbitals1 = std::tuple{
    hermite::orbitals1<XMAX,0>,
    hermite::orbitals1<XMAX,1>
  };

  template<int ... Args>
  constexpr auto& orbitals(const Basis2<Args...>&) {
    static_assert(orbitals2.size());
    return orbitals2;
  }

  template<int X>
  constexpr auto& orbitals(const Basis1<X>&) {
    return std::get<X%2>(orbitals1);
  }

  // use unrolled hermite to pure code or matrix one
  inline
  constexpr bool hermite_to_pure_too_complicated(int A, int B) {
    // [ab| > [gd|
    return (std::max(A,B) >= 4 && std::min(A,B) >= 2);
  }


  template<
    typename Bra, typename Ket,
    int DimX, int DimY,
    int MaxShmem, int MinBlocks = 2
    >
  struct md_v0_kernel  {

    static_assert(DimX <= 32 || DimY == 1);

    static constexpr int L = (Bra::L+Ket::L);
    static constexpr int NP = Bra::nherm;
    static constexpr int NQ = nherm2(Ket::L-1);
    static constexpr int NAB = Bra::nbf;
    static constexpr int NCD = Ket::nbf;

    using ThreadBlock = cuda::thread_block<DimX,DimY>;
    static constexpr int num_threads = ThreadBlock::size();
    static constexpr int max_shmem = MaxShmem;
    static constexpr int min_blocks = MinBlocks;

    struct Registers {
      double pCD[NP][NCD];
      double r[Ket::L ? nherm2(L-1) : 1]; // excluded from reg count
    };

    union Shmem {
      struct {
        Hermite abs[DimX];
        struct {
          Hermite hdata;
          double gdata[max(1,NQ*NCD)];
        } cds[DimY];
      };
    };

    template<typename ... Args>
    __device__
    LIBINTX_GPU_FORCEINLINE
    void operator()(
      const Bra &bra,
      const Ket &ket,
      const auto &boys,
      std::tuple<Args...> &args,
      auto &BraKet) const
    {

      using hermite::index2;

      auto &p_orbitals = orbitals(bra);
      auto &q_orbitals = orbitals(ket);

      static constexpr int C = ket.First;
      static constexpr int D = ket.Second;

      constexpr ThreadBlock thread_block;
      const auto& thread_rank = thread_block.thread_rank();

      auto gx = [&](){
        if constexpr (DimX == thread_block.size()) {
          return thread_block;
        }
        else {
          return tiled_partition<DimX>(this_thread_block());
        }
      }();

      // half-warp
      auto hw = tiled_partition<16>(this_thread_block());

      __shared__ Shmem shmem;

      int ij = blockIdx.x*thread_block.x;
      int kl = blockIdx.y*thread_block.y;

      libintx::cuda::memset1(&shmem.cds[threadIdx.y].hdata, 0, gx);
      memset1(&shmem.abs, 0, thread_block);
      //static_assert(sizeof(shmem.abs) == sizeof(Hermite)*DimX);
      // for (int ix = threadIdx.y; ix < DimX; ix += DimY) {
      //   //if (ix+ij < bra.N) continue;
      // }

      [[maybe_unused]]
      double sscd[NCD] = {};

      for (int kab = 0; kab < bra.K; ++kab) {

        thread_block.sync();
        for (int ix = thread_rank/hw.size(); ix < min(DimX,bra.N-ij); ix += thread_block.size()/hw.size()) {
          memcpy1(bra.hdata(ix+ij,kab), &shmem.abs[ix], hw);
        }
        thread_block.sync();

        decltype (Registers::pCD) pCD = {};
        for (int kcd = 0; kcd < ket.K; ++kcd) {

          thread_block.sync();
          if (kl+threadIdx.y < ket.N) {
            memcpy(
              nwords<sizeof(double),Hermite>() + NQ*NCD,
              reinterpret_cast<const double*>(ket.hdata(kl+threadIdx.y,kcd)),
              reinterpret_cast<double*>(&shmem.cds[threadIdx.y].hdata),
              gx
            );
          }
          thread_block.sync();

          const auto &ab = shmem.abs[threadIdx.x];
          const auto &cd = shmem.cds[threadIdx.y].hdata;

          auto &P = ab.r;
          auto &Q = cd.r;
          auto PQ = P-Q;

          double Ck = ab.C*cd.C;
          if (!Ck) continue;

          double p = ab.exp;
          double q = cd.exp;
          double pq = p*q;
          double alpha = pq/(p+q);
          double T = alpha*norm(P,Q);

          double s[L+1] = {};
          boys.template compute<L>(T, 0, s);
          Ck *= rsqrt(pq*pq*(p+q));
          //double Kab = exp(-(a*b)/p*norm(P));
          //double Kcd = 1;//exp(-norm(Q));
          //C *= Kcd;
          for (int i = 0; i <= L; ++i) {
            //printf("s[%i]=%f\n", i, s[i]);
            s[i] *= math::sqrt_4_pi5*Ck;
            Ck *= -2*alpha;
          }

          namespace r1 = libintx::md::r1;
          auto &Ecd = shmem.cds[threadIdx.y].gdata;

          double r[nherm2(L)] = {};
          r1::compute<L>(PQ, s, r);

#pragma unroll
          for (int ip = 0; ip < NP; ++ip) {
            auto p = p_orbitals[ip];
            hermite_to_pure<C,D>(
              [&](auto c, auto d, auto u) {
                constexpr int phase = ((C+D)%2 == 0 ? +1 : -1);
                pCD[ip][index(c) + index(d)*npure(C)] += phase*u*cd.inv_2_exp;
              },
              [&](auto &&q) {
                return r[index2(p+q)];
              }
            );
          }

#pragma unroll
          for (int ip = 0; ip < NP; ++ip) {
            auto p = p_orbitals[ip];
#pragma unroll
            for (int iq = 0; iq < NQ; ++iq) {
              auto q = q_orbitals[iq];
              int phase = (q.L()%2 == 0 ? +1 : -1);
              double pq = phase*r[index2(p+q)];
#pragma unroll
              for (int icd = 0; icd < NCD; ++icd) {
                pCD[ip][icd] += pq*Ecd[icd + iq*NCD];
              }
            }

          }

        } // kcd

        if (!(threadIdx.x+ij < bra.N && kl+threadIdx.y < ket.N)) continue;

        if constexpr (Bra::L == 0) {
          for (int icd = 0; icd < NCD; ++icd) {
            sscd[icd] += pCD[0][icd];
          }
        }
        else if constexpr (Bra::Centers == 1) {

          static constexpr int X = bra.L;
          double inv_2_p = shmem.abs[threadIdx.x].inv_2_exp;

          for (int icd = 0; icd < NCD; ++icd) {

            double xcd[npure(X)] = {};
            if (kab) {
              for (int i = 0; i < npure(X); ++i) {
                xcd[i] = BraKet(
                  threadIdx.x + ij + i*bra.N,
                  icd + (threadIdx.y + kl)*NCD
                );
              }
            }

            hermite_to_cartesian<X>(
              inv_2_p,
              [&](auto p) -> const double& { return pCD[herm::index1(p)][icd]; },
              [&](auto p) -> double& { return pCD[herm::index1(p)][icd]; }
            );

            cartesian_to_pure<X>(
              [&](auto x, auto u) {
                BraKet(
                  threadIdx.x + ij + index(x)*bra.N,
                  icd + (threadIdx.y + kl)*NCD
                ) = u + xcd[index(x)];
              },
              [&](auto x) {
                return pCD[herm::index1(x)][icd];
              }
            );

          }

        }
        else if constexpr (Bra::Centers == 2) {

          static constexpr int A = bra.First;
          static constexpr int B = bra.Second;

          auto&& [Eab] = args;

          double inv_2_p = shmem.abs[threadIdx.x].inv_2_exp;

          for (int iab = 0; iab < npure(A)*npure(B); ++iab) {

            double abcd[NCD] = {};

            if (kab) {
#pragma unroll
              for (int icd = 0; icd < NCD; ++icd) {
                abcd[icd] = BraKet(
                  (threadIdx.x+ij) + iab*bra.N,
                  icd + (threadIdx.y+kl)*NCD
                );
              }
            }

#pragma unroll
            for (int ip = 0; ip < nherm2(A+B-1); ++ip) {
              double E = Eab(threadIdx.x, iab, ip, blockIdx.x, kab);
#pragma unroll
              for (int icd = 0; icd < NCD; ++icd) {
                abcd[icd] += pCD[ip][icd]*E;
              }
            }

#pragma unroll
            for (int ip = 0; ip < ncart(A+B); ++ip) {
              double Eab = inv_2_p*bra.pure_transform[iab+ip*NAB];
#pragma unroll
              for (int icd = 0; icd < NCD; ++icd) {
                abcd[icd] += pCD[ip+nherm2(A+B-1)][icd]*Eab;
              }
            }

#pragma unroll
            for (int icd = 0; icd < NCD; ++icd) {
              BraKet(
                (threadIdx.x+ij) + iab*bra.N,
                icd + (threadIdx.y+kl)*NCD
              ) = abcd[icd];
            }
          }

        }

      } // kab

      if constexpr (Bra::L == 0) {
        if (!(threadIdx.x+ij < bra.N && kl+threadIdx.y < ket.N)) return;
        assert(threadIdx.y+kl < ket.N);
#pragma unroll
        for (int icd = 0; icd < NCD; ++icd) {
          BraKet(
            (threadIdx.x+ij),
            icd + (threadIdx.y+kl)*NCD
          ) = sscd[icd];
        }
      }

    }

  };


  template<
    int Transform,
    typename Bra, typename Ket,
    int DimX, int DimY, int DimZ,
    int MaxShmem,
    int MinBlocks = 2
    >
  struct md_x_cd_kernel {

    static_assert(!Transform || Transform == Bra::Centers);
    static_assert(DimY != 1);
    static_assert(DimZ == 1);

    static constexpr int L = (Bra::L+Ket::L);
    static constexpr int NP = Bra::nherm;
    static constexpr int NQ = nherm2(Ket::L-1);
    static constexpr int C = Ket::First;
    static constexpr int D = Ket::Second;
    static constexpr int NCD = Ket::nbf;

    using ThreadBlock = cuda::thread_block<DimX,DimY,1>;
    static constexpr int num_threads = ThreadBlock::size();
    static constexpr int max_shmem = MaxShmem;
    static constexpr int min_blocks = MinBlocks;

    static constexpr int np_batch = (NP+DimY-1)/DimY;
    static constexpr int ncd_batch = []() {
      for (int i = 1; i < NCD; ++i) {
        int n = (NCD+i-1)/i;
        if (n*NP*DimX*sizeof(double) <= MaxShmem) return n;
      }
      return 1;
    }();

    union Registers {
      struct {
        double V[np_batch][NCD];
        //double r[ncart(C+D)];
      };
    };

    union Shmem {
      struct {
        Hermite Hx[DimX];
        double R[nherm2(L)][DimX];
        Hermite cd;
        double Ecd[NCD*NQ];
      };
      // if Transform == 0 this member isn't needed;
      // then P[NP][1][DimX] =< R[nherm2(L)][DimX];
      struct {
        double inv_2_p[DimX];
        double P[NP][Transform ? ncd_batch : 1][DimX];
        //double Eab[DimY][DimX];
      };
    };

    __device__
    LIBINTX_GPU_FORCEINLINE
    void operator()(
      const Bra &bra, const int &kab,
      const Ket &ket,
      const auto &boys,
      auto &&args,
      auto &&BraKet) const
    {

      auto &p_orbitals = orbitals(bra);
      auto &q_orbitals = orbitals(ket);

      __shared__ Shmem shmem;

      constexpr ThreadBlock thread_block;
      const int thread_rank = thread_block.thread_rank();
      constexpr auto warp = this_warp();

      int ij = blockIdx.x*DimX;
      int kl = blockIdx.y;

      for (int ix = thread_rank/warp.size(); ix < DimX; ix += num_threads/warp.size()) {
        if (ix+ij < bra.N) {
          memcpy1(bra.hdata(ix+ij,kab), &shmem.Hx[ix], warp);
        }
        else {
          memset1(&shmem.Hx[ix], 0, warp);
          if (warp.thread_rank() == 0) shmem.Hx[ix].exp = 1;
        }
      }

      decltype(Registers::V) V = {};

      for (int kcd = 0; kcd < ket.K; ++kcd) {

        thread_block.sync();

        memcpy(
          nwords<sizeof(double),Hermite>(),
          reinterpret_cast<const double*>(ket.hdata(kl,kcd)),
          reinterpret_cast<double*>(&shmem.cd),
          thread_block
        );

        thread_block.sync();

        //if (!shmem.cd.C) continue;

        const auto &ab = shmem.Hx[threadIdx.x];
        const auto &cd = shmem.cd;

        auto &P = ab.r;
        auto &Q = cd.r;

        auto &R = shmem.R;
        auto& Ecd = shmem.Ecd;

        if (threadIdx.y == 0) {
          double p = ab.exp;
          double q = cd.exp;
          double pq = p*q;
          double alpha = pq/(p+q);
          double Ck = ab.C*cd.C;
          Ck *= rsqrt(pq*pq*(p+q));
          double T = alpha*norm(P,Q);
          double s[L+1];
          boys.template compute<L>(T,0,s);
          for (int i = 0; i <= L; ++i) {
            s[i] = math::sqrt_4_pi5*Ck*s[i];
            Ck *= -2*alpha;
          }
          auto PQ = P-Q;
          namespace r1 = libintx::md::r1;
          r1::visit<L,r1::DepthFirst>(
            [&](auto &&r) {
              R[r.index][threadIdx.x] = r.value;
            },
            PQ, s
          );
        }
        else { // (threadIdx.y == 0)
          int xy = thread_block.thread_rank() - DimX;
          for (int i = xy; i < NQ*NCD; i += num_threads-DimX) {
            Ecd[i] = ket.gdata(kl,kcd)[i];
          }
        }
        thread_block.sync();

        if (threadIdx.x+ij >= bra.N) continue;

        auto f = [&](auto &&iy) {

          int ip = iy*DimY+threadIdx.y;
          if (ip >= NP) return;
          auto p = p_orbitals[ip];

#pragma unroll
          for (int iq = 0; iq < NQ; ++iq) {
            auto q = q_orbitals[iq];
            int phase = (q.L()%2 == 0 ?  +1 : -1);
            double pq = phase*R[herm::index2(p+q)][threadIdx.x];
#pragma unroll
            for (int icd = 0; icd < NCD; ++icd) {
              V[iy][icd] += pq*Ecd[icd + iq*NCD];
            }
          }

          constexpr int phase = ((C+D)%2 == 0 ?  +1 : -1);
          if constexpr (!hermite_to_pure_too_complicated(C,D)) {
            hermite_to_pure<C,D>(
              [&](auto &&c, auto &&d, auto u) {
                V[iy][index(c) + index(d)*npure(C)] += phase*u*cd.inv_2_exp;
              },
              [&](auto &&q) {
                return R[herm::index2(p+q)][threadIdx.x];
              }
            ) ;
          }
          else {
#pragma unroll
            for (int iq = 0; iq < ncart(C+D); ++iq) {
              auto q = q_orbitals[NQ+iq];
              auto pq = phase*cd.inv_2_exp*R[herm::index2(p+q)][threadIdx.x];
#pragma unroll
            for (int icd = 0; icd < NCD; ++icd) {
                V[iy][icd] += pq*ket.pure_transform[icd+iq*NCD];
              }
            }
          } // hermite_to_pure_unroll(C,D)

        };

#pragma unroll
        for (int ip = 0; ip < NP; ip += DimY) {
          f(ip/DimY);
        }

      } // kcd

      if constexpr (!Transform) {

#pragma unroll
        for (int ip = 0; ip < NP; ip += DimY) {
          if (ip+threadIdx.y >= NP) break;
          for (int icd = 0; icd < NCD; ++icd) {
            //printf("
            BraKet(threadIdx.x, ip+threadIdx.y, icd, kl, blockIdx.x) = V[ip/DimY][icd];
          }
        }

      }

      else {

      thread_block.sync();

      double inv_2_p = shmem.Hx[threadIdx.x].inv_2_exp;
      auto &P = shmem.P;

#pragma unroll
      for (int icd = 0; icd < NCD; icd += ncd_batch) {

        thread_block.sync();
#pragma unroll
        for (int iy = 0; iy < NP; iy += DimY) {
          int ip = iy+threadIdx.y;
         if (ip >= NP) break;
#pragma unroll
          for (int i = 0; i < ncd_batch; ++i) {
            if (icd+i >= NCD) break;
            P[ip][i][threadIdx.x] = V[iy/DimY][icd+i];
          }
        }
        thread_block.sync();

        if (ij + threadIdx.x >= bra.N) continue;

        if constexpr (Bra::Centers == 1) {
          constexpr int X = Bra::L;
          for (int iy = threadIdx.y; iy < ncd_batch; iy += DimY) {
            if (icd+iy >= NCD) break;
            double U[ncart(X)];
            double V[npure(X)] = {};
            if (kab) {
              for (int ix = 0; ix < npure(X); ++ix) {
                V[ix] = BraKet(threadIdx.x + ij + ix*bra.N, (icd+iy) + kl*NCD);
              }
            }
            hermite_to_cartesian<X>(
              inv_2_p,
              [&](auto p) -> const double& {
                return P[herm::index1(p)][iy][threadIdx.x];
              },
              [&](auto p) -> double& { return U[cart::index(p)]; }
            );
            cartesian_to_pure<X>(
              [&](auto &&x, auto v) {
                int ix = index(x);
                BraKet(threadIdx.x + ij + ix*bra.N, (icd+iy) + kl*NCD) = v + V[ix];
              },
              [&](auto x) {
                return U[cart::index(x)];
              }
            );
          }
        } // (Bra::Centers == 1)

        if constexpr (Bra::Centers == 2) {

          static constexpr int A = Bra::First;
          static constexpr int B = Bra::Second;
          constexpr int NAB = bra.nbf;

          auto& [Eab] = args;

          for (int iab = threadIdx.y; iab < NAB; iab += DimY) {

#pragma unroll
            for (int i = 0; i < ncd_batch; ++i) {
              if (icd+i >= NCD) break;
              double u = 0;
              if (kab) u = BraKet(threadIdx.x + ij + iab*bra.N, (icd+i) + kl*NCD);
              V[0][i] = u;
            }

            //#pragma unroll
            for (int ip = 0; ip < nherm2(A+B-1); ++ip) {
              double E = Eab(threadIdx.x, iab, ip, blockIdx.x);
#pragma unroll
              for (int i = 0; i < ncd_batch; ++i) {
                V[0][i] += E*P[ip][i][threadIdx.x];
              }
            }

            for (int ip = 0; ip < ncart(A+B); ++ip) {
              double Eab = inv_2_p*bra.pure_transform[iab + ip*NAB];
#pragma unroll
              for (int i = 0; i < ncd_batch; ++i) {
                V[0][i] += Eab*P[ip+nherm2(A+B-1)][i][threadIdx.x];
              }
            }

#pragma unroll
            for (int i = 0; i < ncd_batch; ++i) {
              if (icd+i >= NCD) break;
              BraKet(threadIdx.x + ij + iab*bra.N, (icd+i) + kl*NCD) = V[0][i];
            }

          }

        } // (Bra::Centers == 2)


      } // icd

      } // Mode

    }

  };



  template<typename T>
  constexpr bool test(size_t MaxRegisters, size_t MaxShmem = 0) {
    return (
      sizeof(typename T::Registers) <= MaxRegisters &&
      sizeof(typename T::Shmem) <= MaxShmem
    );
  }

  template<size_t MaxRegisters, size_t MaxShmem, typename ... Ts>
  struct find_if {
    static_assert(sizeof...(Ts) == 0);
    using type = void;
  };

  template<size_t MaxRegisters, size_t MaxShmem, typename T, typename ... Ts>
  struct find_if<MaxRegisters, MaxShmem, T, Ts...> {
    using type = std::conditional_t<
      test<T>(MaxRegisters,MaxShmem),
      T,
      typename find_if<MaxRegisters,MaxShmem,Ts...>::type
      >;
  };

  template<typename Kernel, typename ... Args>
  __global__
  __launch_bounds__(Kernel::num_threads, Kernel::min_blocks)
  void launch(Kernel kernel, Args ... args) {
    kernel(args...);
  }

}

#endif // LIBINTX_CUDA_MD_MD_KERNEL_H
