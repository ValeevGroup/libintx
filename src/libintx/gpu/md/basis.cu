// -*-c++-*-

// this must come first to resolve HIP device asserts
#include "libintx/gpu/api/runtime.h"

#include "libintx/gpu/md/basis.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/thread_group.h"
#include "libintx/ao/md/hermite.h"
#include "libintx/pure.transform.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include "libintx/math.h"

#include <functional>

namespace libintx::gpu::md {

  namespace cart = cartesian;
  namespace herm = hermite;

  template<int A, int B>
  struct E2 {

    __device__
    auto& value(int i, int j, int k, int x) {
      constexpr int strides[4] = {
        (A+B+1)*(B+1),
        (A+B+1),
        1,
        (A+B+1)*(B+1)*(A+1)
      };
      return data[i*strides[0]+j*strides[1]+k*strides[2]+x*strides[3]];
    }

    template<typename T>
    __device__
    auto operator()(T &&a, T &&b, T &&p) {
      double v = 1;
      for (int i = 0; i < 3; ++i) {
        v *= value(a[i], b[i], p[i], i);
      }
      return v;
    }

    template<typename G>
    __device__
    void init(double a, double b, const auto &r, const G &thread_group) {
      static_assert(G::size() >= (A+B+1));
      auto p = a + b;
      assert(p);
      auto q = ((a ? a : 1)*(b ? b : 1))/p;
      assert(q);
      fill(3*(A+B+1)*(B+1)*(A+1), this->data, 0, thread_group);
      thread_group.sync();
      if (thread_group.thread_rank() == 0) {
        value(0,0,0,0) = 1;
        value(0,0,0,1) = 1;
        value(0,0,0,2) = 1;
      }
      thread_group.sync();
      auto k = thread_group.thread_rank();
      for (int i = 1; i <= A; ++i) {
        thread_group.sync();
        if (k > i) continue;
#pragma unroll
        for (int x = 0; x < 3; ++x) {
          double v0 = (k ? value(i-1,0,k-1,x) : 0);
          double v1 = value(i-1,0,k,x);
          double v2 = (k < i ? value(i-1,0,k+1,x) : 0);
          double v = (1/(2*p))*v0 - (q*r[x]/a)*v1 + (k+1)*v2;
          value(i,0,k,x) = v;
        }
      }
      // j
      for (int j = 1; j <= B; ++j) {
        for (int i = 0; i <= A; ++i) {
          thread_group.sync();
          if (k > i+j) continue;
          for (int x = 0; x < 3; ++x) {
            double v0 = (k ? value(i,j-1,k-1,x) : 0);
            double v1 = value(i,j-1,k,x);
            double v2 = (k < i+j ? value(i,j-1,k+1,x) : 0);
            double v = (1/(2*p))*v0 + (q*r[x]/b)*v1 + (k+1)*v2;
            value(i,j,k,x) = v;
          }
        }
      }
    }

    double data[(A+1)*(B+1)*(A+B+1)*3];

  };

  template<int A, int B, int P = A+B>
  struct Hermite2 {
    Hermite h;
    double E[nherm2(P)*npure(A)*npure(B)];
  };

  template<int A, int B, int Batch>
  __global__ __launch_bounds__(64)
  void make_basis_kernel(TensorRef<Gaussian1,2> G1, TensorRef<Hermite2<A,B>,2> H2) {

    auto &N = G1.dimensions()[0];
    if (blockIdx.x*blockDim.x >= N) return;

    auto thread_block = this_thread_block();
    extern __shared__ double shmem[];

    auto *shmem_G1 = reinterpret_cast<Gaussian1*>(shmem);

    // if (thread_block.thread_rank() == 0) {
    //   printf("** %i,%i %lu\n", A, B, std::min<size_t>(blockDim.x, N-blockIdx.x*blockDim.x));
    // }

    memcpy(
      std::min<size_t>(blockDim.x, N-blockIdx.x*blockDim.x),
      &G1(blockIdx.x*blockDim.x,blockIdx.y),
      shmem_G1,
      thread_block
    );
    thread_block.sync();

    auto partition = tiled_partition<16>(this_thread_block());

    auto &h = (
      Batch ?
      reinterpret_cast<Hermite*>(shmem_G1 + blockDim.x)[threadIdx.x] :
      reinterpret_cast<Hermite2<A,B>*>(shmem_G1 + blockDim.x)[threadIdx.x].h
    );

    auto &g = shmem_G1[threadIdx.x];
    auto &[ra,rb] = g.r;
    auto &[a,b] = g.exp;
    // P = (AB| overlap
    double Kab = std::exp(-(a*b)/(a+b)*norm(ra,rb));
    //double sij = (ij.first == ij.second ? 1 : 2);
    // shmem values
    h.exp = (a+b);
    h.C = g.C*Kab;
    h.r = center_of_charge(a, ra, b, rb);
    h.inv_2_exp = 1/math::pow<A+B>(2*(a+b));

    if (threadIdx.x + blockIdx.x*blockDim.x < N) assert(h.exp > 0);



    if constexpr (!Batch) {
      auto *shmem_H2 = reinterpret_cast<Hermite2<A,B>*>(shmem_G1 + blockDim.x);
      if constexpr (A+B == 0) {
        shmem_H2[threadIdx.x].E[0] = h.inv_2_exp;
      }

      else {
        libintx::md::E2<double,A,B,A+B> E2(a,b,ra-rb);

        // if (threadIdx.x < N) {
        //   // printf("** %i = r=%f,%f,%f, exp=%f,%f\n", threadIdx.x, g.r[0], g.r[1], g.r[2], a, b);
        //   printf("** E(0,0,0) = %f\n", E2(Orbital{0,0,0},Orbital{0,0,0},Orbital{0,0,0}));
        // }

#pragma unroll
        for (auto p : hermite::orbitals2<A+B-1>) {
          int ip = hermite::index2(p);
          double Eb[ncart(B)][npure(A)] = {};
#pragma unroll
          for (auto b : cartesian::orbitals<B>()) {
            //if (!(p <= b)) continue;
            double Ea[ncart(A)] = {};
#pragma unroll
            for (auto a : cartesian::orbitals<A>()) {
              //if (!(p <= a+b)) continue;
              Ea[index(a)] = E2(a,b,p);
              // if (threadIdx.x < N) {
              //   printf("** Ea[%i,%i] = %f\n", index(a), hermite::index2(p), Ea[index(a)]);
              //   printf("** Ea[%i,%i] = %f\n", index(a), hermite::index2(p), E2(Orbital{0,0,0},b,p));
              // }
            }
            pure::cartesian_to_pure<A>(
              [&](auto a) { return Ea[index(a)]; },
              [&](auto a, auto v) { Eb[index(b)][index(a)] = v; }
            );
          }
#pragma unroll
          for (int ia = 0; ia < npure(A); ++ia) {
            pure::cartesian_to_pure<B>(
              [&](auto b) { return Eb[index(b)][ia]; },
              [&](auto b, auto v) {
                shmem_H2[threadIdx.x].E[ia + index(b)*npure(A) + ip*npure(A,B)] = v;
              }
            );
          }
        }
      }
      //partition.sync();
      thread_block.sync();
      int pidx = partition.size()*(threadIdx.x/partition.size());
      if (pidx + blockIdx.x*blockDim.x < N) {
        memcpy(
          std::min<size_t>(partition.size(), N-(pidx + blockIdx.x*blockDim.x)),
          shmem_H2 + pidx,
          &H2(pidx + blockIdx.x*blockDim.x,blockIdx.y),
          partition
        );
      }
      return;
    } // !Batched

    if constexpr (Batch) {
      partition.sync();
      auto inv_2_exp = h.inv_2_exp;
      int pidx = partition.size()*(threadIdx.x/partition.size());
      for (int i = 0; i < partition.size(); ++i) {
        if (i + pidx + blockIdx.x*blockDim.x >= N) break;
        memcpy1(
          &reinterpret_cast<Hermite*>(shmem_G1 + blockDim.x)[i + pidx],
          &H2(i + pidx + blockIdx.x*blockDim.x,blockIdx.y).h,
          partition
        );
      }
      auto r = ra-rb;
      thread_block.sync();
      libintx::md::E2<double,A,B,A+B-1> E2(a,b,ra-rb);
      //printf("*** Batched %i,%i,%i\n", A,B,A+B-1);
#pragma unroll
      for (auto p : hermite::orbitals2<A+B-1>) {
        int ip = hermite::index2(p);
        double *shmem_E = reinterpret_cast<double*>(shmem) + threadIdx.x*npure(A)*ncart(B);
        //double Eb[ncart(B)][npure(A)] = {};
#pragma unroll
        for (auto b : cartesian::orbitals<B>()) {
          //if (!(p <= b)) continue;
          double Ea[ncart(A)] = {};
#pragma unroll
          for (auto a : cartesian::orbitals<A>()) {
            if (!(p <= a+b)) continue;
            Ea[index(a)] = E2(a,b,p);
          }
          pure::cartesian_to_pure<A>(
            [&](auto a) { return Ea[index(a)]; },
            //[&](auto a, auto v) { Eb[index(b)][index(a)] = v; }
            [&](auto a, auto v) { shmem_E[index(a) + index(b)*npure(A)] = v; }
          );
        }
#pragma unroll
        for (int ia = 0; ia < npure(A); ++ia) {
          double Eb[ncart(B)] = {};
#pragma unroll
          for (int ib = 0; ib < ncart(B); ++ib) {
            Eb[ib] = shmem_E[ia + ib*npure(A)];
          }
          pure::cartesian_to_pure<B>(
            //[&](auto b) { return Eb[index(b)][ia]; },
            [&](auto b) { return Eb[index(b)]; },
            [&](auto b, auto v) { shmem_E[ia + index(b)*npure(A)] = v; }
          );
        }
        partition.sync();
        for (int i = 0; i < partition.size(); ++i) {
          if (i + pidx + blockIdx.x*blockDim.x >= N) break;
          int ip = hermite::index2(p);
          memcpy(
            npure(A,B),
            &reinterpret_cast<double*>(shmem)[(i + pidx)*npure(A)*ncart(B)],
            &H2(i + pidx + blockIdx.x*blockDim.x,blockIdx.y).E[ip*npure(A,B)],
            partition
          );
        }
        partition.sync();
      } // ip
    }

  }


  template<int A, int B>
  void init_basis(
    const TensorRef<Gaussian1,2> &G1,
    TensorRef<Hermite2<A,B>,2> &H, // Hermite data
    gpuStream_t stream)
  {

    constexpr uint NP = nherm2(A+B);

    auto [N,K] = G1.dimensions();

    if constexpr (A+B <= 5) {
      //dim3 block = K == 1 ? dim3{ 64, 1 } : dim3{ 32, 2 };
      constexpr dim3 block = { 64, 1, 1 };
      dim3 grid = { int(N+block.x-1)/block.x, (uint)K, 1 };
      constexpr int Batch = (
        sizeof(Gaussian1) + sizeof(Hermite2<A,B,A+B>) > 32*sizeof(double)
      );
      static_assert(A+B > 0 || !Batch);
      constexpr int shmem = (
        Batch ?
        std::max(sizeof(Gaussian1) + sizeof(Hermite), npure(A)*ncart(B)*sizeof(double)) :
        sizeof(Gaussian1) + sizeof(Hermite2<A,B,A+B>)
      );
      static_assert(block.x*shmem <= 48*1024);
      //println( A, B, N, grid.x, shmem);
      make_basis_kernel<A,B,Batch><<<grid,block,shmem*block.x,stream>>>(G1, H);
      gpu::check_last_error();
      //gpu::stream::synchronize(stream);
    }
    else {
      libintx_assert("not implemented");
      // dim3 grid = { (unsigned int)N };
      // if (pure) {
      //   constexpr bool Pure = true;
      //   constexpr uint NX = std::max(ncart(A),npure(B));
      //   using Block = thread_block<NX, std::min(NP,128/NX)>;
      //   //ssert(false);
      //   //printf("BLOCK<%i,%i,%i>\n", Block::x, Block::y, Block::z);
      //   make_basis<Block,A,B,Pure><<<grid,Block(),0,stream>>>(pairs, H.data(), extent, k_stride);
      //   constexpr libintx::md::pure_transform<A,B> pure_transform;
      //   // gpu::memcpy(
      //   //   pure_transform_ptr,
      //   //   pure_transform.data,
      //   //   sizeof(pure_transform.data)
      //   // );
      // }
      // else {
      //   constexpr bool Pure = false;
      //   constexpr uint NA = ncart(A);
      //   constexpr uint NB = ncart(B);
      //   constexpr uint MaxThreads = std::min<uint>(128,NA*NB*NP);
      //   constexpr uint NX = NB;
      //   constexpr uint NY = std::min<uint>(MaxThreads/NX,NA);
      //   constexpr uint NZ = (NY != NA) ? 1 : std::min<uint>(MaxThreads/(NX*NY),64);
      //   static_assert(NZ);
      //   static_assert(NY == ncart(A) || NZ == 1);
      //   //printf("BLOCK<%i,%i,%i>\n", NX, NY, NZ);
      //   using Block = thread_block<NX,NY,NZ>;
      //   make_basis<Block,A,B,Pure><<<grid,Block()>>>(pairs, H.data(), extent, k_stride);
      //   pure_transform_ptr = nullptr;
      // }
    }

  }

  Basis1 make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    device::vector<Hermite> &H,
    gpuStream_t stream)
  {

    libintx_assert(!A.empty());
    libintx_assert(!idx.empty());

    int L = A[idx.front()].L;
    int K = A[idx.front()].K;
    int N = idx.size();

    for (auto i : idx) {
      libintx_assert(A[i].K == K);
      libintx_assert(A[i].L == L);
    }

    std::vector<Hermite> a;
    a.reserve(K*idx.size());
    for (int k = 0; k < K; ++k) {
      for (auto i : idx) {
        auto &r = center(A[i]);
        auto &g = A[i].prims;
        auto e = g[k].a;
        auto C = g[k].C;
        a.push_back( { e, C, r, 1.0/(2*e) } );
      }
    }
    H.assign(a.data(), a.size());

    return Basis1{L,K,N,H.data()};

  }

  void HermiteBasis::init(
    pair<const Basis<Gaussian>&> basis,
    const std::vector<Index2> &pairs,
    const double *norms,
    gpuStream_t stream)
  {
    std::vector< std::tuple<double,Index2> > pairs2(pairs.size());
    for (size_t i = 0; i < pairs.size(); ++i) {
      double norm = (norms ? norms[i] : math::infinity<double>);
      pairs2[i] = { norm, pairs[i] };
    }
    this->init(basis, pairs2, stream);
  }

  void HermiteBasis::init(
    pair<const Basis<Gaussian>&> basis,
    const std::vector< std::tuple<double,Index2> > &pairs,
    gpuStream_t stream)
  {

    {
      auto [nij,ij] = pairs.at(0);
      auto &first = basis.first[ij.first];
      auto &second = basis.second[ij.second];
      this->first = first;
      this->second = second;
      this->K = first.K*second.K;
      this->N = pairs.size();
      this->gaussian1.resize(N*K);
    }

    TensorRef<Gaussian1,2> G1(this->gaussian1.data(), { this->N, this->K });

    for (size_t idx = 0; auto [nij,ij] : pairs) {
      auto &a = basis.first[ij.first];
      auto &b = basis.second[ij.second];
      libintx_assert(a == this->first && b == this->second);
      libintx_assert(a.K*b.K == this->K);
      double r2 = norm(a.r,b.r);
      for (int kj = 0, k = 0; kj < b.K; ++kj) {
        for (int ki = 0; ki < a.K; ++ki) {
          Gaussian1 g = {
            .exp = { a.prims[ki].a, b.prims[kj].a },
            .r = { a.r, b.r },
            .C = a.prims[ki].C*b.prims[kj].C,
            .norm = static_cast< decltype(Gaussian1::norm) >(nij)
          };
          G1(idx,k++) = g;
        }
      }
      ++idx;
    } // pairs

    size_t N_aligned = alignment*((N + alignment - 1)/alignment);

    jump_table(
      std::make_index_sequence<LMAX+1>{},
      std::make_index_sequence<LMAX+1>{},
      first.L, second.L,
      [&](auto A, auto B) {
        this->hermite.resize(
          (sizeof(Hermite2<A,B>)/sizeof(double))*N_aligned*K
        );
        TensorRef<Hermite2<A,B>,2> H2{
          reinterpret_cast<Hermite2<A,B>*>(this->hermite.data()),
          { N_aligned, (size_t)K }
        };
        strides[0] = sizeof(Hermite2<A,B>)/sizeof(double);
        strides[1] = strides[0]*N_aligned;
        init_basis<A,B>(G1,H2,stream);
      }
    );

  }

}
