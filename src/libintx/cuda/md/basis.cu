#include "libintx/cuda/md/basis.h"
#include "libintx/cuda/api/api.h"
#include "libintx/cuda/api/stream.h"
#include "libintx/cuda/api/thread_group.h"
#include "libintx/engine/md/hermite.h"
#include "libintx/pure.transform.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include "libintx/math.h"
#include <numbers>

namespace libintx::cuda::md {

  using Center = array<double,3>;

  __device__
  constexpr auto orbitals = hermite::orbitals<2*LMAX>;

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

    __device__
    void init(double a, double b, const auto &r, const auto &thread_group) {
      static_assert(thread_group.size() >= (A+B+1));
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


  template<typename ThreadBlock, int A, int B, bool Pure>
  __global__ __launch_bounds__(ThreadBlock::size())
  void make_basis(const Gaussian2 *gbasis, int K, double *H, int ldH) {

    namespace cart = cartesian;

    auto thread_block = ThreadBlock();

    constexpr int DimX = ThreadBlock::x;
    constexpr int DimY = ThreadBlock::y;
    constexpr int NP = nherm2(A+B);

    __shared__ Gaussian2 ab;
    memcpy1(&gbasis[blockIdx.x], &ab, thread_block);
    thread_block.sync();

    __shared__ Center AB;
    if (thread_block.thread_rank() == 0) {
      AB = ab.r.first - ab.r.second;
    }

    for (int ki = 0, k = 0; ki < ab.first.K; ++ki) {
      for (int kj = 0; kj < ab.second.K; ++kj, ++k) {

        __shared__ E2<A,B> E;
        __shared__ double* Hk;
        __shared__ double a, b;

        {
          __shared__ Hermitian h;
          if (thread_block.thread_rank() == 0) {
            Hk = H + k*ldH + blockIdx.x*K*ldH;
            auto& [ai,Ci] = ab.first.prims[ki];
            auto& [aj,Cj] = ab.second.prims[kj];
            // P = (AB| overlap
            double Kab = std::exp(-(ai*aj)/(ai+aj)*norm(AB));
            //double sij = (ij.first == ij.second ? 1 : 2);
            // shmem values
            a = ai;
            b = aj;
            h.exp = (a+b);
            h.C = Ci*Cj*Kab;
            h.r = center_of_charge(a, ab.r.first, b, ab.r.second);
          }
          thread_block.sync();
          E.init(a, b, AB, thread_block);
          memcpy1(&h, Hermitian::hdata(Hk), thread_block);
        }
        thread_block.sync();

        if constexpr (Pure) {

          constexpr int NA = npure(A);
          constexpr int NB = npure(B);

          static_assert(ncart(A) <= DimX);
          static_assert(npure(B) <= DimX);

          __shared__ double h[NB*ncart(A)*DimY];

          for (int batch = 0; batch < (NP+DimY-1)/DimY; ++batch) {

            int ip = batch*DimY + threadIdx.y;
            int np = min(DimY,NP-batch*DimY);

#define h(i,j,p) h[(j) + (i)*NB + (p)*NB*ncart(A)]

            // [a'b'p] -> [a'bp]
            if (threadIdx.x < ncart(A) && ip < NP) {
              int i = threadIdx.x;
              double v[ncart(B)] = {};
              for (int j = 0; j < ncart(B); ++j) {
                auto a = orbitals[cart::index(A)+i];
                auto b = orbitals[cart::index(B)+j];
                auto p = orbitals[ip];
                v[j] = E(a,b,p);
              }
              pure::transform<B>(
                [&](auto j, auto v) { h(i,index(j),threadIdx.y) = v; },
                [&](auto j) { return v[index(j)]; }
              );
            }
            thread_block.sync();

            // [a'bp] -> [abp]
            double v[ncart(A)] = {};
            if (threadIdx.x < NB && ip < NP) {
              int j = threadIdx.x;
              for (int i = 0; i < ncart(A); ++i) {
                v[i] = h(i,j,threadIdx.y);
              }
            }
            thread_block.sync();
#undef h

#define h(i,j,p) h[(j) + (i)*NB + (p)*NB*NA]
            if (threadIdx.x < NB && ip < NP) {
              int j = threadIdx.x;
              pure::transform<A>(
                [&](auto i, auto v) {
                  //printf("%i,%i,%i %e\n", (int)index(i), (int)j, (int)threadIdx.y, v);
                  h(index(i),j,threadIdx.y) = v;
                },
                [&](auto i) { return v[index(i)]; }
              );
            }
            thread_block.sync();
#undef h

            memcpy(NA*NB*np, h, Hermitian::gdata(Hk)+batch*DimY*NA*NB, thread_block);
            thread_block.sync();

          }

        }

        if constexpr (!Pure) {
          for (int ip = threadIdx.z; ip < nherm2(A+B); ip += blockDim.z) {
            for (int i = threadIdx.y; i < ncart(A); i += blockDim.y) {
              int j = threadIdx.x;
              int idx = j;
              idx += i*ncart(B);
              idx += ip*ncart(B)*ncart(A);
              auto a = orbitals[cart::index(A)+i];
              auto b = orbitals[cart::index(B)+j];
              auto p = orbitals[ip];
              //printf("%i,%i,%i %f @%i\n", i, j, ip, E(p,a,b), H-h);
              double e = E(a,b,p);
              Hermitian::gdata(Hk)[idx] = e;
            }
          }
        }

      }
    }

  }

  template<int A, int B>
  Basis2 make_basis(
    const std::vector<Gaussian2> &ab,
    device::vector<double> &H)
  {

    constexpr uint NP = nherm2(A+B);

    // auto idx = pairs.at(0);
    auto a = ab[0].first;
    auto b = ab[0].second;
    int K = a.K*b.K;
    int N = ab.size();

    bool pure = (a.pure && b.pure);
    int nh = Hermitian::extent(a,b);
    H.resize(nh*K*N);

    dim3 grid = { (unsigned int)N };

    if (pure) {
      constexpr bool Pure = true;
      constexpr uint NX = std::max(ncart(A),npure(B));
      using Block = thread_block<NX, std::min(NP,128/NX)>;
      //ssert(false);
      //printf("BLOCK<%i,%i,%i>\n", Block::x, Block::y, Block::z);
      make_basis<Block,A,B,Pure><<<grid,Block()>>>(ab.data(), K, H.data(), nh);
    }
    else {
      constexpr bool Pure = false;
      constexpr uint NA = ncart(A);
      constexpr uint NB = ncart(B);
      constexpr uint MaxThreads = std::min<uint>(128,NA*NB*NP);
      constexpr uint NX = NB;
      constexpr uint NY = std::min<uint>(MaxThreads/NX,NA);
      constexpr uint NZ = (NY != NA) ? 1 : std::min<uint>(MaxThreads/(NX*NY),64);
      static_assert(NZ);
      static_assert(NY == ncart(A) || NZ == 1);
      //printf("BLOCK<%i,%i,%i>\n", NX, NY, NZ);
      using Block = thread_block<NX,NY,NZ>;
      make_basis<Block,A,B,Pure><<<grid,Block()>>>(ab.data(), K, H.data(), nh);
    }

    return Basis2 {
      .first = a,
      .second = b,
      .K = K,
      .N = N,
      .data = H.data()
    };

  }

  Basis2 make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    device::vector<double> &H,
    cudaStream_t stream)
  {

    std::vector<Gaussian2> ab;
    ab.reserve(pairs.size());
    for (auto [i,j] : pairs) {
      Gaussian2 g = {
        shell(A[i]), shell(B[j]),
        { center(A[i]), center(B[j]) }
      };
      ab.push_back(g);
    }

    cuda::host::register_pointer(ab.data(), ab.size());

    auto a = ab[0].first;
    auto b = ab[0].second;

    using F = std::function<
      Basis2(
        const std::vector<Gaussian2> &ab,
        device::vector<double> &H
      )>;

    static auto make_basis = make_array<F,LMAX+1,LMAX+1>(
      [](auto ... args) -> F {
        return &md::make_basis<args...>;
      }
    );

    auto basis = make_basis[a.L][b.L](ab, H);

    cuda::stream::synchronize(stream);
    cuda::host::unregister_pointer(ab.data());

    return basis;

  }

}
