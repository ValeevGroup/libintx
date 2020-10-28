#include "libintx/cuda/forward.h"
#include "libintx/cuda/jengine/os/jengine.h"
#include "libintx/cuda/eri/os/eri3.h"
#include "libintx/cuda/api/api.h"
#include "libintx/cuda/api/stream.h"
#include "libintx/engine.h"
#include "libintx/utility.h"

#include <utility>

#include <cuda_fp16.h>

namespace libintx::cuda::jengine::os {

  using libintx::cuda::os::ERI;
  using libintx::cuda::os::Vector3;

  inline double norm2(const double *v, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) {
      s += fabs(v[i]);
    }
    return s;
  }

  struct Index2 {
    uint32_t a:16;
    uint32_t b:16;
    uint64_t offset:62;
    uint64_t scale:2;
    float max;
  };

  struct Index1 {
    uint32_t offset;
    uint16_t x;
    __half max;
  };

  struct Block1 {

    struct Block {
      std::shared_ptr<Gaussian> X;
      cuda::device::vector<Index1> index;
      double *data;
      float max;
    };

    std::vector<Block> blocks;
    cuda::device::vector<double> dvec;
    float max;

    auto begin() const { return blocks.begin(); }
    auto end() const { return blocks.end(); }

    explicit Block1(const JEngine::Basis &basis, const JEngine::Screening *screening, const double *V) {
      this->max = 0;
      //size_t offset = 0;
      blocks.resize(basis.blocks.size());
      auto block = blocks.begin();
      for (auto [X,Xs] : basis.blocks) {
        int nx = nbf(*X);
        block->X = X;
        block->max = 0;
        std::vector<Index1> index;
        for (auto x : Xs) {
          //block->start.push_back(x.start);
          float v = 1;
          if (V) v = norm2(V+x.start, nx);
          float max1 = v*fabs(screening->max1(x.shell));
          index.push_back(Index1{(uint32_t)x.start, (uint16_t)x.center, max1});
          block->max = std::max(max1, block->max);
        }
        block->index.assign(index.data(), index.size());
        block->data = this->dvec.data();
        this->max = std::max(this->max, block->max);
	// printf("X-Block %p: max=%f\n", X.get(), block->max);
        ++block;
      }
      if (V) this->dvec.assign(V, basis.nbf);
      else this->dvec.assign_zero(basis.nbf);
    }

  };

  struct Block2 {

    std::shared_ptr<Gaussian> A, B;
    std::vector< std::pair<size_t,size_t> > bfs;
    cuda::device::vector<Index2> index;
    cuda::device::vector<double> dvec;
    cuda::host::vector<double> hvec;

    float max = 0;
    bool need_update = false;

    template<typename Iterator>
    void initialize(
      Iterator first, Iterator second,
      const JEngine::TileIn &D,
      const JEngine::Screening *screening, float xmax,
      Stream &stream)
    {
      if (first->gaussian->L < second->gaussian->L) {
        std::swap(first, second);
      }
      const auto& [A,As] = *first;
      const auto& [B,Bs] = *second;
      this->A = A;
      this->B = B;
      bfs.clear();
      int na = nbf(*A);
      int nb = nbf(*B);
      int nab = na*nb;
      this->hvec.resize(As.size()*Bs.size()*nab);
      auto buffer = this->hvec.data();
      size_t offset = 0;
      std::vector<Index2> index;
      for (auto a : As) {
        for (auto b : Bs) {
          if (first == second && (a.shell < b.shell)) continue;
          float max = 1;
          if (D) {
            std::pair<int,int> ra{a.start, a.start+na};
            std::pair<int,int> rb{b.start, b.start+nb};
            D(ra, rb, buffer);
            max = norm2(buffer, nab);
          }
          max *= xmax;
          max *= screening->max2(a.shell,b.shell);
          this->max = std::max(max, this->max);
          if (screening->skip(max)) {
            //printf("skipping %i,%i\n", a.shell,b.shell);
            continue;
          }
          buffer += nab;
          uint32_t scale = ((a.shell == b.shell) ? 1 : 2);
          index.push_back(
            { (uint32_t)a.center, (uint32_t)b.center, offset, scale, 1 }
          );
          bfs.push_back({a.start, b.start});
          offset += nab;
        }
      }
      this->index.resize(index.size());
      cuda::copy(index.data(), index.data()+index.size(), this->index.data());
      this->dvec.resize(nab*index.size());
      if (D) {
        cuda::copy(hvec.data(), hvec.data()+dvec.size(), dvec.data(), stream);
      }
      if (!D) {
        this->dvec.memset(0);
      }
    }

    void update(const JEngine::TileOut &J) {
      if (!this->need_update) return;
      int na = nbf(*A);
      int nb = nbf(*B);
      this->hvec.assign(this->dvec.data(), this->dvec.size());
      auto *buffer = this->hvec.data();
      assert(buffer);
      for (auto [a,b] : this->bfs) {
        std::pair ra{a, a+na};
        std::pair rb{b, b+nb};
        J(ra, rb, buffer);
        buffer += na*nb;
      }
      this->bfs.clear();
    }

  };


  struct Transform {

    enum Type { X=1, J=2 };

    const Type type;
    Transform(Type type) : type(type) {}

    template<class ... Args>
    __device__
    void operator()(const Args& ... args) const {
      if (type == Type::X) {
        compute_x(args...);
        return;
      }
      if (type == Type::J) {
        compute_j(args...);
        return;
      }
      __trap();
    }

    __device__
    static void compute_x(
      int NAB, int NX, double *V,
      const Index2 &AB, const Index1 &X,
      const double *input, double* output)
    {
      const auto &thread_rank = this_thread_block().thread_rank();
      const auto &num_threads = this_thread_block().size();

      const double *D = input + AB.offset;
      for (int ij = thread_rank; ij < NAB; ij += num_threads) {
        double dij = D[ij];
        for (int x = 0; x < NX; ++x) {
          double h = dij*V[x + ij*NX];
          // 2-fold reduction
          if (ij | 0x1 < NAB) {
            auto mask = __activemask();
            // exchange registers
            int src = thread_rank%32 ^ 0x1;
            double h2 = __shfl_sync(mask, h, src);
            h += h2;
          }
          V[x + ij*NX] = h;
        }
      }

      __syncthreads();
      double *Y = output + X.offset;
      double s = AB.scale;
      for (int x = thread_rank; x < NX; x += num_threads) {
        double Hx = 0;
        for (int ij = 0; ij < NAB; ij += 2) {
          Hx += V[x + ij*NX];
        }
        atomicAdd(Y+x, s*Hx);
        //printf("X(%i)=%f*%f\n", x, Hx, s);
      }

    }

    __device__
    static void compute_j(
      int NAB, int NX, double *V,
      const Index2 &AB, const Index1 &X,
      const double *input, double* output)
    {

      const auto &thread_rank = this_thread_block().thread_rank();
      const auto &num_threads = this_thread_block().size();

      double *Y = V + NAB*NX;
      for (int x = thread_rank; x < NX; x += num_threads) {
        Y[x] = input[X.offset+x];
        //printf("X[%i]=%f\n", X.offset+x, Y[x]);
      }

      __syncthreads();
      double s = AB.scale;
      double *J = output+AB.offset;
      for (int k = thread_rank; k < NAB; k += num_threads) {
        double Jk = 0;
        for (int x = 0; x < NX; ++x) {
          Jk += Y[x]*V[x + k*NX];
        }
        //printf("J(%i)=%f %f %f\n", k, s*Jk, Jk, V[k]);
        atomicAdd(J+k, s*Jk);
      }

    }

  };

  template<int _AB, int _X>
  struct ComputeKernel : os::ERI<3, _AB, _X, Boys, ComputeKernel<_AB,_X> > {

    __device__
    void operator()() {
      __shared__ os::Vector3 rA, rB, rX;
      __shared__ Index2 ab;
      __shared__ Index1 x;
      if (threadIdx.x+threadIdx.y == 0) {
        ab = ABs[blockIdx.x];
        x = Xs[blockIdx.y];
        rA = centers[ab.a];
        rB = centers[ab.b];
        rX = centers[x.x];
      }
      __syncthreads();
      // screen
      if (float(ab.max)*float(x.max) < this->cutoff) return;
      double *V = this->compute(rA,rB,rX);
      __syncthreads();
      int NA = nbf(this->A);
      int NB = nbf(this->B);
      int NX = npure(ComputeKernel::X::L);
      transform(NA*NB, NX, V, ab, x, input, output);
    }

    Transform transform;
    const Index2 *ABs = nullptr;
    const Index1 *Xs = nullptr;
    const Double<3> *centers = nullptr;
    const double *input = nullptr;
    double* output = nullptr;
    float cutoff;

    template<class ... Args>
    ComputeKernel(const Transform &transform, const Args& ... args)
      : os::ERI<3, _AB, _X, Boys, ComputeKernel<_AB,_X> >(args..., cuda::boys()),
        transform(transform)
    {
    }

  };

  struct Engine {

    static Engine X(const Double<3> *centers, float cutoff = 1e-12) {
      return Engine(Transform::X, centers, cutoff);
    }

    static Engine J(const Double<3> *centers, float cutoff = 1e-12) {
      return Engine(Transform::J, centers, cutoff);
    }

    template<int _AB, int _X>
    static void launch_compute_kernel(
      const Transform &transform,
      const Gaussian &A, const Gaussian &B, const Gaussian &X,
      int NAB, const Index2 *ABs,
      int NX, const Index1 *Xs,
      const Double<3> *centers,
      const double *input, double* output,
      float cutoff,
      const Stream &stream)
    {
      auto kernel = ComputeKernel<_AB,_X>(transform,A,B,X);
      kernel.ABs = ABs;
      kernel.Xs = Xs;
      kernel.centers = centers;
      kernel.input = input;
      kernel.output = output;
      kernel.cutoff = cutoff;
      dim3 grid = { (uint32_t)NAB, (uint32_t)NX };
      kernel.launch(grid, stream);
    }

    void compute(
      const Block2 &ab, const Block1 &x,
      const double *input, double* output,
      const JEngine::Screening *screening,
      const Stream &stream) const
    {
      using ComputeKernel = std::function<void(
        const Transform &transform,
        const Gaussian &A, const Gaussian &B, const Gaussian &X,
        int NAB, const Index2 *ab,
        int NX, const Index1 *x,
        const Double<3> *centers,
        const double *input, double* output,
        float cutoff,
        const Stream &stream
      )>;
      static auto compute_kernel_table = make_array<ComputeKernel>(
        [](auto ab, auto x) {
          return ComputeKernel(
            Engine::launch_compute_kernel<ab.value, x.value>
          );
        },
        std::make_index_sequence<LMAX*2+1>{},
        std::make_index_sequence<XMAX+1>{}
      );
      if (!ab.index.size()) return;
      for (const auto &x : x) {
        if (screening->skip(x.max*ab.max)) continue;
        //   "JEngine::compute: %p %p %p %p\n",
        //   ab.index.data(), x.index.data(), input, output
        // );
        const auto &A = *ab.A;
        const auto &B = *ab.B;
        const auto &X = *x.X;
        auto compute_kernel = compute_kernel_table[A.L+B.L][X.L];
        compute_kernel(
          this->transform,
          A, B, X,
          ab.index.size(), ab.index.data(),
          x.index.size(), x.index.data(),
          this->centers,
          input, output,
          this->cutoff,
          stream
        );
      }
    }

  private:

    Transform transform;
    const Double<3> *centers = nullptr;
    float cutoff = 0;

    Engine(Transform transform, const Double<3> *centers, float cutoff)
      : transform(transform) ,
        centers(centers),
        cutoff(cutoff)
    {
    }

  };


  JEngine::JEngine(
    const std::vector< std::tuple< Gaussian, Double<3> > > &basis,
    const std::vector< std::tuple< Gaussian, Double<3> > > &df_basis,
    std::function<void(double*)> v_transform,
    std::shared_ptr<const libintx::JEngine::Screening> screening)
  {
    std::vector< Double<3> > centers;
    basis_ = Basis(basis, centers);
    df_basis_ = Basis(df_basis, centers);
    v_transform_ = v_transform;
    screening_ = screening;
    centers_.assign(centers.data(), centers.size());
  }

  void JEngine::J(const TileIn &D, const TileOut &J) {
    std::vector<double> X(df_basis_.nbf, 0.0);
    this->J1(D, X.data());
    v_transform_(X.data());
    this->J2(X.data(), J);
  }

  void JEngine::J1(const TileIn &D, double *X) {
    auto t = time::now();
    double tgpu = 0;
    const auto *screening = this->screening_.get();
    Block1 x(this->df_basis_, screening, nullptr);
    const auto &basis = this->basis_;
    StreamPool< std::shared_ptr<Block2> > stream_pool(1);
    auto engine = Engine::X(this->centers_.data());
    for (auto first = basis.blocks.begin(); first != basis.blocks.end(); ++first) {
      for (auto second = basis.blocks.begin(); second <= first; ++second) {
        auto& [stream,ab] = stream_pool.next();
        if (!ab) ab.reset(new Block2);
        ab->initialize(first, second, D, screening, x.max, stream);
        stream.synchronize();
        auto t = time::now();
        engine.compute(*ab, x, ab->dvec.data(), x.dvec.data(), screening, stream);
        stream.synchronize();
        tgpu += time::since(t);
      }
    }
    stream_pool.synchronize();
    cuda::copy(x.dvec.begin(), x.dvec.end(), X);
    cuda::device::synchronize();
    // printf("JEngine::X: t=%f, t(gpu)=%f\n", time::since(t), tgpu);
  }

  void JEngine::J2(const double *X, const TileOut &J) {
    //auto t = time::now();
    double tgpu = 0;
    StreamPool< std::shared_ptr<Block2> > stream_pool(1);
    // Block2 ab;
    // ab.stream = std::make_unique<Stream>();

    const auto *screening = this->screening_.get();

    Block1 x(this->df_basis_, screening, X);

    const auto &basis = this->basis_;
    auto engine = Engine::J(this->centers_.data());
    for (auto first = basis.blocks.begin(); first != basis.blocks.end(); ++first) {
      for (auto second = basis.blocks.begin(); second <= first; ++second) {
        auto& [stream,ab] = stream_pool.next();
        if (!ab) ab.reset(new Block2);
        { auto t = time::now(); stream.synchronize(); tgpu += time::since(t); }
        ab->update(J);
        ab->initialize(first, second, nullptr, screening, x.max, stream);
        engine.compute(*ab, x, x.dvec.data(), ab->dvec.data(), screening, stream);
        cuda::copy(
          ab->dvec.begin(), ab->dvec.end(),
          ab->hvec.data(),
          stream
        );
        ab->need_update = true;
      }
    }
    for (auto& [stream,ab] : stream_pool) {
      stream.synchronize();
      ab->update(J);
    }
    cuda::device::synchronize();
    // printf("JEngine::J: t=%f, t(gpu)=%f\n", time::since(t), tgpu);
  }

  JEngine::Basis::Basis(
    const std::vector< std::tuple<Gaussian, Double<3> > > &basis,
    std::vector< Double<3> > &centers)
  {
    for (size_t idx = 0; idx < basis.size(); ++idx) {
      const Gaussian& s = std::get<0>(basis[idx]);
      auto block_iterator = std::find_if(
        blocks.begin(), blocks.end(),
        [&s](const auto &b) { return s == *b.gaussian; }
      );
      if (block_iterator == blocks.end()) {
        block_iterator = blocks.insert(
          block_iterator,
          { std::make_shared<Gaussian>(s), {} }
        );
      }
      const auto& r = std::get<1>(basis[idx]);
      int ridx = 0;
      for (; ridx < centers.size(); ++ridx) {
        if (centers.at(ridx) == r) break;
      }
      if (ridx == centers.size()) centers.push_back(r);
      //printf("Shell %i, ptr=%p\n", i, std::get<0>(*shell_iterator).get());
      block_iterator->list.push_back(
        Block::Index{ (int)idx, ridx, this->nbf }
      );
      this->nbf += libintx::nbf(s);
    }
  }

}


namespace libintx::cuda {

  std::unique_ptr<libintx::JEngine> make_jengine(
    const std::vector< std::tuple< Gaussian, array<double,3> > > &basis,
    const std::vector< std::tuple< Gaussian, array<double,3> > > &df_basis,
    std::function<void(double*)> v_transform,
    std::shared_ptr<const libintx::JEngine::Screening> screening)
  {
    using JEngine = libintx::cuda::jengine::os::JEngine;
    return std::make_unique<JEngine>(
      basis, df_basis, v_transform, screening
    );
  }

}
