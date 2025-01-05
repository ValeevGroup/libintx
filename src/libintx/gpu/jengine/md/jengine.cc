#include "libintx/gpu/jengine/md/jengine.h"
#include "libintx/gpu/jengine/md/forward.h"
#include "libintx/gpu/boys.h"
#include "libintx/integral/md/hermite.h"

#include "libintx/utility.h"
#include "libintx/thread_pool.h"

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/gpu/api/api.h"
#include "libintx/gpu/api/runtime.h"

#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include <mutex>

namespace libintx::gpu::jengine::md {

  constexpr int THREADS_PER_DEVICE = 1;

  template<int Step, class Boys>
  void df_jengine_kernel(
    int p, int q,
    const Boys &boys,
    int NP, const Primitive2 *P,
    int NQ, const Primitive2 *Q,
    const double* input,
    double* output,
    float cutoff,
    Stream &s)
  {
    using Kernel = std::function<void(
      const Boys &boys,
      int NP, const Primitive2 *P,
      int NQ, const Primitive2 *Q,
      const double*, double*,
      float cutoff,
      Stream &s
    )>;
    static auto kernel_table = make_array<Kernel,2*LMAX+1,XMAX+1>(
      [](auto p, auto q) {
        return &df_jengine_kernel<p.value, q.value, Step, Boys>;
      }
    );
    auto kernel = kernel_table[p][q];
    assert(kernel);
    kernel(boys, NP, P, NQ, Q, input, output, cutoff, s);
  }

  JEngine::JEngine(
    const std::vector< std::tuple< Gaussian, Double<3> > > &basis,
    const std::vector< std::tuple< Gaussian, Double<3> > > &df_basis,
    std::function<void(double*)> v_transform,
    std::shared_ptr<const Screening> screening)
    : v_transform_(v_transform),
      cutoff_((screening ? screening->max() : 0))
  {
    // DF-ket
    for (size_t i = 0, kbf = 0; i < df_basis.size(); ++i) {
      auto [x,r] = df_basis[i];
      int l = x.L;
      if (Q_.blocks.size() < size_t(l+1)) {
        Q_.blocks.resize(l+1);
      }
      float max = (screening ? screening->max1(i) : 1);
      Q_.max = std::max(Q_.max,max);
      int nprim = x.K;
      int nbf = libintx::nbf(x);
      int kherm = Q_.blocks.at(l).primitives.size()*nherm1(x.L);
      for (int k = 0; k < nprim; ++k) {
        auto [a,C] = x.prims[k];
        Primitive2 q = {
                        .exp = { a, 0 },
                        .r = { r, {} },
                        .C = C,
                        .norm = max
                        // .kbf = (int)kbf,
                        // .pure = (bool)x.pure
        };
        Q_.blocks.at(l).primitives.push_back(q);
      }
      Q_.blocks.at(l).index.push_back(
        Index1{
          .shell=(int)i,
          .kbf=(int)kbf,
          .kherm=kherm
        }
      );
      Q_.basis.push_back(
        Shell{
         .L = (uint8_t)l,
         .pure = (bool)x.pure,
         .r = r,
         .prims = x.prims,
         .K = x.K,
         .range = { kbf, kbf+nbf }
        }
      );
      // printf("df %i L=%i K=%i bf=[%i:%i] herm=[%i:%i]\n",
      //        i, l, x.K, kbf, kbf+nbf, kherm, kherm+x.K*nherm1(l));
      kbf += nbf;
      kherm += x.K*nherm1(l);
      Q_.nbf += nbf;
      Q_.nherm += x.K*nherm1(l);
    }
    for (size_t L = 0, kherm = 0; L < Q_.blocks.size(); ++L) {
      Q_.blocks.at(L).L = L;
      Q_.blocks.at(L).kherm = kherm;
      kherm += nherm1(L)*Q_.blocks.at(L).primitives.size();
      //printf("L=%i, n=%i\n", L, Q_.blocks.at(L).index.size());
    }

    // Bra

    int pmax = 0;
    for (size_t i = 0, kbf = 0; i < basis.size(); ++i) {
      const auto &[a,r] = basis[i];
      //printf("obs %i L=%i K=%i\n", i, a.L, a.K);
      Shell s;
      s.L = a.L;
      s.pure = a.pure;
      s.prims = a.prims;
      s.K = a.K;
      s.r = r;
      s.range = { kbf, kbf+nbf(s) };
      AB_.basis.push_back(s);
      pmax = std::max(pmax, 2*s.L);
      kbf += nbf(s);
    }

    for (int i = 0; i <= pmax; ++i) {
      AB_.pairs.push_back({i,{}});
    }

    for (size_t i = 0; i < AB_.basis.size(); ++i) {
      for (size_t j = 0; j <= i; ++j) {
        const auto &A = AB_.basis[i];
        const auto &B = AB_.basis[j];
        auto &[P,pairs] = AB_.pairs.at(A.L + B.L);
        float max = (screening ? screening->max2(i,j) : 1);
        if (max*this->Q_.max < this->cutoff_) continue;
        pairs.push_back({i,j,max});
      }
    }

    std::reverse(AB_.pairs.begin(), AB_.pairs.end());

    std::remove_if(
      AB_.pairs.begin(), AB_.pairs.end(),
      [](auto &&p) { return std::get<1>(p).empty(); }
    );

    // initialise boys
    gpu::boys();

  }

  template<typename It>
  struct JEngine::ab_pairs_iterator {

    ab_pairs_iterator(It begin, It end)
      : it_(begin), end_(end) {}

    auto next(const std::vector<Shell> &basis) {
      std::unique_lock lock(mutex_);
    start:
      using iterator = decltype(std::get<1>(*it_).cbegin());

      if (it_ == end_) {
        return std::tuple{int(0),range{iterator(),iterator()}};
      }

      const auto& [p,ab] = *it_;
      auto idx = this->idx_;

      iterator begin = ab.begin()+idx;
      iterator end = begin;
      size_t kg = 0, kh = 0;
      size_t np = nherm2(p);

      while (true) {
        if (ab.begin()+idx == ab.end()) {
          ++it_;
          idx = 0;
          break;
        }
        auto [i,j,norm] = ab.at(idx);
        const auto &A = basis.at(i);
        const auto &B = basis.at(j);
        size_t K = A.K*B.K;
        size_t ng = nbf(A)*nbf(B);
        if (maxij && size_t(end-begin)+1 > maxij) break;
        if (maxg && kg+ng > maxg) break;
        if (maxh && kh+K*np > maxh) break;
        //printf("next p=%i idx=%i %i %i\n", p, idx, i, j);
        ++idx;
        ++end;
        kg += ng;
        kh += K*np;
      }
      this->idx_ = idx;
      if (begin == end) goto start;
      return std::tuple{int(p),range{begin,end}};
    }

    size_t maxg = 0;
    size_t maxh = 0;
    size_t maxij = 0;

  private:
    It it_, end_;
    size_t idx_ = 0;
    std::mutex mutex_;
  };

  void JEngine::J(const TileIn &D, const TileOut &J, const AllSum& allsum) {

    auto t = time::now();

    gpu::host::register_pointer(AB_.basis.data(), AB_.basis.size());
    gpu::host::register_pointer(Q_.basis.data(), Q_.basis.size());
    for (const auto &q : Q_.blocks) {
      if (q.index.empty()) continue;
      gpu::host::register_pointer(q.index.data(), q.index.size());
    }

    // printf("JEngine-X t(init)=%f\n", time::since(t));
    int num_devices = gpu::device::count();
    thread_pool threads(THREADS_PER_DEVICE*num_devices);

    host::vector<double> X(Q_.nbf);
    X.memset(0);

    t = time::now();
    {
      ab_pairs_iterator ab_pairs{AB_.pairs.rbegin(), AB_.pairs.rend()};
      ab_pairs.maxij = this->maxij;
      ab_pairs.maxg = this->maxg;
      for (int i = 0; i < THREADS_PER_DEVICE*num_devices; ++i) {
        auto task = [&](){ compute_x(D, X.data(), ab_pairs, 0); };
        threads.push_task(task);
      }
    }
    threads.wait();
    printf("J-Engine X: %f\n", time::since(t));

    allsum(X.data(), X.size());

    t = time::now();
    v_transform_(X.data());
    printf("J-Engine V*X: %f\n", time::since(t));

    t = time::now();
    {
      ab_pairs_iterator ab_pairs{AB_.pairs.begin(), AB_.pairs.end()};
      ab_pairs.maxij = 128*1024;
      ab_pairs.maxg = 16*1024*1024/8;
      for (int i = 0; i < THREADS_PER_DEVICE*num_devices; ++i) {
        auto task = [&](){ compute_j(X.data(), J, ab_pairs, 0); };
        threads.push_task(task);
      }
    }
    threads.wait();
    printf("J-Engine J: %f\n", time::since(t));

    gpu::host::unregister_pointer(AB_.basis.data());
    gpu::host::unregister_pointer(Q_.basis.data());
    for (const auto &q : Q_.blocks) {
      gpu::host::unregister_pointer(q.index.data());
    }

  }


  template<typename It>
  void JEngine::compute_x(
    const TileIn &D, double *X,
    ab_pairs_iterator<It> &ab_pairs,
    int device_id)
  {

    auto t = time::now();

    struct {
      double kernel = 0;
      double ijs = 0, D = 0, P = 0;
      double sync = 0;
      decltype(t) start = time::now();
    } times;

    gpu::current_device::set(device_id);
    Stream stream;

    const auto &boys = gpu::boys();

    device::vector<double> Xq;
    Xq.resize(Q_.nherm);
    Xq.memset(0);

    std::vector<double> G2;
    G2.reserve(ab_pairs.maxg);

    std::vector<Index2> ijs;
    ijs.reserve(ab_pairs.maxij);

    struct {
      struct Q {
        int L;
        size_t kherm;
        device::vector<Primitive2> primitives;
      };
      std::vector<Q> Q;
      device::vector<Index2> ijs;
      device::vector<Primitive2> P;
      device::vector<double> G2;
      device::vector<double> H2;
    } stream_data;

    for (auto &Q : Q_.blocks) {
      stream_data.Q.push_back({Q.L, Q.kherm});
      stream_data.Q.back().primitives.assign(Q.primitives.data(), Q.primitives.size());
    }
    stream_data.ijs.reserve(ijs.capacity());
    stream_data.G2.reserve(G2.capacity());

    while (true) {

      auto [p,ab] = ab_pairs.next(AB_.basis);
      if (ab.begin() == ab.end()) break;

      //printf("Engine-X task p=%i\n", p);

      size_t kbf = 0, kprim = 0;
      ijs.clear();
      for (auto [i,j,norm] : ab) {
        //printf("%i,%i,ijs=%i\n", i,j,ijs.size());
        const auto &A = AB_.basis.at(i);
        const auto &B = AB_.basis.at(j);
        if (!D(A.range,B.range,nullptr)) continue;
        int K = A.K*B.K;
        int n = nbf(A)*nbf(B);
        ijs.push_back(
          {
            .first=i,
            .second=j,
            .L={A.L,B.L},
            .kbf=(int)kbf,
            .kprim=(int)kprim,
            .norm=norm
          }
        );
        kbf += n;
        kprim += K;
      }
      times.ijs += time::since(t);

      if (!ijs.size()) continue;

      t = time::now();
      G2.resize(kbf);
      for (auto &ij : ijs) {
        const auto &A = AB_.basis[ij.first];
        const auto &B = AB_.basis[ij.second];
        D(A.range, B.range, G2.data()+ij.kbf);
      }
      times.D += time::since(t);

      t = time::now();
      stream.synchronize();
      times.sync += time::since(t);

      t = time::now();
      stream_data.G2.assign(G2.data(), kbf);
      stream_data.ijs.assign(ijs.data(), ijs.size());
      stream_data.P.resize(kprim);
      stream_data.H2.resize(kprim*nherm2(p));
      cartesian_to_hermite_2(
        p,
        stream_data.ijs.size(),
        stream_data.ijs.data(),
        gpu::host::device_pointer(AB_.basis.data()),
        stream_data.P.data(),
        stream_data.G2.data(),
        stream_data.H2.data(),
        stream
      );
      times.P += time::since(t);

      // t = time::now();
      // stream.synchronize();
      // times.sync += time::since(t);

      t = time::now();
      for (const auto& Q : stream_data.Q) {
        df_jengine_kernel<1>(
          p, Q.L,
          boys,
          stream_data.P.size(), stream_data.P.data(),
          Q.primitives.size(), Q.primitives.data(),
          stream_data.H2.data(),
          Xq.data()+Q.kherm,
          this->cutoff_,
          stream
        );
      }
      //stream.synchronize();
      times.kernel += time::since(t);

    }

    stream.synchronize();

    for (const auto &q : Q_.blocks) {
      hermite_to_cartesian_1(
        q.L, q.index.size(),
        gpu::host::device_pointer(q.index.data()),
        gpu::host::device_pointer(Q_.basis.data()),
        Xq.data()+q.kherm,
        gpu::host::device_pointer(X)
      );
    }

  }


  template<typename It>
  void JEngine::compute_j(
    const double *X, const TileOut &J,
    ab_pairs_iterator<It> &ab_pairs,
    int device_id)
  {

    // // printf("JEngine::J\n");

    auto t = time::now();

    struct {
      decltype(t) start = time::now();
      double sync = 0;
      double jsync = 0;
      double ijs = 0, J = 0, P = 0, Jp = 0;
      double mem = 0;
      double main = 0;
    } times;

    // printf("JEngine-X task device=%i at %f\n", device_id, time::since(t));

    gpu::current_device::set(device_id);
    Stream stream;
    Stream jstream;

    // printf("Engine-X task device=%i set at %f\n", device_id, time::since(t));

    const auto &boys = gpu::boys();

    struct {
      struct Q {
        int L;
        size_t kherm;
        device::vector<Primitive2> primitives;
      };
      std::vector<Q> Q;
      device::vector<Index2> ijs;
      device::vector<Primitive2> P;
      device::vector<double> H2;
      device::vector<double> Xh;
    } stream_data;

    // stream_data.P.reserve(5000000);
    // stream_data.H2.resize(5000000);
    // stream_data.H2.memset(0);

    stream_data.Xh.resize(Q_.nherm);
    for (const auto &q : Q_.blocks) {
      cartesian_to_hermite_1(
        q.L, q.index.size(),
        gpu::host::device_pointer(q.index.data()),
        gpu::host::device_pointer(Q_.basis.data()),
        gpu::host::device_pointer(X),
        stream_data.Xh.data()+q.kherm
      );
    }

    host::vector<double> G2;
    G2.reserve(ab_pairs.maxg);

    host::vector<Index2> ijs, ijs0;
    ijs.reserve(ab_pairs.maxij);
    ijs0.reserve(ijs.capacity());

    struct {
      host::vector<double> G2;
      host::vector<Index2> ijs;
    } j_update;
    j_update.G2.reserve(G2.capacity());
    j_update.ijs.reserve(ijs.capacity());

    for (auto &Q : Q_.blocks) {
      stream_data.Q.push_back({Q.L, Q.kherm});
      stream_data.Q.back().primitives.assign(Q.primitives.data(), Q.primitives.size());
    }
    stream_data.ijs.reserve(ijs.capacity());

    gpu::Event event;

    while (true) {

      auto t = time::now();

      auto [p,ab] = ab_pairs.next(AB_.basis);
      if (ab.begin() == ab.end()) break;

      size_t kbf = 0, kprim = 0;
      ijs0.clear();
      for (auto [i,j,norm] : ab) {
        //printf("%i,%i,ijs=%i\n", i,j,ijs.size());
        const auto &A = AB_.basis.at(i);
        const auto &B = AB_.basis.at(j);
        if (!J(A.range,B.range,nullptr)) continue;
        int K = A.K*B.K;
        int n = nbf(A)*nbf(B);
        ijs0.push_back(
          {
            .first=i,
            .second=j,
            .L={A.L,B.L},
            .kbf=(int)kbf,
            .kprim=(int)kprim,
            .norm=norm
          }
        );
        kbf += n;
        kprim += K;
      }
      times.ijs += time::since(t);

      if (!ijs0.size()) continue;

      // t = time::now();
      // event.synchronize();
      // times.sync += time::since(t);

      stream.synchronize();

      t = time::now();
      ijs.swap(ijs0);
      stream_data.ijs.resize(ijs.size());
      copy(ijs.begin(), ijs.end(), stream_data.ijs.data(), stream);
      //stream_data.ijs.assign(ijs.data(), ijs.size(), stream);
      stream_data.P.resize(kprim);
      times.mem += time::since(t);

      cartesian_to_hermite_2(
        p,
        stream_data.ijs.size(),
        stream_data.ijs.data(),
        gpu::host::device_pointer(AB_.basis.data()),
        stream_data.P.data(),
        nullptr,
        nullptr,
        stream
      );

      t = time::now();
      stream_data.H2.resize(kprim*nherm2(p));
      memset(stream_data.H2, 0, stream);
      times.mem += time::since(t);

      for (const auto& Q : stream_data.Q) {
        df_jengine_kernel<2>(
          p, Q.L, boys,
          stream_data.P.size(), stream_data.P.data(),
          Q.primitives.size(), Q.primitives.data(),
          stream_data.Xh.data()+Q.kherm,
          stream_data.H2.data(),
          this->cutoff_,
          stream
        );
      }

      G2.resize(kbf);
      hermite_to_cartesian_2(
        p,
        stream_data.ijs.size(),
        stream_data.ijs.data(),
        gpu::host::device_pointer(AB_.basis.data()),
        stream_data.H2.data(),
        gpu::host::device_pointer(G2.data()),
        stream
      );

      t = time::now();
      jstream.synchronize();
      times.jsync += time::since(t);

      stream.wait(event);

      j_update.ijs.swap(ijs);
      j_update.G2.swap(G2);
      jstream.add_callback(
        [&](auto ...){
          event.synchronize();
          for (auto &ij : j_update.ijs) {
            const auto &A = AB_.basis[ij.first];
            const auto &B = AB_.basis[ij.second];
            J(A.range, B.range, j_update.G2.data()+ij.kbf);
          }
        }
      );

    } // while(true)

    {
      auto t = time::now();
      jstream.synchronize();
      times.jsync += time::since(t);
    }

    times.main += time::since(t);

  }

  // printf("JEngine-J t(J)=%f\n", times.J);
  // printf("JEngine-J t(ijs)=%f\n", times.ijs);
  // printf("JEngine-J t(mem)=%f\n", times.mem);
  // printf("JEngine-J t(sync)=%f\n", times.sync);
  // printf("JEngine-J t(jsync)=%f\n", times.jsync);
  // printf("JEngine-J t(loop)=%f\n", times.main);
  // printf("JEngine-J t(total)=%f\n", time::since(times.start));

}

std::unique_ptr<libintx::JEngine> libintx::gpu::make_jengine(
  const std::vector< std::tuple< Gaussian, Double<3> > > &basis,
  const std::vector< std::tuple< Gaussian, Double<3> > > &df_basis,
  std::function<void(double*)> v_transform,
  std::shared_ptr<const libintx::JEngine::Screening> screening)
{
  return std::make_unique<jengine::md::JEngine>(basis, df_basis, v_transform, screening);
}
