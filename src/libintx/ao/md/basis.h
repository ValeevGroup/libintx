#ifndef LIBINTX_AO_MD_BASIS_H
#define LIBINTX_AO_MD_BASIS_H

#include "libintx/shell.h"
#include "libintx/ao/md/engine.h"

namespace libintx::md {

  template<int Centers, typename T = double, int Batch = 1>
  struct HermiteBasis;

  template<typename T = double>
  struct alignas(T) Hermite {

    static constexpr auto Lanes = []() {
      if constexpr (std::is_scalar_v<T>) return std::integral_constant<int,1>{};
      else return std::integral_constant<int,T::size()>{};
    }();

    T exp;
    T C;
    array<T,3> r;
    T inv_2_exp;
    double norm = 0;

    LIBINTX_GPU_ENABLED
    static auto* hdata(T *p) {
      return reinterpret_cast<Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* hdata(const T *p) {
      return reinterpret_cast<const Hermite*>(p);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(T *p) {
      return reinterpret_cast<T*>(hdata(p)+1);
    }

    LIBINTX_GPU_ENABLED
    static auto* gdata(const T *p) {
      return reinterpret_cast<const T*>(hdata(p)+1);
    }

    LIBINTX_GPU_ENABLED
    static constexpr size_t extent(const Shell &A, const Shell &B) {
      return (sizeof(Hermite)/sizeof(T) + nbf(A)*nbf(B)*nherm2(A.L+B.L));
    }

  };

  template<typename T>
  struct HermiteBasis<1,T> {
    const int L, K, N;
    int Batch;
    const Hermite<T> *data;
    const Hermite<T>* hermite(int i, int k) const {
      return (this->data + k + i*this->K);
    }
    const auto batch(size_t idx) const {
      return HermiteBasis<1,T>{ L, K, Batch, 1, data+idx*K*Batch };
    }
  };

  template<typename T>
  struct HermiteBasis<2,T> {
    using Hermite = md::Hermite<T>;
    Shell first, second;
    int K, N;

    static constexpr auto Lanes = Hermite::Lanes;
    //HermiteBasis() = default;
    // HermiteBasis(const Shell &first, const Shell &second, int K, int N) {
    //   init(first, second, K, N);
    // }

    const Hermite* hermite(int i, int k) const {
      return reinterpret_cast<const Hermite*>(data_ + (k + i*this->K)*extent_);
    }

    const T* hermite_to_ao(int i, int k) const {
      return reinterpret_cast<const T*>(hermite(i,k) + 1);
    }

    const auto batch(size_t idx, size_t batch = 1) const {
      assert(idx*batch*Lanes < this->N);
      int M = std::min(batch*Lanes, this->N-idx*batch*Lanes);
      //printf("idx=%i batch=%i, N=%i, M=%i\n", idx, batch, N, M);
      return HermiteBasis<2,T>{ first, second, K, M, extent_, data_+idx*batch*K*extent_ };
    }

    //private:
    size_t extent_;
    T* data_ = nullptr;
  };

  template<typename T = double>
  HermiteBasis<1,T> make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    int Batch,
    std::vector< Hermite<T> >&
  );

  template<typename T = double>
  HermiteBasis<2,T> make_basis(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    const double *norms,
    Phase<int> phase,
    int Batch,
    std::vector<T>&
  );

}


#endif /* LIBINTX_AO_MD_BASIS_H */
