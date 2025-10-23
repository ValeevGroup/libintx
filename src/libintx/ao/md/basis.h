#ifndef LIBINTX_AO_MD_BASIS_H
#define LIBINTX_AO_MD_BASIS_H

#include "libintx/shell.h"
#include "libintx/ao/md/engine.h"
#include "libintx/ao/screening.h"

namespace libintx::md {

  template<int Centers>
  struct HermiteBasis<Centers> {
    virtual ~HermiteBasis() = default;
  };

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
    T norm = T(0);

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
  struct HermiteBatch {
    using Hermite = md::Hermite<T>;
    static constexpr auto Lanes = Hermite::Lanes;
    float norm = math::infinity<float>;
    int K, N;
    const Hermite* hermite(int i, int k) const {
      return reinterpret_cast<const Hermite*>(data_.get() + (k + i*this->K)*extent_);
    }
    const T* hermite_to_ao(int i, int k) const {
      return reinterpret_cast<const T*>(hermite(i,k) + 1);
    }
    size_t extent_;
    std::shared_ptr<T[]> data_;
  };

  template<typename T>
  struct HermiteBasis<2,T> : HermiteBasis<2> {

    Shell first, second;
    int K;
    int Batch;
    std::vector< HermiteBatch<T> > batches;

    HermiteBasis() = default;

    HermiteBasis(const Shell &first, const Shell &second, int K, int Batch)
      : first(first), second(second), K(K), Batch(Batch)
    {
    }

  };

  template<typename T = double>
  HermiteBasis<1,T> make_basis(
    const Basis<Gaussian> &A,
    const std::vector<Index1> &idx,
    int Batch,
    std::vector< Hermite<T> >&
  );

  template<typename T>
  HermiteBatch<T> make_basis_batch(
    const Basis<Gaussian> &A,
    const Basis<Gaussian> &B,
    const std::vector<Index2> &pairs,
    const double *norms,
    ao::screening::Norm<T> primitive_norm,
    Phase<int> phase
  );

  template<typename T>
  std::shared_ptr< HermiteBasis<2,T> > make_batch_basis(
    int Batch,
    const Basis<Gaussian> &first,
    const Basis<Gaussian> &second,
    const std::vector<Index2>&,
    const double *norms,
    ao::screening::Norm<T> primitive_norm,
    Phase<int> phase,
    libintx::num_threads = { 1 }
  );

}


#endif /* LIBINTX_AO_MD_BASIS_H */
