#include "libintx/gpu/md/md3.h"
#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"

namespace libintx::gpu::md {

  struct ERI3::Memory {
    device::vector<Hermite> p;
    device::vector<double> q;
    std::array<device::vector<double>,1> buffer;
  };

  ERI3::ERI3(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, gpuStream_t stream) {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
    memory_.reset(new Memory);
  }

  ERI3::~ERI3() {}

  void ERI3::compute(
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    double *V,
    std::array<size_t,2> dims)
  {

    using Kernel = std::function<void(
      ERI3&, const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    static auto x_cd_kernels = make_array<Kernel,XMAX+1,2*LMAX+1>(
      [](auto x, auto cd) {
        return &ERI3::compute<x,cd>;
      }
    );

    auto stream = this->stream_;
    auto p = make_basis(bra_, bra, this->memory_->p, stream);
    auto q = make_basis(ket_, ket_, ket, this->memory_->q, stream);
    auto kernel = x_cd_kernels[p.L][q.first.L+q.second.L];
    kernel(*this, p, q, TensorRef{V,dims}, stream);

  }

  template<int Idx>
  double* ERI3::allocate(size_t size) {
    auto &v = std::get<Idx>(this->memory_->buffer);
    v.resize(size);
    return v.data();
  }

  template double* ERI3::allocate<0>(size_t size);

  template<>
  std::unique_ptr< IntegralEngine<1,2> > integral_engine<1,2>(
    const Basis<Gaussian> &bra,
    const Basis<Gaussian> &ket,
    const gpuStream_t &stream)
  {
    return std::make_unique<ERI3>(bra, ket, stream);
  }

}
