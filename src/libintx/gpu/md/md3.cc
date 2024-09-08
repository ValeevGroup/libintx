#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include <functional>

namespace libintx::gpu::md {

  struct IntegralEngine<3>::Memory {
    device::vector<Hermite> p;
    device::vector<double> q;
    std::array<device::vector<double>,1> buffer;
  };

  IntegralEngine<3>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, gpuStream_t stream) {
    libintx_assert(!bra.empty());
    libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
    stream_ = stream;
    memory_.reset(new Memory);
  }

  IntegralEngine<3>::~IntegralEngine() {}

  void IntegralEngine<3>::compute(
    Operator op,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    double *V,
    const std::array<size_t,2> &dims)
  {

    using Kernel = std::function<void(
      IntegralEngine&, const Basis1&, const Basis2&, TensorRef<double,2>, gpuStream_t
    )>;

    static auto x_cd_kernels = make_array<Kernel,XMAX+1,2*LMAX+1>(
      [](auto x, auto cd) {
        return &IntegralEngine::compute<x,cd>;
      }
    );

    auto stream = this->stream_;
    auto p = make_basis(bra_, bra, this->memory_->p, stream);
    auto q = make_basis(ket_, ket_, ket, this->memory_->q, stream);
    auto kernel = x_cd_kernels[p.L][q.first.L+q.second.L];
    kernel(*this, p, q, TensorRef{V,dims}, stream);

  }

  template<int Idx>
  double* IntegralEngine<3>::allocate(size_t size) {
    auto &v = std::get<Idx>(this->memory_->buffer);
    v.resize(size);
    return v.data();
  }

  template double* IntegralEngine<3>::allocate<0>(size_t size);

} // libintx::gpu::md

template<>
std::unique_ptr< libintx::gpu::IntegralEngine<3> > libintx::gpu::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket,
  const gpuStream_t &stream)
{
  return std::make_unique< gpu::md::IntegralEngine<3> >(bra, ket, stream);
}
