#include "libintx/gpu/md/engine.h"
#include "libintx/gpu/md/basis.h"
#include "libintx/config.h"
#include "libintx/utility.h"
#include <functional>

namespace libintx::gpu::md {

  struct IntegralEngine<4>::Memory {
    device::vector<double> p;
    device::vector<double> q;
    std::array<device::vector<double>,3> buffer;
  };

  IntegralEngine<4>::IntegralEngine(gpuStream_t stream) {
    stream_ = stream;
    memory_.reset(new Memory);
  }

  IntegralEngine<4>::IntegralEngine(const Basis<Gaussian> &bra, const Basis<Gaussian> &ket, gpuStream_t stream)
    : IntegralEngine(stream)
  {
    // libintx_assert(!bra.empty());
    // libintx_assert(!ket.empty());
    bra_ = bra;
    ket_ = ket;
  }

  IntegralEngine<4>::~IntegralEngine() {}

  void IntegralEngine<4>::compute(
    Operator,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {
    auto stream = this->stream_;
    HermiteBasis p; p.init({bra_, bra_}, bra, norms.bra, stream);
    HermiteBasis q; q.init({ket_, ket_}, ket, norms.ket, stream);
    this->compute(p, q, TensorRef{V,dims}, stream);
  }

  void IntegralEngine<4>::compute(
    const HermiteBasis& bra, const HermiteBasis& ket,
    TensorRef<double,2> V,
    gpuStream_t stream)
  {
    using Kernel = std::function<void(
      IntegralEngine&,
      const HermiteBasis&, const HermiteBasis&,
      TensorRef<double,2>, gpuStream_t
    )>;
    static auto ab_cd_kernels = make_array<Kernel,2*LMAX+1,2*LMAX+1>(
      [](auto ab, auto cd) {
        return &IntegralEngine::compute<ab,cd>;
      }
    );
    auto kernel = ab_cd_kernels[bra.first.L+bra.second.L][ket.first.L+ket.second.L];
    kernel(*this, bra, ket, V, stream);
  }

  template<int Idx>
  double* IntegralEngine<4>::allocate(size_t size) {
    auto &v = std::get<Idx>(this->memory_->buffer);
    v.resize(size);
    return v.data();
  }

  template double* IntegralEngine<4>::allocate<0>(size_t size);
  template double* IntegralEngine<4>::allocate<1>(size_t size);
  template double* IntegralEngine<4>::allocate<2>(size_t size);

}

template<>
std::unique_ptr< libintx::gpu::IntegralEngine<4> > libintx::gpu::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket,
  const gpuStream_t &stream)
{
  return std::make_unique< gpu::md::IntegralEngine<4> >(bra, ket, stream);
}
