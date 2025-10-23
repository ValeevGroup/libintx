#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/md4.kernel.h"

#include "libintx/config.h"
#include "libintx/tensor.h"
#include "libintx/utility.h"

namespace libintx::md {

  std::shared_ptr< HermiteBasis<2> > IntegralEngine<4>::make_bra(
    const std::vector<Index2> &pairs,
    const double* norms) const
  {
    return make_batch_basis<kernel::simd_t>(
      simd::size<kernel::simd_t>,
      this->basis(0),
      this->basis(1),
      pairs,
      norms, nullptr, // primitive_norm
      Phase<int>{+1},
      this->num_threads
    );
  }


  std::shared_ptr< HermiteBasis<2> > IntegralEngine<4>::make_ket(
    const std::vector<Index2> &pairs,
    const double* norms) const
  {
    Shell C = this->basis(2)[pairs.at(0).first];
    Shell D = this->basis(3)[pairs.at(0).second];
    return make_batch_basis<double>(
      kernel::Ket::batch(C.L, D.L),
      this->basis(2),
      this->basis(3),
      pairs,
      norms, nullptr, // primitive_norm
      Phase<int>{-1},
      this->num_threads
    );
  }

  template<Operator Op, typename Params>
  void IntegralEngine<4>::compute(
    const Params &params,
    const HermiteBasis<2> &bra2,
    const HermiteBasis<2> &ket2,
    const Visitor &V)
  {

    using simd_t = typename kernel::simd_t;

    auto &bra = dynamic_cast<const HermiteBasis<2,simd_t>&>(bra2);
    auto &ket = dynamic_cast<const HermiteBasis<2,double>&>(ket2);

    int A = bra.first.L;
    int B = bra.second.L;
    int C = ket.first.L;
    int D = ket.second.L;

    libintx_assert(A <= LMAX);
    libintx_assert(B <= LMAX);
    libintx_assert(C <= LMAX);
    libintx_assert(D <= LMAX);

    size_t computed = 0;
    size_t screened = 0;

#pragma omp parallel num_threads(int{this->num_threads})
    {

      auto kernel = kernel::make_kernel<Op,simd_t>(A, B, C, D);

      size_t NA = npure(A);
      size_t NB = npure(B);
      size_t NC = npure(C);
      size_t ND = npure(D);

      // int K = nprim(a)*nprim(b);
      std::vector<simd_t> V_batch((bra.Batch*ket.Batch*NA*NB*NC*ND)/simd::size<simd_t>);

#pragma omp for collapse(2) schedule(dynamic,1), reduction(+:screened,computed)
      for (size_t kl = 0; kl < ket.batches.size(); ++kl) {
        for (size_t ij = 0; ij < bra.batches.size(); ++ij) {
          auto &p = bra.batches[ij];
          auto &q = ket.batches[kl];
          //println("*",p.norm, q.norm, precision);
          if (p.norm*q.norm < precision) {
            screened += p.N*q.N;
            continue;
          }
          computed += p.N*q.N;
          std::fill(V_batch.begin(), V_batch.end(), 0.0);
          kernel->compute({}, p, q, V_batch.data());
          BraKet<Index1> idx = { int(ij*bra.Batch), int(kl*ket.Batch) };
          BraKet<size_t> dims = { size_t(p.N), size_t(q.N) };
          std::array<size_t,6> dims6 = {
            (size_t)bra.Batch,
            NA, NB, NC, ND,
            dims.ket
          };
          V(idx, dims, TensorRef{ reinterpret_cast<const double*>(V_batch.data()), dims6 });
        }
      } // kl_batch

    } // omp parallel

    this->screened += screened;
    this->computed += computed;

  }

  void IntegralEngine<4>::compute(
    Operator Op,
    const HermiteBasis<2> &bra,
    const HermiteBasis<2> &ket,
    const Visitor &V)
  {
    if (Op == Coulomb) {
      Coulomb::Operator::Parameters params;
      this->compute<Coulomb>(params, bra, ket, V);
    }
  }

  void IntegralEngine<4>::compute(
    Operator Op,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {
    size_t NAB = nbf(basis(0)[bra[0].first])*nbf(basis(1)[bra[0].second]);;
    std::vector<size_t> ket_start = { 0 };
    for (auto [k,l] : ket) {
      size_t idx = nbf(basis(2)[k])*nbf(basis(3)[l]) + ket_start.back();
      ket_start.push_back(idx);
    }
    auto v = [&](BraKet<Index1> idx, BraKet<size_t> batch, const auto &U) {
      size_t ncd = 0;
      for (size_t kl = 0; kl < batch.ket; ++kl) {
        auto [k,l] = ket[idx.ket+kl];
        ncd += nbf(basis(2)[k])*nbf(basis(3)[l]);
      }
      for (size_t icd = 0; icd < ncd; ++icd) {
        for (size_t iab = 0; iab < NAB; ++iab) {
          const auto *src =  U.data() + (iab + icd*NAB)*U.dimensions()[0];
          auto *dst = V + idx.bra + iab*bra.size() + (icd + ket_start[idx.ket])*dims[0];
          std::copy_n(src, batch.bra, dst);
        }
      }
    };
    this->compute(
      Op,
      *this->make_bra(bra, norms.bra),
      *this->make_ket(ket, norms.ket),
      v
    );
  }

  IntegralEngine<4>::IntegralEngine(const std::shared_ptr< Basis<Gaussian> > (&basis)[4])
  : basis_{basis[0], basis[1], basis[2], basis[3]}
  {
    for (auto &b : basis_) {
      libintx_assert(b);
      libintx_assert(!b->empty());
    }
  }

  IntegralEngine<4>::~IntegralEngine() {}

}

template<>
std::unique_ptr< libintx::ao::IntegralEngine<4> > libintx::ao::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket)
{
  std::shared_ptr< Basis<Gaussian> > braket[4];
  braket[0] = braket[1] = std::make_shared< Basis<Gaussian> >(bra);
  braket[2] = braket[3] = std::make_shared< Basis<Gaussian> >(ket);
  return std::make_unique< libintx::md::IntegralEngine<4> >(braket);
}
