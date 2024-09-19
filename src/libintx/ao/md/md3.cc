#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/md3.kernel.h"
#include "libintx/simd.h"

#include "libintx/config.h"
#include "libintx/utility.h"

namespace libintx::md {

  template<Operator Op, typename Params>
  void IntegralEngine<3>::compute(
    const Params &params,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    const Visitor &V)
  {

    const auto &cd = make_basis< pair<const Gaussian&> >(basis(1), basis(2), ket);
    auto &X = basis(0)[bra.front()];
    auto &C = cd.front().first;
    auto &D = cd.front().second;

    using Kernel = kernel::Kernel<Op,Params>;
    using T = typename Kernel::simd_t;
    const int Lanes = Kernel::Lanes;
    const int Batch = Kernel::batch(X.L,C.L,D.L);

    size_t n_p_batches = (bra.size() + Lanes*Batch - 1)/(Lanes*Batch);
    std::vector< Hermite<T> > p_allocator;
    auto p = make_basis<T>(basis(0), bra, Batch, p_allocator);
    p.Batch = Batch;

#pragma omp parallel num_threads(this->num_threads);
    {

      // int K = nprim(a)*nprim(b);
      int ldV = Lanes*Batch;
      std::unique_ptr<T[]> V_batch(
        new (std::align_val_t{64}) T[Batch*npure(X.L)*npure(C.L,D.L)]
      );

      std::vector<double> q_allocator;
      HermiteBasis<2> q;

#pragma omp for schedule(dynamic,1)
      for (size_t kl = 0; kl < cd.size(); ++kl) {
        auto &[C,D] = cd[kl];
        auto q = make_basis<double>(
          basis(1), basis(2),
          { ket[kl] }, nullptr,
          Phase<int>{-1},
          Batch, q_allocator
        );
        std::unique_ptr<Kernel> kernel;
        jump_table(
          std::make_index_sequence<(XMAX+1)>{},
          std::make_index_sequence<(LMAX+1)*(LMAX+1)>{},
          X.L, C.L+D.L*(LMAX+1),
          [&](auto X, auto CD) {
            constexpr int C = CD%(LMAX+1);
            constexpr int D = CD/(LMAX+1);
            kernel = md::kernel::make_kernel<X,C,D,Op,Params>();
          }
        );
        for (size_t i_batch = 0; i_batch < n_p_batches; ++i_batch) {
          std::fill_n(V_batch.get(), Batch*npure(X.L)*npure(C.L,D.L), 0);
          kernel->compute(params, p.batch(i_batch), q, V_batch.get());
          for (int i = 0; i < 1; ++i) {
            //printf("%i = %f\n", i, (double)V_batch[0]);
          }
          BraKet<size_t> idx = { i_batch*Lanes*Batch, kl };
          auto batch_size = std::min<size_t>(Lanes*Batch, bra.size() - idx.bra);
          V(batch_size, idx, reinterpret_cast<double*>(V_batch.get()), ldV);
        }
      }

    } // omp parallel

  }

  void IntegralEngine<3>::compute(
    Operator op,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    const Visitor &V)
  {
    //using T = double;
    assert(op == Coulomb);
    if (op == Coulomb) {
      this->compute<Coulomb>(Coulomb::Operator::Parameters{}, bra, ket, norms, V);
    }
  }

  void IntegralEngine<3>::compute(
    Operator op,
    const std::vector<Index1> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    double *V,
    const std::array<size_t,2> &dims)
  {
    size_t NA = nbf(basis(0)[bra[0]]);
    std::vector<size_t> ket_start = { 0 };
    for (auto [k,l] : ket) {
      size_t idx = nbf(basis(1)[k])*nbf(basis(2)[l]) + ket_start.back();
      ket_start.push_back(idx);
    }
    auto v = [&](size_t batch, BraKet<size_t> idx, const double *U, size_t ldU) {
      auto [k,l] = ket[idx.ket];
      auto &c = basis(1)[k];
      auto &d = basis(2)[l];
      size_t ncd = nbf(c)*nbf(d);
      for (size_t icd = 0; icd < ncd; ++icd) {
        for (size_t ia = 0; ia < NA; ++ia) {
          const auto *src =  U + (ia + icd*NA)*ldU;
          //printf("%i\n", icd + ket_start[idx.ket]);
          for (size_t i = 0; i < batch; ++i) {
            //printf("%f\n", src[i]);
          }
          auto *dst = V + idx.bra + ia*bra.size() + (icd + ket_start[idx.ket])*dims[0];
          std::copy_n(src, batch, dst);
        }
      }
    };
    this->compute(op, bra, ket, norms, v);
  }

  IntegralEngine<3>::IntegralEngine(const std::shared_ptr< Basis<Gaussian> > (&basis)[3])
    : basis_{ basis[0], basis[1], basis[2] }
  {
    libintx_assert(basis[0] && !basis[0]->empty());
    libintx_assert(basis[1] && !basis[1]->empty());
    libintx_assert(basis[2] && !basis[2]->empty());
  }

  IntegralEngine<3>::~IntegralEngine() {}

} // libintx::md

template<>
std::unique_ptr< libintx::ao::IntegralEngine<3> > libintx::ao::integral_engine(
  const Basis<Gaussian> &bra,
  const Basis<Gaussian> &ket)
{
  return std::make_unique< libintx::md::IntegralEngine<3> >(
    std::make_shared< Basis<Gaussian> >(bra),
    std::make_shared< Basis<Gaussian> >(ket)
  );
}
