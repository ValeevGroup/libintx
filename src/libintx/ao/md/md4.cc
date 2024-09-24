#include "libintx/ao/md/engine.h"
#include "libintx/ao/md/basis.h"
#include "libintx/ao/md/md4.kernel.h"

#include "libintx/config.h"
#include "libintx/utility.h"

namespace libintx::md {

  template<Operator Op, typename Params>
  void IntegralEngine<4>::compute(
    const Params &params,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    const Visitor &V)
  {


    using Kernel = kernel::Kernel<Op,Params>;
    using T = typename Kernel::simd_t;
    const int Lanes = Kernel::Lanes;

    const auto &ab = make_basis< pair<const Gaussian&> >(this->basis(0), this->basis(1), bra);
    int A = this->basis(0)[bra.at(0).first].L;
    int B = this->basis(1)[bra.at(0).second].L;
    int C = this->basis(2)[ket.at(0).first].L;
    int D = this->basis(3)[ket.at(0).second].L;

    auto batch = Kernel::batch(A,B,C,D);

    //printf("batch = {%i,%i}\n", batch.bra, batch.ket);

    using make_kernel = std::function<
      std::unique_ptr<Kernel>(int,int,int,int)
      >;
    static auto kernel_table = make_array<make_kernel,2*LMAX+1,2*LMAX+1>(
      [&](auto ab, auto cd) {
        return make_kernel(&kernel::make_kernel<ab,cd,Op,Params>);
      }
    );

    libintx_assert(batch.bra == 1);

    // double precision = 0.0;
    // if (norms.bra && norms.ket) {
    //   precision = this->precision_;
    // }

    struct BraBatch {
      HermiteBasis<2,T> basis;
      std::vector<T> allocator;
    };

    struct KetBatch {
      HermiteBasis<2,double> basis;
      std::vector<double> allocator;
    };

    std::vector<BraBatch> bra_batches((bra.size() + Lanes*batch.bra - 1)/(Lanes*batch.bra));
    std::vector<KetBatch> ket_batches((ket.size() + batch.ket - 1)/batch.ket);

#pragma omp parallel num_threads(this->num_threads)
    {

#pragma omp for schedule(static,1)
      for (size_t ij = 0; ij < bra_batches.size(); ++ij) {
        std::vector<Index2> indices(
          bra.begin() + ij*Lanes*batch.bra,
          bra.begin() + std::min<size_t>(bra.size(), (ij+1)*Lanes*batch.bra)
        );
        const double *bra_norms = (norms.bra ? norms.bra + ij*Lanes*batch.bra : nullptr);
        auto &bra_batch = bra_batches.at(ij);
        bra_batch.basis = make_basis<T>(
          this->basis(0), this->basis(1),
          indices, bra_norms,
          Phase<int>{+1},
          Lanes,
          bra_batch.allocator
        );
      }

#pragma omp for schedule(static,1)
      for (size_t kl = 0; kl < ket_batches.size(); ++kl) {
        std::vector<Index2> indices(
          ket.begin() + kl*batch.ket,
          ket.begin() + std::min<size_t>(ket.size(), (kl+1)*batch.ket)
        );
        const double *ket_norms = (norms.ket ? norms.ket + kl*batch.ket : nullptr);
        auto &ket_batch = ket_batches.at(kl);
        ket_batch.basis = make_basis(
          this->basis(2), this->basis(3),
          indices, ket_norms,
          Phase<int>{-1},
          1,
          ket_batch.allocator
        );
      }

    // std::vector<T> p_allocator;
    // auto p = make_basis<T>(this->basis(0), this->basis(1), bra, Phase<int>{1}, 1, p_allocator);

      auto kernel = kernel_table[A+B][C+D](A, B, C, D);

      // int K = nprim(a)*nprim(b);
      std::unique_ptr<T[]> V_batch(
        new (std::align_val_t{64}) T[batch.bra*batch.ket*npure(A,B)*npure(C,D)]
      );

#pragma omp for collapse(2) schedule(dynamic,1)
      for (size_t kl = 0; kl < ket_batches.size(); ++kl) {
        for (size_t ij = 0; ij < bra_batches.size(); ++ij) {
          auto &p = bra_batches[ij];
          auto &q = ket_batches[kl];
          //if (ij_batch_norms[ij_batch]*kl_batch_norm < precision) continue;
          std::fill_n(V_batch.get(), batch.bra*batch.ket*npure(A,B)*npure(C,D), 0);
          kernel->compute(params, p.basis, q.basis, V_batch.get());
          BraKet<size_t> idx = { Lanes*ij*batch.bra, kl*batch.ket };
          BraKet<size_t> dims = {
            std::min<size_t>(Lanes*batch.bra, bra.size() - idx.bra),
            static_cast<size_t>(q.basis.N)
          };
          int ldV = Lanes*batch.bra;
          V(dims, idx, reinterpret_cast<double*>(V_batch.get()), ldV);
        }
      } // kl_batch

    } // omp parallel

  }

  void IntegralEngine<4>::compute(
    Operator Op,
    const std::vector<Index2> &bra,
    const std::vector<Index2> &ket,
    BraKet<const double*> norms,
    const Visitor &V)
  {
    if (Op == Coulomb) {
      Coulomb::Operator::Parameters params;
      this->compute<Coulomb>(params, bra, ket, norms, V);
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
    auto v = [&](BraKet<size_t> batch, BraKet<size_t> idx, const double *U, size_t ldU) {
      size_t ncd = 0;
      for (int kl = 0; kl < batch.ket; ++kl) {
        auto [k,l] = ket[idx.ket+kl];
        ncd += nbf(basis(2)[k])*nbf(basis(3)[l]);
      }
      for (size_t icd = 0; icd < ncd; ++icd) {
        for (size_t iab = 0; iab < NAB; ++iab) {
          const auto *src =  U + (iab + icd*NAB)*ldU;
          auto *dst = V + idx.bra + iab*bra.size() + (icd + ket_start[idx.ket])*dims[0];
          std::copy_n(src, batch.bra, dst);
        }
      }
    };
    this->compute(Op, bra, ket, norms, v);
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
