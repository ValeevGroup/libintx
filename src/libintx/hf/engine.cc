#include "libintx/hf/engine.h"
#include "libintx/hf/solver.h"
#include "libintx/ao/md/engine.h"
#include "libintx/blas.h"
#include "libintx/utility.h"

#include <map>

namespace libintx::hf {

  using md::IntegralEngine;
  using LKIndex = std::pair< std::pair<int,int>, std::vector<Index1> >;

  void apply(auto f, range<int> r1, range<int> r2) {
    //printf("%i:%i,%i:%i\n", r1.begin(), r1.end(), r2.begin(), r2.end());
    for (auto i2 = r2.begin(); i2 != r2.end(); ++i2) {
      for (auto i1 = r1.begin(); i1 != r1.end(); ++i1) {
        f(i1,i2);
      }
    }
  }

  void add1(
    const auto &ranges, Index2 idx,
    auto scale, const auto* __restrict__ S,
    auto* __restrict__ T, int ldT)
  {
    auto [i,j] = idx;
    const auto &ri = ranges[i];
    const auto &rj = ranges[j];
    auto f = [=,Sij=S](auto i, auto j) mutable {
      int ij = i+j*ldT;
      T[ij] += scale*(*Sij);
      ++Sij;
    };
    apply(f, ri, rj);
  }

  void add2(const auto &ranges, const std::vector<Index2> idx, const double *S, double *T, int ldT) {
    for (auto [i,j] : idx) {
      const auto &ri = ranges[i];
      const auto &rj = ranges[j];
      auto f = [=,Sij=S,ldS=idx.size()](auto i, auto j) mutable {
        int ij = i+j*ldT;
        if (j <= i) T[ij] += *Sij;
        int ji = j+i*ldT;
        if (j < i) T[ji] += *Sij;
        Sij += ldS;
      };
      apply(f, ri, rj);
      ++S;
    }
  }

  auto make_basis_index(auto begin, auto end, int ish = 0) {
    std::map<LKIndex::first_type, LKIndex::second_type> index_map;
    for (auto it = begin; it != end; ++it) {
      index_map[{it->L, it->K}].push_back(ish++);
    }
    // std::vector<LKIndex> basis_index;
    // for (auto &[k,v] : index_map) {
    //   basis_index.push_back({k,v});
    // }
    // return basis_index;
    return std::vector<LKIndex>(index_map.begin(), index_map.end());
  }

  auto make_1e_basis_index(const Basis<Gaussian> &basis, int maxbf = 64) {
    std::vector< std::vector<LKIndex> > basis_index;
    for (auto it = basis.begin(); it != basis.end(); (void)it) {
      auto first = it;
      int n = 0;
      while (it != basis.end()) {
        if (n + nbf(*it) > maxbf) break;
        n += nbf(*it++);
      }
      int ish = first - basis.begin();
      auto lk_index = make_basis_index(first, it, ish);
      basis_index.push_back(lk_index);
    }
    return basis_index;
  }

  void compute1e(
    IntegralEngine<2> &ao,
    Operator Op,
    const Basis<Gaussian> &basis,
    MatrixRef<double> T,
    libintx::num_threads num_threads)
  {

    static constexpr int nbra = 64;
    static constexpr int nket = (32*1024/sizeof(double))/nbra;
    static constexpr int maxbf = nbra*nket;

    auto bra = make_1e_basis_index(basis, nbra);
    auto ket = make_basis_index(basis.begin(), basis.end());

    auto compute = [&](int nab, const auto &batch) {
      if (batch.empty()) return;
      double buffer[nbra*nket];
      ao.compute(Op, batch, buffer);
      add2(basis.ranges(), batch, buffer, T.data, T.ld);
    };

#pragma omp parallel num_threads(num_threads.value)
    {
      std::vector<Index2> batch;
#pragma omp for schedule(dynamic,1)
      for (size_t k = 0; k < bra.size()*ket.size(); ++k) {
        auto &[B,js] = ket.at(k/bra.size());
        for (auto &[A,is] : bra.at(k%bra.size())) {
          int nab = npure(A.first,B.first);
          for (auto j : js) {
            for (auto i : is) {
              if (j > i) continue;
              //printf("%i,%i\n", i, j);
              batch.push_back({i,j});
              if ((batch.size()+1)*nab <= maxbf) continue;
              compute(nab,batch);
              batch.clear();
            }
          }
          compute(nab,batch); // remainder
          batch.clear();
        }
      }
    }

  }


  FockEngine::FockEngine(
    const Wfn<Gaussian> &wfn,
    ao::screening::Norm<double> norm,
    double precision
  )
    : wfn_(wfn), norm_(norm), precision(precision)
  {
    libintx_assert(this->wfn_.basis()->max().L <= LMAX);
  }

  FockEngine::~FockEngine() {}

  std::tuple<size_t,size_t> FockEngine::init2(double smin) {
    ao_pairs_ = ao::screening::pairs<float>(*wfn_.basis(), norm_, smin, this->num_threads);
    size_t N = wfn_.basis()->size();
    return std::tuple<size_t,size_t>{ ao_pairs_.size(), (N*N+N)/2 };
  }

  void FockEngine::S(MatrixRef<double> S) {
    //printf("FockEngine::S(%p,%lu)\n", S, ldS);
    const auto &basis = *wfn_.basis();
    md::IntegralEngine<2> ao(basis, basis);
    compute1e(ao, Operator::Overlap, basis, S, this->num_threads);
  }

  void FockEngine::T(MatrixRef<double> T) {
    //printf("FockEngine::T(%p,%lu)\n", T, ldT);
    const auto &basis = *wfn_.basis();
    md::IntegralEngine<2> ao(basis, basis);
    compute1e(ao, Operator::Kinetic, basis, T, this->num_threads);
  }

  void FockEngine::V(MatrixRef<double> V) {
    //printf("FockEngine::V(%p,%lu)\n", V, ldV);
    const auto &basis = *wfn_.basis();
    md::IntegralEngine<2> ao(basis, basis);
    ao.set(Nuclear::Operator::Parameters{ this->wfn_.atoms() });
    compute1e(ao, Operator::Nuclear, basis, V, this->num_threads);
  }

  std::tuple<size_t,double,double> FockEngine::X(MatrixRef<double> X) {
    size_t N = this->wfn_.basis()->nbf();
    this->S(X);
    blas::scoped_num_threads scoped_num_threads(this->num_threads);
    return solver::orthogonaliser(N, X);
  }

  void FockEngine::D(
    MatrixRef<const double> F,
    MatrixRef<const double> X, size_t NX,
    MatrixRef<double> D,
    MatrixRef<double> C)
  {
    //printf("FockEngine::D\n");
    size_t N = this->wfn_.basis()->nbf();
    size_t nocc2 = this->wfn_.nocc2();
    std::unique_ptr<double[]> tmp;
    if (!C.data) {
      tmp = std::make_unique<double[]>(N*N);
      C = { tmp.get(), N };
    }
    blas::scoped_num_threads scoped_num_threads(this->num_threads);
    solver::solve(N, NX, F, X, C, D.data);
    blas::gemm(
      blas::NoTranspose, blas::Transpose,
      N, N, nocc2,
      1.0, C.data, C.ld, C.data, C.ld,
      0.0, D.data, D.ld
    );
  }

}

std::unique_ptr<libintx::hf::FockEngine> libintx::hf::fock_engine(
  const Wfn<Gaussian>& wfn,
  double precision)
{
  return std::make_unique<libintx::hf::FockEngine>(wfn, ao::screening::norm<2>, precision);
}
