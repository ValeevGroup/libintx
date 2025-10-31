#include "libintx/blas.h"

namespace libintx::hf::solver {

  // returns {X,X^{-1},rank,S_condition_number,X_condition_number}, where
  // X is the generalized square-root-inverse such that X.transpose() * S * X = I
  // Copied from libints/tests/hartree-fock/hartree-fock++.cc
  //
  // produces "canonical" 1/sqrt: X = eigenvectors(S) . 1/sqrt(eigenvalues(S))
  // X rows are in original basis (AO), cols are transformed basis ("orthogonal" AO)
  //
  // S is conditioned to max_condition_number

  std::tuple<size_t,double,double> orthogonaliser(
    size_t N, MatrixRef<double> S,
    double max_condition_number = 1e8)
  {
    std::vector<double> s(N);
    blas::syev(N, 'U', S, s.data());
    double s_max = s.back();
    double s_min = s.front();
    double condition_number = std::min(
      s_max / std::max(s_min, std::numeric_limits<double>::min()),
      1.0 / std::numeric_limits<double>::epsilon()
    );
    double threshold = s_max / max_condition_number;
    size_t n_cond = 0;
    for (size_t j = 0; j < N; ++j) {
      if (s[j] < threshold) continue;
      s_min = s[j];
      n_cond = N-j;
      for (size_t jj = 0; jj < N-j; ++jj) {
        for (size_t i = 0; i < N; ++i) {
          S(i,jj) = 1.0/sqrt(s[j+jj])*S(i,j+jj);
        }
      }
      for (size_t jj = N-j; jj < N; ++jj) {
        for (size_t i = 0; i < N; ++i) {
          S(i,jj) = 0.0;
        }
      }
      break;
    }
    double X_condition_number = s_max/s_min;
    return std::tuple{ n_cond, condition_number, X_condition_number };
  }

  void solve(
    size_t N, size_t NX,
    MatrixRef<const double> F,
    MatrixRef<const double> X,
    MatrixRef<double> C,
    double* work)
  {
    double *V = work;
    // C = X'F
    blas::gemm(
      blas::Transpose, blas::NoTranspose,
      NX, N, N,
      1.0, X.data, X.ld, F.data, F.ld,
      0.0, C.data, NX
    );
    // V = CX
    blas::gemm(
      blas::NoTranspose, blas::NoTranspose,
      NX, NX, N,
      1.0, C.data, C.ld, X.data, X.ld,
      0.0, V, NX
    );
    std::vector<double> s(NX);
    blas::syev(NX, 'U', { V, NX }, s.data());
    // C = XV
    blas::gemm(
      blas::NoTranspose, blas::NoTranspose,
      N, N, NX,
      1.0, X.data, X.ld, V, NX,
      0.0, C.data, C.ld
    );
  }

}
