#ifndef LIBINTX_BLAS_H
#define LIBINTX_BLAS_H

#include "libintx/forward.h"
#include <cstddef>
#include <memory>
#include <string>

namespace libintx::blas {

  std::string version();

  enum Op {
    NoTranspose,
    Transpose
  };

  int get_num_threads();
  void set_num_threads(int num_threads);

  struct scoped_num_threads {
    explicit scoped_num_threads(int num_threads)
      : state_(get_num_threads())
    {
      set_num_threads(num_threads);
    }
    ~scoped_num_threads() {
      set_num_threads(this->state_);
    }
  private:
    const int state_;
    scoped_num_threads(scoped_num_threads&&) = delete;
    scoped_num_threads(const scoped_num_threads&) = delete;
  };

  template<typename T>
  struct GemmKernel {
    struct Impl;
    GemmKernel(std::unique_ptr<Impl> impl);
    ~GemmKernel();
    void compute(const double*, const double*, double*);
  private:
    std::unique_ptr<Impl> impl_;
  };

  template<typename T>
  std::unique_ptr< GemmKernel<T> > make_gemm_kernel(
    Op OpA, Op OpB,
    size_t m, size_t n, size_t k,
    T alpha,
    const T *A, size_t ldA,
    const T *B, size_t ldB,
    T beta,
    T *C, size_t ldC
  );

  void gemm(
    Op OpA, Op OpB,
    size_t m, size_t n, size_t k,
    double alpha,
    const double *A, size_t ldA,
    const double *B, size_t ldB,
    double beta,
    double *C, size_t ldC
  );

  void syev(
    size_t N,
    char uplo,
    MatrixRef<double> A,
    double *x
  );

  void sygvd(
    size_t N, char uplo,
    MatrixRef<double> A,
    MatrixRef<double> B,
    double* x = nullptr,
    int type = 1
  );

}

#endif /* LIBINTX_BLAS_H */
