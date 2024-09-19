#include "libintx/blas.h"
#include "libintx/config.h"
#include <string>

#if defined(LIBINTX_APPLE_ACCELERATE) && !defined(ACCELERATE_NEW_LAPACK)
#define ACCELERATE_NEW_LAPACK
#endif

#if defined(LIBINTX_CBLAS_H)
#define libintx_cblas_h <LIBINTX_CBLAS_H>
#include libintx_cblas_h
#else // defined(LIBINTX_CBLAS_H)

#if defined(LIBINTX_INTEL_MKL)
#define LIBINTX_INTEL_MKL_JIT 1
#pragma message("Using Intel MKL")
#if __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#include <mkl_version.h>
#else
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_version.h>
#endif // __has_include(<mkl_cblas.h>)

#elif defined(LIBINTX_APPLE_ACCELERATE) // defined(LIBINTX_INTEL_MKL)
#pragma message("Using Apple Accelerate")
#include <Accelerate/Accelerate.h>

#else // defined(LIBINTX_APPLE_ACCELERATE)
#pragma message("Using generic CBLAS")
#include <cblas.h>
#endif // LIBINTX_INTEL_MKL

#endif // defined(LIBINTX_CBLAS_H)

#if defined(LIBINTX_LAPACKE_H)
#define libintx_lapacke_h <LIBINTX_LAPACKE_H>
#include libintx_lapacke_h
#else // defined(LIBINTX_LAPACKE_H)

#if defined(LIBINTX_INTEL_MKL)
#pragma message("Using Intel MKL LAPACKE")
#if __has_include(<mkl_lapacke.h>)
#include <mkl_lapacke.h>
#else
#include <mkl/mkl_lapacke.h>
#endif // __has_include(<mkl_lapacke.h>)

#elif defined(LIBINTX_APPLE_ACCELERATE) // defined(LIBINTX_INTEL_MKL)
#pragma message("Using Apple Accelerate LAPACK")
#include <Accelerate/Accelerate.h>
#define LIBINTX_APPLE_LAPACK

#else // defined(LIBINTX_APPLE_ACCELERATE)
#pragma message("Using generic LAPACKE")
#include <lapacke.h>
#endif // LIBINTX_INTEL_MKL

#endif // defined(LIBINTX_LAPACKE_H)

#include "libintx/utility.h"
#include <memory>

std::string libintx::blas::version() {
#if defined(LIBINTX_INTEL_MKL)
  //mkl_get_version_string(version, 128);
  return (
    std::string("Intel MKL ") +
    std::to_string(__INTEL_MKL__) + "." +
    std::to_string(__INTEL_MKL_UPDATE__)
  );
#elif defined(LIBINTX_APPLE_ACCELERATE)
  return "Apple Accelerate";
#endif
  return "?";
}

void libintx::blas::gemm(
  Op OpA, Op OpB,
  size_t m, size_t n, size_t k,
  double alpha,
  const double *A, size_t ldA,
  const double *B, size_t ldB,
  double beta,
  double *C, size_t ldC)
{

  cblas_dgemm(
    CblasColMajor,
    (OpA == Transpose ? CblasTrans : CblasNoTrans),
    (OpB == Transpose ? CblasTrans : CblasNoTrans),
    m, n, k,
    alpha,
    A, ldA,
    B, ldB,
    beta,
    C, ldC
  );

}

void libintx::blas::sygvd(
  size_t N, char uplo,
  cmajor<double*> A,
  cmajor<double*> B,
  double* w,
  int type)
{

  std::unique_ptr<double[]> tmp;
  if (!w) {
    tmp = std::make_unique<double[]>(N);
    w = tmp.get();
  }

#ifndef LIBINTX_APPLE_LAPACK

  LAPACKE_dsygvd(
    LAPACK_COL_MAJOR, type,
    'V', uplo,
    N, A.data, A.ld, B.data, B.ld,
    w
  );

#else // Apple doesnt have lapacke

  //using lapack_int = MKL_INT;
  using lapack_int = __LAPACK_int;
  lapack_int itype = type;
  lapack_int n = N;
  lapack_int lda = A.ld;
  lapack_int ldb = B.ld;
  lapack_int lwork = -1;
  lapack_int liwork = -1;
  lapack_int info = 0;

  double work_query = 0;
  lapack_int iwork_query = 0;
  dsygvd_(
    &itype, "V", &uplo,
    &n,
    A.data, &lda,
    B.data, &ldb,
    w,
    &work_query, &lwork,
    &iwork_query, &liwork,
    &info
  );
  libintx_assert(info == 0);

  std::unique_ptr<double[]> work(new double[static_cast<size_t>(work_query)]);
  std::unique_ptr<lapack_int[]> iwork(new lapack_int[static_cast<size_t>(iwork_query)]);

  dsygvd_(
    &itype, "V", &uplo,
    &n,
    A.data, &lda,
    B.data, &ldb,
    w,
    work.get(), &lwork,
    iwork.get(), &liwork,
    &info
  );
  libintx_assert(info == 0);

#endif
}

namespace libintx::blas {

  template<typename T>
  struct GemmKernel<T>::Impl {
    void* kernel;
  };

  template<>
  GemmKernel<double>::GemmKernel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

  template<>
  GemmKernel<double>::~GemmKernel() {
  }

  template<>
  void GemmKernel<double>::compute(const double *A, const double *B, double *C) {
#if LIBINTX_INTEL_MKL_JIT
    auto impl = this->impl_->kernel;
    auto gemm = mkl_jit_get_dgemm_ptr(impl);
    gemm(impl, (double*)A, (double*)B, C);
#endif
  }

  template<>
  std::unique_ptr< GemmKernel<double> > make_gemm_kernel(
    Op OpA, Op OpB,
    size_t m, size_t n, size_t k,
    double alpha,
    const double *A, size_t ldA,
    const double *B, size_t ldB,
    double beta,
    double *C, size_t ldC)
  {
    std::unique_ptr< GemmKernel<double> > gemm;
#if LIBINTX_INTEL_MKL_JIT
    void* kernel = nullptr;
    mkl_jit_status_t status = mkl_jit_create_dgemm(
      &kernel,
      MKL_COL_MAJOR,
      MKL_NOTRANS,
      MKL_TRANS,
      m, n, k, alpha, ldA, ldB, beta, ldC
    );
    if (MKL_JIT_SUCCESS == status) {
      //printf("MKL_JIT_SUCCESS\n");
      auto impl = std::make_unique< GemmKernel<double>::Impl >();
      impl->kernel = kernel;
      gemm = std::make_unique< GemmKernel<double> >(std::move(impl));
    }
#endif
    return gemm;
  }

}
