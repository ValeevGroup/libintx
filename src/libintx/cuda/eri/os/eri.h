#ifndef LIBINTX_CUDA_ERI_OS_ERI_H
#define LIBINTX_CUDA_ERI_OS_ERI_H

#include "libintx/cuda/eri.h"
#include "libintx/cuda/eri/os/eri3.h"
#include "libintx/cuda/forward.h"
#include "libintx/cuda/api/thread_group.h"
#include "libintx/engine.h"

namespace libintx::cuda {

struct ObaraSaika;

template<>
struct ERI<3,ObaraSaika> {

  ERI(const Gaussian &A, const Gaussian &B, const Gaussian &X) {
    auto compute_kernel = [&](auto ab, auto x) {
      return ComputeKernel<ab.value, x.value>(A,B,X,cuda::boys());
    };
    kernel_ = make_ab_x_kernel<Kernel>(compute_kernel, A.L+B.L, X.L);
  }

  void compute(int K, const IntegralTuple<3> *list, const Double<3> *centers) const {
    kernel_(K,list,centers);
  }

  template<int _AB, int _X>
  struct ComputeKernel : os::ERI<3, _AB, _X, Boys, ComputeKernel<_AB,_X> > {
    using os::ERI<3,_AB,_X,Boys,ComputeKernel>::ERI;
    void operator()(int K, const IntegralTuple<3> *list, const Double<3> *centers) {
      this->launch(K, cudaStream_t(0), list, centers);
    }
    __device__
    void operator()(const IntegralTuple<3> *list, const Double<3> *centers) {
      __shared__ os::Vector3 rA, rB, rX;
      if (threadIdx.x+threadIdx.y == 0) {
        auto index = list[blockIdx.x].index;
        rA = centers[index[0]];
        rB = centers[index[1]];
        rX = centers[index[2]];
      }
      double *V2 = this->compute(rA,rB,rX);
      __syncthreads();
      int NA = nbf(this->A);
      int NB = nbf(this->B);
      int NX = npure(ComputeKernel::X::L);
      double scale = list[blockIdx.x].scale;
      double* output = list[blockIdx.x].data;
      const auto &thread_rank = this_thread_block().thread_rank();
      const auto &num_threads = this_thread_block().size();
      for (int i = thread_rank; i < NA*NB*NX; i += num_threads) {
        output[i] = scale*V2[i];
      }
    }
  };

protected:
  using Kernel = std::function<void(int,const IntegralTuple<3>*,const Double<3>*)>;
  Kernel kernel_;

};

}

#endif /* LIBINTX_CUDA_ERI_OS_ERI_H */
