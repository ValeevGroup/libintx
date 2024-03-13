#ifndef BOYS_CUDA_CHEBYSHEV_H
#define BOYS_CUDA_CHEBYSHEV_H

#include "libintx/boys/boys.h"
#include "libintx/boys/asymptotic.h"

//#include <cuda/api_wrappers.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/device.hpp>

#include <assert.h>

namespace boys {
namespace cuda {

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev {

    constexpr static double Delta = double(MaxT)/Segments;

    template<bool Flags>
    explicit Chebyshev(::cuda::device_t<Flags> device) {
      shared_table_ = ::cuda::memory::device::make_unique<double[]>(device, (Order+1)*M*Segments);
      table_ = shared_table_.get();
      auto table = boys::chebyshev_interpolation_table(Order, M, MaxT, Segments);
      size_t bytes = sizeof(double)*(Order+1)*M*Segments;
      ::cuda::memory::copy(shared_table_.get(), table.get(), bytes);
    }

    __device__
    double compute(double x, int m) const {

      assert(x >= 0);

      if (x >= MaxT) {
        return asymptotic(x,m);
      }

      // ---------------------------------------------
      // small and intermediate arguments => interpolate Fm and (optional) downward recursion
      // ---------------------------------------------
      // which interval does this x fall into?
      const double x_over_delta = x * (1.0/Delta);
      const int k = int(x_over_delta); // the interval index
      const double xd = x_over_delta - (double)k - 0.5; // this ranges from -0.5 to 0.5

      assert(k < Segments);

      // Table(i,m,k)
      int idx = m*(Order+1) + k*M*(Order+1);

      double Fm = 0;
      double xi = 1;
      for (int i = 0; i <= Order; ++i) {
        Fm += xi*this->table_[idx+i];
        xi *= xd;
      }

      return Fm;

    }

    template<int N>
    __device__
    void compute(double x, int m, double (&Fm)[N+1]) const {

      assert(x >= 0);

      // ---------------------------------------------
      // small and intermediate arguments => interpolate Fm and (optional) downward recursion
      // ---------------------------------------------
      // which interval does this x fall into?
      const double x_over_delta = x * (1.0/Delta);
      const int k = int(x_over_delta); // the interval index

      if (k >= Segments) {
        return asymptotic(x,m,Fm);
      }

      const double xd = x_over_delta - (double)k - 0.5; // this ranges from -0.5 to 0.5

      // Table(i,m,k)
      const double *p = this->table_ + m*(Order+1) + k*M*(Order+1);
      assert(m*Order + k*M*Order < (Order+1)*M*Segments);

      for (size_t j = 0; j <= N; ++j) {
        double fj = 0;
        double xi = 1;
        for (int i = 0; i <= Order; ++i) {
          fj += xi*p[i];
          xi *= xd;
        }
        Fm[j] = fj;
        p += (Order+1);
      }
      //return Fm;

    }

  private:
    const double* table_;
    std::shared_ptr<double[]> shared_table_;

  };

}
}

#endif /* BOYS_CUDA_CHEBYSHEV_H */
