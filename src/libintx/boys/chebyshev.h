#ifndef BOYS_CHEBYSHEV_H
#define BOYS_CHEBYSHEV_H

#include "libintx/boys/boys.h"
#include "libintx/boys/asymptotic.h"
#include "libintx/boys/reference.h"
#include "libintx/math/interpolate/chebyshev.h"

namespace boys {

  template<typename T, typename R = long double>
  std::unique_ptr<T[]> chebyshev_interpolation_table(int Order, int M, int MaxT, int Segments) {
    R Delta = R(MaxT)/R(Segments);
    ChebyshevInterpolation<R> interpolate(Order);
    boys::Reference reference;
    std::unique_ptr<T[]> table(new T[(Order+1)*M*Segments]);
    for (int i = 0; i < Segments; ++i) {
      R a = i*Delta;
      R b = a + Delta;
      for (int m = 0; m < M; ++m) {
        auto f = [m,&reference](R x) {
          return (R)reference.compute(x,m);
        };
        auto p = interpolate.generate(f, a, b);
        size_t idx = m*(Order+1) + i*M*(Order+1);
        for (int k = 0; k <= Order; ++k) {
          table[k+idx] = p[k];
        }
      }
    }
    return table;
  }

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev : Boys {

    constexpr static double Delta = double(MaxT)/Segments;

    Chebyshev() {
      table_ = chebyshev_interpolation_table<double>(Order, M, MaxT, Segments);
    }

    template<int N>
     __attribute__((noinline))
    void compute(double x, int m, double (&Fm)[N]) const {

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
      const double *p = this->table_.get() + m*(Order+1) + k*M*(Order+1);
      assert(m*Order + k*M*Order < (Order+1)*M*Segments);


      double xi[Order+1];
      xi[0] = 1;
      for (int i = 1; i <= Order; ++i) {
        xi[i] = xd*xi[i-1];
      }

      size_t j = 0;

#ifdef __SSE4_1__
      for (; j < N-1; j += 2) {
        auto *p0 = p + 0*(Order+1);
        auto *p1 = p + 1*(Order+1);
        p += 2*(Order+1);
        auto f0 = _mm_setzero_pd();
        auto f1 = _mm_setzero_pd();
        for (int i = 0; i <= Order-2; i += 2) {
          auto x = _mm_loadu_pd(xi+i);
          f0 += x*_mm_loadu_pd(p0+i);
          f1 += x*_mm_loadu_pd(p1+i);
        }
        auto f = _mm_hadd_pd(f0,f1);
        _mm_storeu_pd(Fm+j, f);
        assert(Fm[j+0] >= 0);
        assert(Fm[j+1] >= 0);
      }
#endif

      for (; j < N; ++j) {
        double fj = p[0];
        for (int i = 1; i <= Order; i += 1) {
          fj += xi[i]*p[i];
        }
        Fm[j] = fj;
        assert(Fm[j] >= 0);
        p += (Order+1);
      }

    }

    double compute(double x, int m) const override {
      double Fm[1];
      compute(x, m, Fm);
      return Fm[0];
    }

  private:
    std::shared_ptr<double[]> table_;

  };

}

#endif /* BOYS_CHEBYSHEV_H */
