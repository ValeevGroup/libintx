#include "rysq/roots/opq.h"
#include <cmath>

namespace rysq {

  template<int N>
  struct stieltjes_grid_size {
    static const int value = 20+N*5;
  };

#define RYSQ_STIELTJES_GRID_SIZE(N,K)           \
  template<> struct stieltjes_grid_size<N> {    \
    static const int value = K;                 \
  };

  RYSQ_STIELTJES_GRID_SIZE(0,20);
  RYSQ_STIELTJES_GRID_SIZE(1,20);
  RYSQ_STIELTJES_GRID_SIZE(2,25);

  template<int K>
  struct Stieltjes {

    static const Stieltjes& instance() {
      static Stieltjes stieltjes;
      return stieltjes;
    }

    template<int N>
    //RYSQ_GPU_ENABLED
    bool compute(double x, double *X, double *W) const {

      static_assert(N > 0, "N > 0");

      struct { double X[K], W[K]; } grid;
      double a[N], b[N];

      for (int i = 0; i < K; ++i) {
        double S = this->X_[i];
        grid.X[i] = S*S;
        grid.W[i] = this->W_[i]*exp(-x*S*S);
      }

      int status = 0;
      status = opq::stieltjes<N,K>(grid.X, grid.W, a, b);
      if (status != 0) {
        throw std::runtime_error("Error in stieltjes method: " + std::to_string(status));
      }

      status = opq::coefficients<N>(a, b, X, W);
      if (status != 0) {
        throw std::runtime_error("Error in stieltjes method: " + std::to_string(status));
      }

      return true;

    }

  private:

    double X_[K];
    double W_[K];

    Stieltjes() {
      double a[K] = { 0.5 };
      double b[K] = { 1.0 };
      for (size_t j = 1; j < K; ++j) {
        a[j] = 0.5;
        b[j] = 0.25/(4.0 - (1.0/(j*j)));
      }
      opq::coefficients<K>(a, b, X_, W_);
    }

  };

}
