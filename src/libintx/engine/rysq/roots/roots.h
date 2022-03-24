#include "rysq/roots/stieltjes.h"
#include <cstdlib>

namespace rysq {

  template<int N>
  struct Roots {
    Roots() : stieltjes_(Stieltjes<K>::instance())
    {
    }
    //RYSQ_GPU_ENABLED
    bool compute(double x, double *X, double *W) const {
      return stieltjes_.template compute<N ? N : 1>(x, X, W);
    }
  private:
    static const int K = stieltjes_grid_size<N>::value;
    const Stieltjes<K> stieltjes_;
  };

}
