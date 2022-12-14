#ifndef BOYS_BOYS_H
#define BOYS_BOYS_H

#include <memory>

namespace boys {

  struct Boys {

    virtual ~Boys() {}

    virtual double compute(double t, int m) const = 0;

  protected:
    Boys() {}

  };

  std::unique_ptr<Boys> reference();
  std::unique_ptr<Boys> chebyshev();
  std::unique_ptr<double[]> chebyshev_interpolation_table(int Order, int M, int MaxT, int Segments);

}

#endif /* BOYS_BOYS_H */
