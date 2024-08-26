#include "libintx/forward.h"
#include "test.h"

namespace libintx::reference {

  void initialize();

  void compute(size_t n, const Gaussian& x, const Gaussian& c, const Gaussian& d, double*);
  void compute(size_t n, const Gaussian&, const Gaussian&, const Gaussian&, const Gaussian&, double*);

  inline double time(size_t n, const Gaussian& a, const Gaussian& c, const Gaussian& d, double *data = nullptr) {
    initialize();
    auto t = time::now();
    compute(n,a,c,d,data);
    return time::since(t);
  }

  inline double time(size_t n, const Gaussian& a, const Gaussian& b, const Gaussian& c, const Gaussian& d, double *data = nullptr) {
    initialize();
    auto t = time::now();
    compute(n,a,b,c,d,data);
    return time::since(t);
  }

}
