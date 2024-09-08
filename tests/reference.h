#include "libintx/shell.h"
#include <cstddef>
#include <any>

namespace libintx::reference {

  double time(
    Operator op, const std::any &params,
    size_t n, const Gaussian& a, const Gaussian& b,
    double *data = nullptr
  );

  double time(
    Operator op, const std::any &params,
    size_t n, const Gaussian& a, const Gaussian& c, const Gaussian& d,
    double *data = nullptr
  );

  double time(
    Operator op, const std::any &params,
    size_t n, const Gaussian& a, const Gaussian& b, const Gaussian& c, const Gaussian& d,
    double *data = nullptr
  );

}
