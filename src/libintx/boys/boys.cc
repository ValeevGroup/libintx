#include "libintx/boys/boys.h"
#include "libintx/boys/reference.h"
#include "libintx/boys/chebyshev.h"

std::unique_ptr<libintx::boys::Boys> libintx::boys::reference() {
  return std::make_unique<Reference>();
}

std::unique_ptr<libintx::boys::Boys> libintx::boys::chebyshev() {
  return std::make_unique< Chebyshev<7,16,117,7*117> >();
}

std::unique_ptr<double[]> libintx::boys::chebyshev_interpolation_table(int Order, int M, int MaxT, int Segments) {
  return libintx::boys::chebyshev_interpolation_table<double>(Order, M, MaxT, Segments);
}
