#include "boys/boys.h"
#include "boys/reference.h"
#include "boys/chebyshev.h"

std::unique_ptr<boys::Boys> boys::reference() {
  return std::make_unique<Reference>();
}

std::unique_ptr<boys::Boys> boys::chebyshev() {
  return std::make_unique< Chebyshev<7,16,117,7*117> >();
}

std::unique_ptr<double[]> boys::chebyshev_interpolation_table(int Order, int M, int MaxT, int Segments) {
  return boys::chebyshev_interpolation_table<double>(Order, M, MaxT, Segments);
}
