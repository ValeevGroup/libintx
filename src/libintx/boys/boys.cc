#include "boys/boys.h"
#include "boys/reference.h"

std::unique_ptr<boys::Boys> boys::reference() {
  return std::make_unique<Reference>();
}
