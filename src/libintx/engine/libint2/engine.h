#ifndef LIBINTX_ENGINE_LIBINT2_H
#define LIBINTX_ENGINE_LIBINT2_H

#include "libintx/engine.h"

#include <memory>
#include <string>

namespace libintx {
namespace libint2 {

  std::unique_ptr< Kernel<3> > kernel(const Gaussian&, const Gaussian&, const Gaussian&);
  std::unique_ptr< Kernel<4> > kernel(const Gaussian&, const Gaussian&, const Gaussian&, const Gaussian&);

}
}

#endif /* LIBINTX_ENGINE_LIBINT2_H */
