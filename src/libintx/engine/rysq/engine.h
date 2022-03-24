#ifndef LIBINTX_RYSQ_ENGINE_H
#define LIBINTX_RYSQ_ENGINE_H

#include "libintx/engine.h"

#include <memory>

namespace libintx {
namespace rysq {

  std::unique_ptr< Kernel<3> > eri(const Gaussian&, const Gaussian&, const Gaussian&);
  std::unique_ptr< Kernel<4> > eri(const Gaussian&, const Gaussian&, const Gaussian&, const Gaussian&);

}
}

#endif /* LIBINTX_RYSQ_ENGINE_H */
