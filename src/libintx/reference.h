#ifndef LIBINTX_REFERENCE_H
#define LIBINTX_REFERENCE_H

#ifdef LIBINTX_LIBINT2
#include "libintx/engine/libint2/engine.h"
#define LIBINTX_REFERENCE_ERI libintx::libint2::kernel
#else
#include "libintx/engine/rysq/engine.h"
#warning Libint2 disabled, using Rys Quadrature as reference
#define LIBINTX_REFERENCE_ERI libintx::rysq::eri
#endif

namespace libintx::reference {

  template<class ... Args>
  auto eri(Args&& ... args) {
    return LIBINTX_REFERENCE_ERI(args...);
  }

}

#endif /* LIBINTX_REFERENCE_H */
