#ifndef LIBINTX_OS_KERNEL_H
#define LIBINTX_OS_KERNEL_H

#include "libintx/engine.h"

#include <memory>
#include <string>

namespace libintx {
namespace os {

  std::unique_ptr< Kernel<3> > eri(const Gaussian&, const Gaussian&, const Gaussian&);

}
}

#endif /* LIBINTX_OS_KERNEL_H */
