#include "libintx/config.h"
#include "libintx/boys/gpu/chebyshev.h"

namespace libintx::gpu {

  using Boys = boys::gpu::Chebyshev<7,std::max(LMAX*4,LMAX*2+XMAX)+1,117,117*7>;

  const Boys& boys();

}
