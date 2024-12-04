#ifndef LIBINTX_GPU_JENGINE_H
#define LIBINTX_GPU_JENGINE_H

#include "libintx/jengine.h"
#include <memory>

namespace libintx::gpu {

  std::unique_ptr<libintx::JEngine> make_jengine(
    const std::vector< std::tuple< Gaussian, array<double,3> > > &basis,
    const std::vector< std::tuple< Gaussian, array<double,3> > > &df_basis,
    std::function<void(double*)> V_linv,
    std::shared_ptr<const libintx::JEngine::Screening> screening = nullptr
  );

}

#endif /* LIBINTX_GPU_JENGINE_H */
