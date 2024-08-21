#ifndef LIBINTX_GPU_JENGINE_OS_H
#define LIBINTX_GPU_JENGINE_OS_H

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/cuda/eri.h"
#include "libintx/cuda/jengine.h"
#include "libintx/cuda/api/api.h"

#include <vector>
#include <utility>
#include <memory>
#include <map>
#include <functional>

namespace libintx::gpu::jengine::os {

  struct JEngine : libintx::JEngine {

    using TileIndex = std::pair<size_t,size_t>;
    using TileIn = std::function<void(TileIndex, TileIndex, double*)>;
    using TileOut = std::function<void(TileIndex, TileIndex, const double*)>;

    using libintx::JEngine::Screening;

    JEngine(
      const std::vector< std::tuple< Gaussian, Double<3> > > &basis,
      const std::vector< std::tuple< Gaussian, Double<3> > > &df_basis,
      std::function<void(double*)> v_transform,
      std::shared_ptr<const libintx::JEngine::Screening> screening = nullptr
    );

    void J(const TileIn &D, const TileOut &J) override;

    void J1(const TileIn &D, double *X);
    void J2(const double *X, const TileOut &J);

  public:

    struct Basis {
      struct Block {
        struct Index {
          int shell;
          int center;
          int start;
        };
        std::shared_ptr<Gaussian> gaussian;
        std::vector<Index> list;
      };
      int nbf = 0;
      std::vector<Block> blocks;
      Basis() = default;
      explicit Basis(
        const std::vector< std::tuple<Gaussian, Double<3> > > &basis,
        std::vector< Double<3> > &centers
      );
    };

  protected:
    Basis basis_, df_basis_;
    std::function<void(double*)> v_transform_;
    std::shared_ptr<const Screening> screening_;
    cuda::device::vector< Double<3> > centers_;

  };

}


#endif /* LIBINTX_GPU_JENGINE_OS_H */
