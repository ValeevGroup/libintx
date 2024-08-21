#ifndef LIBINTX_CUDA_MD_JENGINE_H
#define LIBINTX_CUDA_MD_JENGINE_H

#include "libintx/gpu/jengine/md/forward.h"

#include "libintx/array.h"
#include "libintx/shell.h"
#include "libintx/gpu/forward.h"
#include "libintx/gpu/jengine.h"
#include "libintx/gpu/api/api.h"

#include <map>

namespace libintx::gpu::jengine::md {

  struct JEngine : libintx::JEngine {

    using libintx::JEngine::Screening;

    JEngine(
      const std::vector< std::tuple< Gaussian, Double<3> > > &basis,
      const std::vector< std::tuple< Gaussian, Double<3> > > &df_basis,
      std::function<void(double*)> v_transform,
      std::shared_ptr<const Screening> screening = nullptr
    );

    void J(const TileIn &D, const TileOut &J, const AllSum&) override;

    // memory limits
    size_t maxij = 128*1024;
    size_t maxg = 16*1024*1024/8;

  private:

    std::function<void(double*)> v_transform_;
    float cutoff_ = 0;

    struct {
      using Pairs = std::vector< std::tuple<int,int,float> >;
      std::vector< std::tuple<int, Pairs> > pairs;
      std::vector<Shell> basis;
      // using iterator = decltype(pairs)::iterator;
      // using iterator = decltype(pairs)::reverse_iterator;
    } AB_;

    struct {
      struct Block {
        int L = -1;
        size_t kherm = size_t(-1);
        std::vector<Primitive2> primitives;
        std::vector<Index1> index;
        int kbf;
      };
      size_t nbf = 0;
      size_t nherm = 0;
      std::vector<Shell> basis;
      std::vector<Block> blocks;
      float max = 0;
    } Q_;

    using iterator = decltype(AB_.pairs)::iterator;

    template<typename>
    struct ab_pairs_iterator;

    template<typename It>
    void compute_x(const TileIn &D, double *X, ab_pairs_iterator<It>&, int device);

    template<typename It>
    void compute_j(const double *X, const TileOut &J, ab_pairs_iterator<It>&, int device);

  };

}

#endif /* LIBINTX_CUDA_MD_JENGINE_H */
