#ifndef LIBINTX_JENGINE_H
#define LIBINTX_JENGINE_H

#include "libintx/forward.h"
#include <functional>

namespace libintx {

  struct JEngine {

    using TileIndex = std::pair<size_t,size_t>;
    using TileIn = std::function<void(TileIndex, TileIndex, double*)>;
    using TileOut = std::function<void(TileIndex, TileIndex, const double*)>;

    struct Screening;

    virtual ~JEngine() = default;
    virtual void J(const TileIn &D, const TileOut &J) = 0;

  };

  struct JEngine::Screening {
    virtual ~Screening() {}
    //virtual float max1() const = 0;
    virtual float max1(int) const = 0;
    virtual float max2(int,int) const = 0;
    virtual float max() const = 0;
    virtual bool skip(float) const = 0;
  };

}

#endif /* LIBINTX_JENGINE_H */
