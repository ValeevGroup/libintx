#include "libintx/ao/md/basis.h"
#include "libintx/simd.h"

#include <memory>

namespace libintx::md::kernel {

  template<Operator Op, typename Parameters>
  struct Kernel {
#ifdef LIBINTX_SIMD_DOUBLE
    using simd_t = LIBINTX_SIMD_DOUBLE;
    static constexpr int Lanes = simd_t::size();
#else
    using simd_t = double;
    static constexpr int Lanes = 1;
#endif
    virtual ~Kernel() = default;
    static constexpr int batch(int X, int C, int D) {
      int words = 4096-npure(C,D)*nherm2(C+D);
      int batch = std::min(128, words/nherm2(C+D));
      return std::max(1, batch/(Lanes*npure(X)));
    }
    virtual void compute(
      const Parameters&,
      const HermiteBasis<1,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      simd_t* __restrict__ V
    ) = 0;
  private:
    std::unique_ptr<simd_t> memory_;
  };

  template<int Bra, int Ket, Operator Op, typename Parameters>
  std::unique_ptr< Kernel<Op,Parameters> > make_kernel(int C, int D);

  template<int X, int C, int D, Operator Op, typename Parameters>
  std::unique_ptr< Kernel<Op,Parameters> > make_kernel() {
    return make_kernel<X,C+D,Op,Parameters>(C,D);
  }

}
