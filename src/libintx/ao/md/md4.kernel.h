#include "libintx/ao/md/basis.h"
#include "libintx/simd.h"

#include <memory>

namespace libintx::md::kernel {

  template<Operator Op, typename Parameters, int ...>
  struct Kernel {
#ifdef LIBINTX_SIMD_DOUBLE
    using simd_t = LIBINTX_SIMD_DOUBLE;
    static constexpr int Lanes = simd_t::size();
#else
    using simd_t = double;
    static constexpr int Lanes = 1;
#endif
    static constexpr auto batch(int a, int b, int c, int d) {
      int ket = std::max(64/npure(c,d),1);
      return BraKet<int>{ 1, ket };
    }
    virtual ~Kernel() = default;
    virtual void compute(
      const Parameters&,
      const HermiteBasis<2,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      simd_t* __restrict__ V
    ) = 0;

    virtual void compute_p_cd(
      const Parameters&,
      const HermiteBasis<2,simd_t> &bra,
      const HermiteBasis<2,double> &ket,
      const std::function<void(int,int,simd_t(&)[],int)> &V,
      double precision
    ) = 0;

  private:
    std::unique_ptr<simd_t> memory_;
  };

  template<int Bra, int Ket, Operator Op, typename Parameters>
  std::unique_ptr< Kernel<Op,Parameters> > make_kernel(int,int,int,int);

}
