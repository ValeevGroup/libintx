#include "libintx/ao/md/basis.h"
#include "libintx/simd.h"

#include <memory>

namespace libintx::md::kernel {

#ifdef LIBINTX_SIMD_DOUBLE
  using simd_t = LIBINTX_SIMD_DOUBLE;
#else
  using simd_t = double;
#endif

  struct Ket {
    using type = double;
    static constexpr auto batch(int c, int d) {
      return std::max(64/npure(c,d),1);
    }
  };

  template<Operator Op>
  struct Parameters {
    double precision = 0.0;
  };

  template<Operator Op, typename T, typename U, int ...>
  struct Kernel {

    virtual ~Kernel() = default;

    virtual void compute(
      const Parameters<Op>&,
      const HermiteBatch<T> &bra,
      const HermiteBatch<U> &ket,
      T* __restrict__ V
    ) = 0;

    virtual void compute_p_cd(
      const Parameters<Op>&,
      const HermiteBatch<T> &bra,
      const HermiteBatch<U> &ket,
      const std::function<void(const HermiteBatch<T>&,int,const T(&)[],int)> &V
    ) = 0;

  private:
    std::unique_ptr<T> memory_;
  };

  template<int Bra, int Ket, Operator Op, typename T, typename U>
  std::unique_ptr< Kernel<Op,T,U> > make_kernel(int,int,int,int);

  template<Operator Op, typename T, typename U = double, int L = LMAX>
  auto make_kernel(int A, int B, int C, int D) {
    using Factory = std::function<
      std::unique_ptr< Kernel<Op,T,U> >(int,int,int,int)
      >;
    static auto kernel_table = make_array<Factory,2*L+1,2*L+1>(
      [&](auto AB, auto CD) {
        return Factory(&kernel::make_kernel<AB,CD,Op,T,U>);
      }
    );
    return kernel_table[A+B][C+D](A,B,C,D);
  }

}
