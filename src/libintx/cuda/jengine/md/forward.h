#ifndef LIBINTX_CUDA_MD_FORWARD_H
#define LIBINTX_CUDA_MD_FORWARD_H

#include "libintx/array.h"
#include "libintx/shell.h"

namespace libintx::cuda {
  class Stream;
}

namespace libintx::cuda::jengine::md {

  using Center = Double<3>;

  struct alignas(64) Shell {
    // LIBINTX_GPU_ENABLED Shell() {}
    // explicit Shell(const Gaussian& g);// : Gaussian(g) {}
    int16_t L, pure;
    Center r;
    decltype(Gaussian::prims) prims;
    int K;
    struct {
      size_t begin, end;
      operator auto() const {
        return std::pair{ begin, end };
      }
    } range;
    LIBINTX_GPU_ENABLED
    operator auto() const {
      return libintx::Shell{ L, pure };
    }
  };

  static_assert(sizeof(Shell) == 4*64);

  struct alignas(32) Index1 {
    int shell;
    int kbf, kherm;
  };
  static_assert(sizeof(Index1) == 1*32);

  struct alignas(32) Index2 {
    int first, second;
    array<int16_t,2> L;
    int kbf;
    int kprim;
    float norm;
  };
  static_assert(sizeof(Index2) == 1*32);

  struct alignas(32) Primitive2 {
    array<double,2> exp;
    array<Center,2> r;
    double C;
    float norm;
  };
  static_assert(sizeof(Primitive2) == 3*32);

  void hermitian_to_cartesian_1(
    int p, int n,
    const Index1 *index1,
    const Shell *basis,
    const double* Xp, double* X
  );

  void cartesian_to_hermitian_1(
    int p, int n,
    const Index1 *index1,
    const Shell *basis,
    const double*, double*
  );

  void hermitian_to_cartesian_2(
    int p, int n,
    const Index2 *index2,
    const Shell *basis,
    const double*, double*,
    Stream&
  );

  void cartesian_to_hermitian_2(
    int p, int n2,
    const Index2 *ijs,
    const Shell *basis,
    Primitive2 *P,
    const double*, double*,
    Stream&
  );

  template<int Bra, int Ket, int Step, class Boys>
  void df_jengine_kernel(
    const Boys &boys,
    int NP, const Primitive2 *P,
    int NQ, const Primitive2 *Q,
    const double* input,
    double* output,
    float cutoff,
    Stream&
  );

}

#endif /* LIBINTX_CUDA_MD_FORWARD_H */
