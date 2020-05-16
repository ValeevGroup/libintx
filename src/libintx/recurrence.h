#ifndef LIBINT_RECURRENCE_H
#define LIBINT_RECURRENCE_H

#include "libintx/array.h"

#include <stdint.h>

namespace libintx {
namespace recurrence {

struct Recurrence {
  uint8_t axis;
  uint8_t value;
  uint16_t index = 0;
  constexpr explicit Recurrence(uint8_t axis = 0, uint8_t value = 0, int index = 0)
    : axis(axis), value(value), index(index)
  {
  }
};

template<int A>
constexpr Recurrence recurrence(uint8_t i, uint8_t j, uint8_t k) {
  using cartesian::index;
  if (i) {
    uint8_t iA = (i-A);
    return int(i) < A ? Recurrence{0} : Recurrence{0, iA, index<0>(iA,j,k)};
  }
  if (j) {
    uint8_t jA = (j-A);
    return int(j) < A ? Recurrence{1} : Recurrence{1, jA, index<0>(i,jA,k)};
  }
  if (k) {
    uint8_t kA = (k-A);
    return int(k) < A ? Recurrence{2} : Recurrence{2, kA, index<0>(i,j,kA)};
  }
  return Recurrence{};
}

// recurrence tuple corresponding to last ([i,j,k] - A - B - C...)
template<int A, int B, int ... C>
constexpr Recurrence recurrence(uint8_t i, uint8_t j, uint8_t k) {
  if (i) {
    uint8_t iA = (i-A);
    return int(i) < A ? Recurrence{0} : recurrence<B,C...>(iA,j,k);
  }
  if (j) {
    uint8_t jA = (j-A);
    return int(j) < A ? Recurrence{1} : recurrence<B,C...>(i,jA,k);
  }
  if (k) {
    uint8_t kA = (k-A);
    return int(k) < A ? Recurrence{2} : recurrence<B,C...>(i,j,kA);
  }
  return Recurrence{};
}

constexpr inline uint32_t multiplicity(size_t a, size_t b) {
  return math::binomial<uint32_t>(a+b, b);
}

LIBINTX_GPU_ENABLED
constexpr inline int coefficient(size_t i, size_t j) {
  // if (j == 0) return 1;
  // return i*coefficient(i-1,j-1);
  return math::factorial(i)/math::factorial(i-j);
}

// Index of [(i+1,j,k), (i,j+1,k), (i,j,k+1)] relative to (i+j+k)
struct Index {
  uint16_t L;
  uint16_t index;
  uint32_t coefficient;
  uint16_t multiplicity;
  struct Generator;
};

struct Index::Generator {
  template<typename Idx1, typename Idx2>
  constexpr Index operator()(Idx1 t, Idx2 a) const {
    auto T = cartesian::orbital(t);
    auto A = cartesian::orbital(a);
    uint16_t L = (T+A).L();
    uint16_t index = cartesian::index<0>(T+A);
    uint32_t c = 1;
    uint16_t m = 1;
    for (size_t i = 0; i < 3; ++i) {
      m *= math::binomial<uint16_t>(A[i]+T[i], T[i]);
      c *= recurrence::coefficient(A[i]+T[i], T[i]);
    }
    return Index{ L, index, c, m };
  }
};

  //LIBINTX_GPU_CONSTANT
const auto recurrence_table = make_array<Index>(
  Index::Generator(),
  cartesian::index_sequence_x,
  cartesian::index_sequence_ab
);

}
}

#endif /* LIBINT_RECURRENCE_H */
