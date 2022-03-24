#include "rysq/roots/fit.h"

namespace rysq {

  __constant__
  //static const
  struct R3_X1 {
    Hermite<3,-2,6> X, W;
    constexpr static const double Xmax = 0;
  } R3_X1;

  __constant__
  //static const
  struct R3_X2 {
    Hermite<3,0,8> X, W;
    constexpr static const double Xmax = 2;
  } R3_X2;

  __constant__
  //static const
  struct R3_X3 {
    Hermite<3,-8,8> X, W;
    constexpr static const double Xmax = 3;
  } R3_X3;

  __constant__
  //static const
  struct R3_X4 {
    Polynomial<3,10> X, W;
    constexpr static const double Xmax = 4;
  } R3_X4;

  __device__
  static inline void roots3(double X, double (&roots)[3], double (&weights)[3], int idx) {
    evaluate(X, roots, weights, idx, R3_X1, R3_X2, R3_X3);//, R3_X4);

  }

}
