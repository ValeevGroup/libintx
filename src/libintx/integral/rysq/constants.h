#ifndef RYSQ_CONSTANTS_H
#define RYSQ_CONSTANTS_H

namespace rysq {

  static const double SQRT_4_POW_PI_5 = 34.986836655249725;

  template<int C>
  struct Constant {
    template<typename T>
    constexpr explicit operator T() const { return T(C); }
    constexpr operator double() const { return double(C); }
  };

  using Zero = Constant<0>;
  using One = Constant<1>;

}

#endif /* RYSQ_CONSTANTS_H */
