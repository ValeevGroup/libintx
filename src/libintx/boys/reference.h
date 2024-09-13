#ifndef LIBINTX_BOYS_REFERENCE_H
#define LIBINTX_BOYS_REFERENCE_H

#include "libintx/boys/boys.h"

#include <cmath>
#include <stdexcept>
#include <cassert>
#include <limits>
#include <string>


namespace libintx::boys {

  struct Reference : Boys {

    double compute(double T, int m) const override {
      if (T < m/4+1) { // low T, high m
        return maclaurin(T,m);
      }
      return recursive(T,m);
    }

    // not stable for high m
    static double recursive(long double T, int m) {
      if (T == 0) return 1.0/(2*m+1);
      long double Fn = std::sqrt(M_PI/(4*T))*std::erf(std::sqrt(T));
      for (int i = 0; i < m; ++i) {
        Fn = ((2*(i)+1)*Fn - std::exp(-T))/(2*T);
      }
      return Fn;
    }

    /// computes a single value of \f$ F_m(T) \f$ using MacLaurin series to full precision of @c double
    static double maclaurin(long double T, int m) {

      assert(m < 100);

      static const double T_crit = std::numeric_limits<double>::is_bounded == true ? -log( std::numeric_limits<double>::min() * 100.5 / 2. ) : double(0) ;
      if (std::numeric_limits<double>::is_bounded && T > T_crit) {
        throw std::domain_error(
          "boys::Reference::maclaurin: double lacks precision for argument T=" + std::to_string(T)
        );
      }

      static const long double half = double(1)/2;
      long double denom = (m + half);
      using std::exp;
      long double term = exp(-T) / (2 * denom);
      long double old_term = 0;
      long double sum = term;
      const long double epsilon = 1e-16; // get_epsilon(T);
      const long double epsilon_divided_10 = epsilon / 10;
      do {
        denom += 1;
        old_term = term;
        term = old_term * T / denom;
        sum += term;
        //rel_error = term / sum , hence iterate until rel_error = epsilon
        // however, must ensure that contributions are decreasing to ensure that omitted contributions are smaller than epsilon
      } while (term > sum * epsilon_divided_10 || old_term < term);

      return sum;
    }

  };

}

#endif /* LIBINTX_BOYS_REFERENCE_H */
