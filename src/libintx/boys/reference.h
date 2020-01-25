#ifndef BOYS_REFERENCE_H
#define BOYS_REFERENCE_H

#include "boys/boys.h"

#include <cmath>
#include <stdexcept>
#include <cassert>
#include <limits>

namespace boys {

  struct Reference : Boys {

    /// computes a single value of \f$ F_m(T) \f$ using MacLaurin series to full precision of @c double
    double compute(double T, int m) const override {
      assert(m < 100);
      static const double T_crit = std::numeric_limits<double>::is_bounded == true ? -log( std::numeric_limits<double>::min() * 100.5 / 2. ) : double(0) ;
      if (std::numeric_limits<double>::is_bounded && T > T_crit) {
        throw std::overflow_error("FmEval_Reference<double>::eval: double lacks precision for the given value of argument T");
      }
      static const double half = double(1)/2;
      double denom = (m + half);
      using std::exp;
      double term = exp(-T) / (2 * denom);
      double old_term = 0;
      double sum = term;
      const double epsilon = 1e-16; // get_epsilon(T);
      const double epsilon_divided_10 = epsilon / 10;
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

#endif /* BOYS_REFERENCE_H */
