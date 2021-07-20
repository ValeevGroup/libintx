#ifndef LIBINTX_TEST_H
#define LIBINTX_TEST_H

#include "libintx/shell.h"

#include <chrono>
#include <random>
#include <ostream>
#include <ctime>

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif

namespace libintx::test {

  template<typename T>
  struct Random {
    Random() : g_(std::time(0)) {}
    template<typename ... Args>
    T operator()(Args ... args) const {
      if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> r(args...);
        return r(g_);
      }
      else {
        std::uniform_int_distribution<T> r(args...);
        return r(g_);
      }
    }
  private:
    mutable std::default_random_engine g_;
  };

  struct enabled {
    const bool status;
    enabled(int I, int J, int K) :
      status(I <= LMAX && J <= LMAX && K <= XMAX)
    {
    }
    operator bool() const { return status; }
    template<typename Id>
    bool operator()(Id) const { return status; }
  };

  struct ReferenceValue {

    ReferenceValue(double value, double tolerance) {
      value_ = value;
      tolerance_ = tolerance;
    }

    template<typename ... Idx>
    ReferenceValue(double value, double tolerance, Idx ... idx)
      : ReferenceValue(value, tolerance)
    {
      idx_ = " @ [" + ((std::to_string(idx) + " ") + ...) + "]";
    }

    operator double() const { return value_; }
    friend
    std::ostream& operator<<(std::ostream& os, const ReferenceValue& v) {
      os << double(v) << v.idx_;
      return os;
    }
    friend
    bool operator==(double lhs, const ReferenceValue &rhs) {
      if (lhs == double(rhs)) return true;
      return (std::abs(lhs-double(rhs)) <= rhs.tolerance_);
    }
  private:
    double value_, tolerance_;
    std::string idx_;
  };

  inline auto gaussian(int L, int K, bool pure = true) {
    std::vector<Gaussian::Primitive> ps(K);
    for (int k = 0; k < K; ++k) {
      ps[k] = { K/(k+1.0), K*(k+1.0) };
      //ps[k] = { 0.5+k, K*(k+1.0) };
    }
    return Gaussian(L, ps, pure);
  }

}

#endif
