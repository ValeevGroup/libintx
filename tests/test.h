#include "libintx/shell.h"

#include <chrono>
#include <ostream>

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif

namespace libintx {
namespace test {

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
}
