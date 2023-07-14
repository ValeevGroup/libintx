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

  auto& default_random_engine() {
    static std::default_random_engine g;
    return g;
  }

  template<typename T, typename ... Args>
  T random(Args ... args) {
    if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<double> r(args...);
      return r(default_random_engine());
    }
    else {
      std::uniform_int_distribution<T> r(args...);
      return r(default_random_engine());
    }
  }

  template<typename T, size_t N, typename ... Args>
  libintx::array<T,N> random(Args ... args) {
    libintx::array<T,N> r;
    for (auto &v : r) {
      v = random<T>(args...);
    }
    return r;
  }

  struct enabled {
    const bool status;
    enabled(int I, int J, int K) :
      status(I <= LMAX && J <= LMAX && K <= XMAX)
    {
    }
    enabled(int I, int J, int K, int L) :
      status(std::max({I,J,K,L}) <= LMAX)
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
      ps[k] = { double(5.12)/(L+k+1.37), double(L+K)*(k+2) };
      //ps[k] = { 0.5+k, K*(k+1.0) };
      //printf("g[%i,%i] = %f*e**%f\n", L, k, ps[k].C, ps[k].a);
    }
    return (
      Gaussian(L, ps, pure)
    );
  }

  template<typename F>
  void check4(
    F Check,
    int na, int nb, int nc, int nd,
    const double *ref, double tolerance = 1e-6)
  {
    for (int kl = 0; kl < nc*nd; ++kl) {
      for (int ij = 0; ij < na*nb; ++ij) {
        //printf("idx=%i,%i,%i\n", ij, kl, (kl) + (ij)*ld);
        test::ReferenceValue ab_cd_reference(
          ref ? ref[kl+ij*nc*nd] : 0,
          tolerance,
          ij%nb, ij/nb,
          kl%nd, kl/nd
        );
        Check(ij, kl, ab_cd_reference);
        //CHECK(*ab++ == ab_x_reference);
      }
    }
  }

}

#endif
