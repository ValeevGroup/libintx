#ifndef LIBINTX_TEST_H
#define LIBINTX_TEST_H

#include "libintx/shell.h"
#include "libintx/utility.h"
#include "libintx/reference.h"

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
    enabled(int X, int C, int D) :
      status(X <= XMAX && C <= LMAX && D <= LMAX)
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
    bool operator==(const ReferenceValue &rhs, double lhs) {
      if (lhs == double(rhs)) return true;
      return (std::abs(lhs-double(rhs)) <= rhs.tolerance_);
    }
    friend
    bool operator==(double lhs, const ReferenceValue &rhs) {
      return operator==(rhs,lhs);
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

  auto basis1(std::tuple<int> L, std::tuple<int> K, size_t N) {
    Basis<Gaussian> basis;
    std::vector<Index1> idx;
    bool pure = true;
    for (int i = 0; i < (int)N; ++i) {
      auto a = test::gaussian(std::get<0>(L), std::get<0>(K), pure);
      auto r0 = test::random<double,3>(-0.25,0.25);
      basis.push_back({a,r0});
      idx.push_back(Index1{i});
    }
    return std::tuple{basis,idx};
  }

  auto basis2(std::pair<int,int> L, std::pair<int,int> K, size_t N) {
    Basis<Gaussian> basis;
    std::vector<Index2> idx;
    bool pure = true;
    for (int i = 0; i < (int)N; ++i) {
      auto a = test::gaussian(L.first, K.first, pure);
      auto b = test::gaussian(L.second, K.second, pure);
      auto r0 = test::random<double,3>(-0.25,0.25);
      auto r1 = test::random<double,3>(-0.25,0.25);
      basis.push_back({a,r0});
      basis.push_back({b,r1});
      idx.push_back(Index2{i*2,i*2+1});
    }
    return std::tuple{basis,idx};
  }

  template<typename F, typename T>
  void check4(F Check, T Ref, double tolerance = 1e-6) {
    auto &dims = Ref.dimensions();
    for (int l = 0; l < dims[3]; ++l) {
      for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
          for (int i = 0; i < dims[0]; ++i) {
            test::ReferenceValue ref(Ref(i,j,k,l), tolerance, i,j,k,l);
            Check(ref,i,j,k,l);
          }
        }
      }
    }
  }

}

namespace libintx::reference {

  template<typename ... Args>
  double time(int N, Args ... args) {
    auto eri = libintx::reference::eri(std::get<0>(args)...);
    auto t = time::now();
    eri->repeat(N, std::get<1>(args)...);
    return time::since(t);
  }

}

#endif
