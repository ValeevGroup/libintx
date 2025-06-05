#ifndef LIBINTX_TEST_H
#define LIBINTX_TEST_H

#include "libintx/shell.h"
#include "libintx/utility.h"
#include "libintx/simd.h"

#include <chrono>
#include <random>
#include <ostream>
#include <iomanip>
#include <ctime>
#include <algorithm>

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif

#include <unsupported/Eigen/CXX11/Tensor>

namespace libintx::test {

  inline auto& default_random_engine() {
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
    explicit ReferenceValue(double value, double epsilon = 1e-12) {
      value_ = value;
      epsilon_ = epsilon;
    }
    template<typename ... Idx>
    auto at(Idx ... idx) const {
      ReferenceValue r = *this;
      r.idx_ = ((std::to_string(idx) + " ") + ...);
      return r;
    }
    operator double() const { return value_; }
    friend
    std::ostream& operator<<(std::ostream& os, const ReferenceValue& v) {
      os << std::setprecision(10) << std::fixed << double(v);
      //os << double(v);
      if (!v.idx_.empty()) os << " @ [ " << v.idx_ << "]";
      return os;
    }
    friend
    bool operator==(const ReferenceValue &rhs, double lhs) {
      // printf("%e,%e,%i\n", lhs, double(rhs), std::equal(double(rhs),lhs));
      // if (lhs == double(rhs)) return true;
      auto m = std::max({ std::abs(rhs), std::abs(lhs), 1.0 });
      return std::islessequal(std::abs(lhs-double(rhs)), m*rhs.epsilon_);
    }
    friend
    bool operator==(double lhs, const ReferenceValue &rhs) {
      return operator==(rhs,lhs);
    }
    ReferenceValue epsilon(double e) const {
      ReferenceValue r = *this;
      r.epsilon_ = e;
      return r;
    }
  private:
    double value_, epsilon_;
    std::string idx_;
  };

  template<typename T, size_t N>
  using Tensor = Eigen::Tensor<T,N>;

  template<typename T = double, typename ... Dims>
  auto zeros(Dims ... dims) {
    constexpr int N = sizeof...(Dims);
    // if constexpr (N == 1) {
    //   return Eigen::VectorXd::Zero().eval();
    // }
    // if constexpr (N == 2) {
    //   return Eigen::MatrixXd::Zero().eval();
    // }
    // else {
    Tensor<T,N> t(dims...);
    t.setZero();
    return t;
  }

  template<typename T = double>
  auto symmetric(size_t N, auto &&g) {
    auto S = zeros(N,N);
    for (size_t j = 0; j < N; ++j) {
      for (size_t i = 0; i <= j; ++i) {
        T v = g(i,j);
        S(i,j) = v;
        S(j,i) = v;
      }
    }
    return S;
  }

  inline auto gaussian(int L, int K, bool pure = true) {
    std::vector<Gaussian::Primitive> ps(K);
    double a = test::random<double>(0.1, 0.5);
    for (int k = 0; k < K; ++k) {
      ps[k] = { a, 1/a };
      a *= 5.0;
      //printf("g[%i,%i] = %f*e**%f\n", L, k, ps[k].C, ps[k].a);
    }
    auto r = test::random<double,3>(-0.25,0.25);
    return (
      Gaussian(L, r, ps, pure)
    );
  }

  template<int N>
  inline auto make_basis(std::array<int,N> L, std::array<int,N> K, size_t shells) {
    using Index = decltype(
      []{
        if constexpr (N == 1) return Index1{};
        if constexpr (N == 2) return Index2{};
      }()
    );
    Basis<Gaussian> basis;
    std::vector<Index> index;
    bool pure = true;
    for (int i = 0; i < (int)shells; ++i) {
      std::array<int,N> idx;
      for (int j = 0; j < N; ++j) {
        auto g = test::gaussian(L[j], K[j], pure);
        basis.push_back(g);
        idx[j] = j+i*N;
      }
      index.push_back(
        std::apply([](auto ... args) { return Index{ args... }; }, idx)
      );
    }
    return std::tuple{ basis, index };
  }

  template<typename F, typename T>
  void check2(F Check, T Ref, double epsilon = 1e-6) {
    auto &dims = Ref.dimensions();
    for (int j = 0; j < dims[1]; ++j) {
      for (int i = 0; i < dims[0]; ++i) {
        auto ref = test::ReferenceValue(Ref(i,j)).at(i,j);
        Check(ref,i,j);
      }
    }
  }

  template<typename F, typename T>
  void check3(F Check, T Ref) {
    auto &dims = Ref.dimensions();
    for (int k = 0; k < dims[2]; ++k) {
      for (int j = 0; j < dims[1]; ++j) {
        for (int i = 0; i < dims[0]; ++i) {
          auto ref = test::ReferenceValue(Ref(i,j,k)).at(i,j,k);
          Check(ref,i,j,k);
        }
      }
    }
  }

  template<typename F, typename T>
  void check4(F Check, T Ref) {
    auto &dims = Ref.dimensions();
    for (int l = 0; l < dims[3]; ++l) {
      for (int k = 0; k < dims[2]; ++k) {
        for (int j = 0; j < dims[1]; ++j) {
          for (int i = 0; i < dims[0]; ++i) {
            auto ref = test::ReferenceValue(Ref(i,j,k,l)).at(i,j,k,l);
            Check(ref,i,j,k,l);
          }
        }
      }
    }
  }

  inline std::string header() {
    std::string header;
#ifdef LIBINTX_SIMD_ISA
    header += "simd: " + str(LIBINTX_SIMD_ISA) + " ";
    header += str(sizeof(LIBINTX_SIMD_DOUBLE)*8) + "-bits";
#else
    header += "simd: OFF";
#endif
    header += "\n";
#ifdef __VERSION__
    header += "cxx: " + str(__VERSION__) + "\n";
#endif
    return header;
  }

  template<int N>
  auto parse_args(int argc, char **argv, int default_value) {
    std::array<int,N> args;
    for (int iarg = 0; iarg < N; ++iarg) {
      args[iarg] = default_value;
      if (argc > 1+iarg) args[iarg] = std::atoi(argv[1+iarg]);
    }
    return args;
  }

}

#endif
