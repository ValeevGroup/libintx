#ifndef LIBINTX_FORWARD_H
#define LIBINTX_FORWARD_H

#define libintx_pragma(expr) _Pragma(#expr)

#define LIBINTX_NOINLINE __attribute__((noinline))

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LIBINTX_GPU_DEVICE __device__
#define LIBINTX_GPU_ENABLED __host__ __device__
#define LIBINTX_GPU_CONSTANT __constant__
#else
#define LIBINTX_GPU_DEVICE
#define LIBINTX_GPU_ENABLED
#define LIBINTX_GPU_CONSTANT
#endif

#if defined(__CUDACC__)
#define LIBINTX_ALWAYS_INLINE __forceinline__
#define libintx_unroll(expr) libintx_pragma(unroll expr)

#elif defined(__clang__)
#define LIBINTX_ALWAYS_INLINE [[gnu::always_inline]] inline
#define libintx_unroll(expr) libintx_pragma(unroll (expr))

#elif defined(__GNUC__)
#define LIBINTX_ALWAYS_INLINE [[gnu::always_inline]] inline
#define libintx_unroll(expr) libintx_pragma(GCC unroll expr)

#endif // __CUDACC__

namespace libintx {

  static constexpr struct None {} None;

  template<typename T, int N>
  struct array;

  struct Shell;

  template<typename>
  struct Basis;
  //struct Gaussian;

  template<typename First, typename Second = First>
  struct pair {
    First first;
    Second second;
    // constexpr operator std::pair<First,Second>() const {
    //   return { first, second };
    // }
  };

  using Index1 = int;
  using Index2 = pair<Index1,Index1>;

  template<typename ... Args>
  bool operator<(const pair<Args...> &a, const pair<Args...> &b) {
    if (a.first == b.first)
      return (a.second < b.second);
    return (a.first < b.first);
  }

  template<int Idx, typename First, typename Second>
  auto get(const pair<First,Second>& idx) {
    static_assert(Idx == 0 || Idx == 1);
    if constexpr (Idx == 0) return idx.first;
    if constexpr (Idx == 1) return idx.second;
  }

  template<typename T, unsigned long int Capacity>
  struct alignas(T) static_vector {
    LIBINTX_GPU_ENABLED
    auto size() const { return size_; }
    LIBINTX_GPU_ENABLED
    constexpr static auto capacity() { return Capacity; }
    LIBINTX_GPU_ENABLED
    auto begin() const { return data; }
    LIBINTX_GPU_ENABLED
    auto end() const { return data + size_; }
    LIBINTX_GPU_ENABLED
    auto& operator[](auto &&idx) {
      return data[idx];
    }
    LIBINTX_GPU_ENABLED
    const auto& operator[](auto &&idx) const {
      return data[idx];
    }
    //private:
    T data[Capacity];
    unsigned long int size_ = 0;
  };

  template<typename T>
  struct range {
    explicit range(T end) : range(T(), end) {}
    range(T begin, T end) : begin_(begin), end_(end) {}
    T begin() const { return begin_; }
    T end() const { return end_; }
    long int size() const { return (end_-begin_); }
    T operator[](T idx) const { return begin_+idx; }
  private:
    T begin_, end_;
  };

  template<typename Bra, typename Ket = Bra>
  struct BraKet {
    Bra bra;
    Ket ket;
    // constexpr operator std::pair<First,Second>() const {
    //   return { first, second };
    // }
  };

  template<typename T = int>
  struct Phase {
    const T value;
  };

  enum class Parity {
    Even,
    Odd
  };

  enum Order {
    RowMajor, ColumnMajor
  };

  template<typename T>
  struct cmajor {
    T data = T();
    unsigned long int ld = 0;
    auto operator()(auto i, auto j) const {
      return data[i + j*ld];
    }
    auto& operator()(auto i, auto j) {
      return data[i + j*ld];
    }
  };

  enum class Operator {
    Overlap,
    Kinetic,
    Nuclear,
    Coulomb
  };

#define LIBINTX_OPERATOR(OPERATOR)                      \
  static constexpr struct OPERATOR {                    \
    struct Operator {                                   \
      struct Parameters;                                \
    };                                                  \
    constexpr operator libintx::Operator() const {      \
      return libintx::Operator::OPERATOR;               \
    }                                                   \
  } OPERATOR

  LIBINTX_OPERATOR(Overlap);
  LIBINTX_OPERATOR(Kinetic);
  LIBINTX_OPERATOR(Nuclear);
  LIBINTX_OPERATOR(Coulomb);

  struct JEngine;

}

namespace libintx::ao {

  template<int ... Args>
  struct IntegralEngine;

  // base class
  template<>
  struct IntegralEngine<> {
    virtual ~IntegralEngine() = default;
  };

}

#endif /* LIBINTX_FORWARD_H */
