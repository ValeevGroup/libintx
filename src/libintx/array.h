#ifndef LIBINTX_ARRAY_H
#define LIBINTX_ARRAY_H

#include <utility>
#include <tuple>
#include <array>
#include <cstddef>
#include <assert.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ARRAY_GPU_ENABLED __host__ __device__
#else
#define ARRAY_GPU_ENABLED
#endif

namespace libintx {

template<typename T, int N>
struct alignas(T) array {
  T data[N];
  ARRAY_GPU_ENABLED
  constexpr const auto& operator[](int i) const {
    assert(i < N);
    return data[i];
  }
  ARRAY_GPU_ENABLED
  constexpr auto& operator[](int i) {
    assert(i < N);
    return data[i];
  }
  ARRAY_GPU_ENABLED
  constexpr static size_t size() {
    return N;
  }
  ARRAY_GPU_ENABLED
  operator const auto&() const {
    return this->data;
  }
  ARRAY_GPU_ENABLED
  operator std::array<T,N>() const {
    return std::to_array(this->data);
  }
};

template<size_t Idx, typename T, int N>
ARRAY_GPU_ENABLED
T& get(array<T,N> &a) {
  static_assert(Idx < N);
  return a[Idx];
}

template<size_t Idx, typename T, int N>
ARRAY_GPU_ENABLED
const T& get(const array<T,N> &a) {
  static_assert(Idx < N);
  return a[Idx];
}

template<typename T, int N>
ARRAY_GPU_ENABLED
constexpr auto* begin(array<T,N> &a) {
  return a.data;
}

template<typename T, int N>
ARRAY_GPU_ENABLED
constexpr auto* begin(const array<T,N> &a) {
  return a.data;
}

template<typename T, int N>
ARRAY_GPU_ENABLED
constexpr auto* end(const array<T,N> &a) {
  return a.data+N;
}

template<typename T, typename ... Ts>
ARRAY_GPU_ENABLED
array(T &&t, Ts&& ... ts) ->
  array<typename std::decay<T>::type, 1+sizeof...(Ts)>;

template<typename T, int N>
ARRAY_GPU_ENABLED
bool operator==(const array<T,N> &lhs, const array<T,N> &rhs) {
  for (int i = 0; i < N; ++i) {
    if (lhs[i] != rhs[i]) return false;
  }
  return true;
}

template<typename T, size_t ... Is>
constexpr auto make_array_f(auto &&f, const std::index_sequence<Is...>&) {
  return array<T,sizeof...(Is)>{ f(std::integral_constant<size_t,Is>{})... };
}

template<size_t I, size_t ... Is>
constexpr auto make_array_f(auto &&f, const std::index_sequence<I,Is...>& is) {
  using T = decltype(f(std::integral_constant<size_t,I>{}));
  return make_array_f<T>(f, is);
}

template<typename T, int N>
ARRAY_GPU_ENABLED
array<T,N> operator-(const array<T,N> &v) {
  array<T,N> w;
  for (int i = 0; i < N; ++i) {
    w[i] = -v[i];
  }
  return w;
}

template<typename V, typename U, int N>
ARRAY_GPU_ENABLED
auto operator-(const array<V,N> &v, const U &u) {
  static_assert(std::tuple_size<U>::value == N);
  return make_array_f(
    [&](auto idx) {
      using std::get;
      return (v[idx] - get<idx>(u));
    },
    std::make_index_sequence<N>()
  );
}

template<typename T, int N>
ARRAY_GPU_ENABLED
T norm(const array<T,N> &v) {
  T norm = 0;
  for (int i = 0; i < N; ++i) {
    norm += v[i]*v[i];
  }
  return norm;
}

template<typename T, int N>
ARRAY_GPU_ENABLED
T norm(const array<T,N> &v, const array<T,N> &u) {
  T norm = 0;
  for (int i = 0; i < N; ++i) {
    T r = (v[i]-u[i]);
    norm += r*r;
  }
  return norm;
}

template<typename T, int N>
ARRAY_GPU_ENABLED
array<T,N> center_of_charge(T a, const array<T,N> &ra, T b, const array<T,N> &rb) {
  array<T,N> v;
  for (int i = 0; i < N; ++i) {
    v[i] = (a*ra[i] + b*rb[i])/(a+b);
  }
  return v;
}

template<typename T, typename U, int N>
array<T,N> array_cast(const array<U,N> &u) {
  return make_array<T>(
    [&](auto idx) { return u[idx]; },
    std::make_index_sequence<N>()
  );
}

template<typename T, class F, size_t ... Is, size_t ... Js>
constexpr auto make_array_f(F &&f, std::index_sequence<Is...>, std::index_sequence<Js...>) {
  return make_array_f(
    [&](auto i) {
      return make_array_f<T>(
        [&](auto j) { return f(i,j); },
        std::index_sequence<Js...>{}
      );
    },
    std::index_sequence<Is...>{}
  );
}

template<typename T, size_t ... Ns, class F>
constexpr auto make_array(F &&f) {
  return make_array_f<T>(f, std::make_index_sequence<Ns>{}...);
}

template<typename ... Ts>
constexpr auto array_cat(Ts ... ts) {
  return std::apply(
    [](auto ... ts) { return array{ts...}; },
    ::std::tuple_cat(ts...)
  );
}

} // libintx

namespace std {

    template<typename T, int N>
    struct tuple_size< libintx::array<T,N> > : std::integral_constant<size_t,N> { };

    template<size_t Idx, typename T, int N>
    struct tuple_element<Idx, libintx::array<T,N> > {
      using type = T;
    };

}

#endif /* LIBINTX_ARRAY_H */
