#ifndef LIBINTX_ARRAY_H
#define LIBINTX_ARRAY_H

#include <utility>
#include <tuple>
#include <array>
#include <cstddef>
#include <assert.h>

#ifdef __CUDACC__
#define ARRAY_GPU_ENABLED __host__ __device__
#else
#define ARRAY_GPU_ENABLED
#endif

namespace libintx {

template<typename T, int N>
struct array {
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
  constexpr size_t size() const {
    return N;
  }
  ARRAY_GPU_ENABLED
  operator const auto&() const {
    return this->data;
  }
};

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
array(T &&t, Ts&& ... ts) ->
  array<typename std::decay<T>::type, 1+sizeof...(Ts)>;

template<typename T, int N>
bool operator==(const array<T,N> &lhs, const array<T,N> &rhs) {
  for (int i = 0; i < N; ++i) {
    if (lhs[i] != rhs[i]) return false;
  }
  return true;
}

template<typename T, int N, typename U>
ARRAY_GPU_ENABLED
void store(const array<T,N> &v, U *u, int stride = 1) {
  for (int i = 0; i < N; ++i) {
    *u = v[i];
    u += stride;
  }
}

template<typename T, int N>
ARRAY_GPU_ENABLED
array<T,N> operator-(const array<T,N> &v, const array<T,N> &u) {
  array<T,N> w;
  for (int i = 0; i < N; ++i) {
    w[i] = v[i]-u[i];
  }
  return w;
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

template<int N>
using Index = array<size_t,N>;

template<int N>
using Double = array<double,N>;



template<typename T>
constexpr auto make_array() {
  return array<T,0>{};
}

template<typename T, typename ... Ts>
constexpr auto make_array(const T &t, Ts&& ... ts) {
  return array<T, 1+sizeof...(Ts)>{ t, ts...};
}

template<typename T, size_t ... Idx>
struct array_initializer {
  template<class F>
  static constexpr T eval(F &&f) {
    return f(std::integral_constant<size_t,Idx>{}...);
  }
};

template<typename T, class F, size_t ... Idx, size_t ... Is>
constexpr auto make_array(F&& f, array_initializer<T,Idx...>, std::index_sequence<Is...>) {
  return make_array(
    array_initializer<T,Idx...,Is>::eval(f)...
  );
}

template<typename T, class F, size_t ... Is>
constexpr auto make_array(F &&f, const std::index_sequence<Is...>&) {
  return make_array<T>( f(std::integral_constant<size_t,Is>{})... );
}

template<typename T, class F, size_t ... Is, size_t ... Js>
constexpr auto make_array(F &&f, std::index_sequence<Is...>, std::index_sequence<Js...>) {
  return make_array(
    make_array<T>(
      f,
      array_initializer<T,Is>{},
      std::index_sequence<Js...>{}
    )...
  );
}

template<typename T, size_t ... Ns, class F>
constexpr auto make_array(F &&f) {
  return make_array<T>(f, std::make_index_sequence<Ns>{}...);
}

template<typename ... Ts>
constexpr auto array_cat(Ts ... ts) {
  return std::apply(
    [](auto ... ts) { return make_array(ts...); },
    ::std::tuple_cat(ts...)
  );
}

}

#endif /* LIBINTX_ARRAY_H */
