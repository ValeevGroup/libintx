#ifndef LIBINTX_TUPLE_H
#define LIBINTX_TUPLE_H

//#include <type_traits>

#ifdef __CUDACC__
#define TUPLE_GPU_ENABLED __host__ __device__
#else
#define TUPLE_GPU_ENABLED
#endif

template<typename ... Ts>
struct tuple;

template<>
struct tuple<> {};

template<typename T>
struct tuple<T> {
  T head;
};

template<typename T, typename ... Ts>
struct tuple<T,Ts...> {
  T head;
  tuple<Ts...> tail;
};

template<size_t Idx>
struct tuple_index {};

template<typename T0, typename ... Ts>
TUPLE_GPU_ENABLED
auto& get(const tuple<T0,Ts...> &t, tuple_index<0>) {
  return t.head;
}

template<typename T0, typename ... Ts, size_t Idx>
TUPLE_GPU_ENABLED
auto& get(const tuple<T0,Ts...> &t, tuple_index<Idx>) {
  return get<Idx-1>(t.tail);
}

template<size_t Idx, typename ... Ts>
TUPLE_GPU_ENABLED
auto& get(const tuple<Ts...> &t) {
  return get(t, tuple_index<Idx>{});
}

#endif /* LIBINTX_TUPLE_H */
