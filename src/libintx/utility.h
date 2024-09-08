#ifndef LIBINTX_UTILITY_H
#define LIBINTX_UTILITY_H

#include <string>
#include <string_view>
#include <chrono>

namespace libintx {

  template<typename T>
  auto str(const T &msg) {
    if constexpr (std::is_convertible_v<T, std::string> or
		  std::is_convertible_v<T, std::string_view>) {
      return msg;
    }
    else {
      return std::to_string(msg);
    }
  }

  template<typename ... As>
  auto str(const As& ... as) {
    return (str(as) + ...);
  }

  template<typename T, typename U>
  constexpr auto max(T&& t, U&& u) {
    return ::std::max(t,u);
  }

  template<typename T, typename ... Ts>
  constexpr auto max(T&& t, Ts&& ... ts) {
    return ::std::max<T>({ t, ts... });
  }

  template<typename It>
  struct iterator_range {
    iterator_range(It begin, It end)
      : begin_(begin), end_(end) {}
    It begin() const { return begin_; }
    It end() const { return end_; }
  private:
    It begin_, end_;
  };

  struct time {

    inline static auto now() {
      return std::chrono::high_resolution_clock::now();
    }

    template<typename T>
    static double since(T t) {
      auto d = (time::now() - t);
      return std::chrono::duration_cast< std::chrono::duration<double> >(d).count();
    }

  };


  template<size_t Bytes, typename T>
  constexpr size_t nwords() {
    static_assert(sizeof(T)%Bytes == 0);
    return sizeof(T)/Bytes;
  }

  template<typename T>
  constexpr auto nqwords = nwords<sizeof(double),T>();

  template<typename T, T ... Is>
  constexpr auto integer_sequence_tuple(std::integer_sequence<T,Is...>) {
    return std::tuple{ std::integral_constant<T,Is>{}... };
  }

  template<typename F, typename T, T ... Is>
  LIBINTX_ALWAYS_INLINE
  constexpr void foreach(const std::integer_sequence<T,Is...>&, F &&f) {
    ( f(std::integral_constant<T,Is>{}), ... );
}

  template<typename F, typename T, T ... First, T ... Second>
  LIBINTX_ALWAYS_INLINE
  constexpr void foreach2(
    std::integer_sequence<T,First...> first,
    std::integer_sequence<T,Second...> second,
    F &&f)
  {
    foreach(
      second,
      [&](auto &&J) {
        foreach(
          first,
          [&](auto &&I) { f(I,J); }
        );
      }
    );
  }

  template<typename T, T First, T ... Ts>
  constexpr void jump_table(std::integer_sequence<T,First,Ts...>, auto label, auto &&f) {
    if (First == label) {
      f(std::integral_constant<T,First>{});
      return;
    }
    if constexpr (sizeof...(Ts)) {
      jump_table(std::integer_sequence<T,Ts...>{}, label, f);
    }
  }

  template<typename T, T ... First, T ... Second>
  constexpr void jump_table(
    std::integer_sequence<T,First...>,
    std::integer_sequence<T,Second...>,
    auto first, auto second,
    auto &&f)
  {
    jump_table(
      std::make_integer_sequence<T,sizeof...(First)*sizeof...(Second)>{},
      first + second*sizeof...(First),
      [&](auto Label) {
        constexpr std::integral_constant<T,Label%sizeof...(First)> first;
        constexpr std::integral_constant<T,Label/sizeof...(First)> second;
        f(first,second);
      }
    );
  }

}

#define libintx_assert(EXPR)                            \
  if (!(EXPR)) { throw std::runtime_error(#EXPR); }

#define LIBINTX_LAMBDA(...) __VA_ARGS__

#endif /* LIBINTX_UTILITY_H */
