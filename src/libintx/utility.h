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

  template<typename ... Ts>
  auto max(Ts&& ... ts) {
    return std::max({ ts... });
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

}

#define LIBINTX_LAMBDA(...) __VA_ARGS__

#endif /* LIBINTX_UTILITY_H */
