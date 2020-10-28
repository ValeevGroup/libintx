#ifndef LIBINTX_UTILITY_H
#define LIBINTX_UTILITY_H

#include <string>
#include <chrono>

namespace libintx {

  template<typename ... As>
  auto str(std::string msg, As&& ... as) {
    return msg + (std::to_string(as) + ...);
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
