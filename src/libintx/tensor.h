#ifndef LIBINTX_TENSOR_H
#define LIBINTX_TENSOR_H

#include "libintx/forward.h"
#include <cstddef>

namespace libintx {

  template<typename T, int Rank>
  struct TensorRef {
    using Dimensions = std::array<size_t,Rank>;
    explicit TensorRef(T* data, const Dimensions& dims)
      : data_(data), dims_(dims) {}
    // template<typename Int>
    // explicit TensorRef(T* data, const std::array<Int,Rank>& dims)
    //   : data_(data),
    //     dims_(
    //       std::apply(
    //         [](auto ... args) { return Dimensions{args...}; },
    //         dims
    //       )
    //     )
    // {
    // }
    template<typename ... Idx>
    LIBINTX_GPU_ENABLED
    auto& operator()(Idx ... idx) {
      static_assert(sizeof...(idx) == Rank);
      return data_[index(idx...)];
    }

    LIBINTX_GPU_ENABLED
    auto* data() { return data_; }

    LIBINTX_GPU_ENABLED
    const auto* data() const { return data_; }

    LIBINTX_GPU_ENABLED
    const auto& dimensions() const { return dims_; }

    template<typename ... Shape>
    auto reshape(Shape ... shape) const {
      return TensorRef<T,sizeof...(Shape)>(data_, { (size_t)shape... });
    }


  private:
    T* data_;
    Dimensions dims_;

  private:

    template<int Dim = 0>
    LIBINTX_GPU_ENABLED
    auto index(auto i, auto ... is) const {
      assert(i < std::get<Dim>(dims_) || (std::get<Dim>(dims_) == 0));
      return (i + std::get<Dim>(dims_)*(index<Dim+1>)(is...));
    }

    template<int Dim = 0>
    LIBINTX_GPU_ENABLED
    auto index(auto i) const {
      assert(i < std::get<Dim>(dims_) || (std::get<Dim>(dims_) == 0));
      return i;
    }

  };

  template<typename T, size_t Rank>
  TensorRef(T* data, const std::array<size_t,Rank> &dims) -> TensorRef<T,Rank>;

}

#endif /* LIBINTX_MDSPAN_H */
