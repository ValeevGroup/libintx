#ifndef RYSQ_VECTOR_H
#define RYSQ_VECTOR_H

#include "rysq/config.h"
#include "rysq/constants.h"

namespace rysq {

template<typename T, int N>
struct Vector {

  double data[N] = {};

  static constexpr int size() {
    return N;
  }

  T operator[](int idx) const {
    return data[idx];
  }

  T& operator[](int idx) {
    return data[idx];
  }

  operator T*() {
    return data;
  }

  Vector& operator/=(const T &s) {
    for (int i = 0; i < N; ++i) {
      this->data[i] /= s;
    }
    return *this;
  }

};

template<int N>
struct Vector<Zero,N> {
  constexpr Zero operator[](int idx) const {
    return Zero();
  }
};

template<typename T, int N, class F>
Vector<T,N> eval(const F &f) {
  Vector<T,N> v;
  for (int i = 0; i < N; ++i) {
    v[i] = f(i);
  }
  return v;
}

template<typename T, int N>
auto operator-(const Vector<T,N> &u, const Vector<Zero,N> &v) {
  return u;
}

template<typename T, int N>
auto operator-(const Vector<T,N> &u, const Vector<T,N> &v) {
  return eval<T,N>([&](int idx) { return (u[idx] - v[idx]); });
}

template<typename T, int N>
auto operator*(const T &s, const Vector<T,N> &v) {
  return eval<T,N>([&](int idx) { return s*v[idx]; });
}

template<typename T, int N>
Vector<T,N> center_of_charge(T a, const Vector<T,N> &u, T b, const Vector<T,N> &v) {
  return eval<T,N>(
    [&](int idx) {
      return (a*u[idx] + b*v[idx])/(a+b);
    }
  );
}

template<typename T, int N>
const Vector<T,N>& center_of_charge(T a, const Vector<T,N> &u, Zero, Vector<Zero,N>) {
  return u;
}

template<typename T, int N>
T dot(const Vector<T,N> &v) {
  T dot = 0;
  for (int i = 0; i < N; ++i) {
    dot += v[i]*v[i];
  }
  return dot;
}

typedef Vector<double,3> Vector3;

}

#endif /* RYSQ_VECTOR_H */
