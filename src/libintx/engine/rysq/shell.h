#ifndef RYSQ_SHELL_H
#define RYSQ_SHELL_H

#include "rysq/constants.h"
#include "rysq/vector.h"

#include <stdint.h>
#include <type_traits>
#include <string>
#include <vector>
#include <array>
#include <cmath>

namespace rysq {
namespace shell {

  inline int nbf(int L) {
    int n = L+1;
    return ((n*n+n)/2);
  }

  struct Shell {

    struct Unit;

    struct Orbital {
      //uint16_t x:4, y:4, z:4;
      uint16_t x, y, z;
    } __attribute__((aligned(16)));

    const int L;

    struct Primitive { double a, C; };
    const std::vector<Primitive> prims;

    Shell(int L, double a, double C)
      : Shell(L, {Primitive{a,C}})
    {
    }

    Shell(int L, std::vector<Primitive> p)
      : L(L), prims(p)
    {
      auto it = this->orbitals_;
      for (int k = 0; k <= L; ++k) {
        for (int z = 0; z <= k; ++z) {
          it->x = L-k;
          it->y = k-z;
          it->z = z;
          ++it;
        }
      }
    }

    auto begin() const {
      return this->orbitals_;
    }

    auto end() const {
      return this->orbitals_ + nbf(this->L);
    }

    // const auto &prims() const { return prims_; }

  private:
    Orbital orbitals_[RYSQ_MAX_CART];

  };


  struct Shell::Unit {

    static const int L = 0;

    struct Primitive {
      const Constant<0> a;
      const Constant<1> C;
    };

    // static constexpr auto prims() {
    //   return std::array<Primitives,1>{ Primitives{} };
    // }

    struct Iterator {
      static const int x = 0, y = 0, z = 0;
      Iterator& operator++() {
        this->index_ = 1;
        return *this;
      }
      bool operator!=(const Iterator &it) const {
        return (it.index_ != this->index_);
      }
      const Iterator* operator->() const {
        return this;
      }
    private:
      friend struct Unit;
      explicit Iterator(int index)
        : index_(index)
      {
      }
      int index_;
    };

    auto begin() const { return Iterator{0}; }
    auto end() const { return Iterator{1}; }

  };


  inline int nbf(const Shell &s) {
    return nbf(s.L);
  }

  inline int nbf(const Shell::Unit &s) {
    return 1;
  }

  template<class ... Args>
  int nbf(const Shell &s, Args&& ... args) {
    return nbf(s)*nbf(args...);
  }


  template<int>
  struct Tuple;

  template<>
  struct Tuple<1> {
    static const size_t size = 1;
    Shell first;
    const auto& get(std::integral_constant<int,0>) const { return first; }
    auto get(std::integral_constant<int,1>) const { return Shell::Unit{}; }
  };

  template<>
  struct Tuple<2> {
    static const size_t size = 2;
    Shell first;
    Shell second;
    const auto& get(std::integral_constant<int,0>) const { return first; }
    const auto& get(std::integral_constant<int,1>) const { return second; }
  };

  template<int K, int N>
  auto get(const Tuple<N> &t)
    -> decltype(t.get(std::integral_constant<int,K>{}))
  {
    return t.get(std::integral_constant<int,K>{});
  }

  template<int N>
  inline int L(const Tuple<N> &t) {
    return (get<0>(t).L + get<1>(t).L);
  }

  template<int N>
  inline int nbf(const Tuple<N> &t) {
    return nbf(get<0>(t))*nbf(get<1>(t));
  }

  inline std::string str(const Tuple<2> &p) {
    return std::to_string(p.first.L) + std::to_string(p.second.L);
  }

  inline std::string str(const Tuple<1> &p) {
    return std::to_string(p.first.L);
  }


  inline const auto& primitives(const Shell &s) {
    return s.prims;
  }

  inline auto primitives(const Shell::Unit &s) {
    return std::array<Shell::Unit::Primitive,1>{};
  }

  template<int N>
  inline int nprims(const Tuple<N> &t) {
    return (
      primitives(get<0>(t)).size()*
      primitives(get<1>(t)).size()
    );
  }


  struct Primitives {
    double e, A;
    Vector3 rA;
  };

  inline double exp(double ai, double aj, double rij) {
    return std::exp(-ai*aj*rij/(ai+aj));
  }

  inline double exp(double ai, Zero aj, double rij) {
    return 1;
  }

  template<int N, class R0, class R1>
  inline std::vector<Primitives> primitives(const Tuple<N> &tuple, const R0 &ri, const R1 &rj) {
    double rij = dot(ri-rj);
    const auto &P = shell::get<0>(tuple);
    const auto &Q = shell::get<1>(tuple);
    std::vector<Primitives> prims;
    prims.reserve(nprims(tuple));
    for (const auto &q : primitives(Q)) {
      for (const auto &p : primitives(P)) {
        double e = p.C*q.C*shell::exp(p.a, q.a, rij);
        double A = p.a+q.a;
        Vector3 rA = center_of_charge(p.a, ri, q.a, rj);
        prims.push_back({e, A, rA});
      }
    }
    return prims;
  }

  struct Primitives2 {
    double C;
    double A; Vector3 rA;
    double B; Vector3 rB;
  };

  inline Primitives2 primitives(const Primitives &bra, const Primitives &ket) {
    double A = bra.A;
    double B = ket.A;
    double C = (bra.e*ket.e)/(A*B*sqrt(A+B));
    return Primitives2{ C, A, bra.rA, B, ket.rA };
  }

  template<int K>
  using Bra = Tuple<K>;

  template<int K>
  using Ket = Tuple<K>;


  template<class Bra, class Ket>
  struct centers;

  template<>
  struct centers< Bra<2>, Ket<2> > {
    const Vector<double,3> &ri, &rj, &rk, &rl;
  };

  template<>
  struct centers< Bra<2>, Ket<1> > {
    const Vector<double,3> &ri, &rj, &rk;
    Vector<Zero,3> rl;
  };

  template<>
  struct centers< Bra<1>, Ket<1> > {
    const Vector<double,3> &ri, &rk;
    Vector<Zero,3> rj, rl;
  };

}
}

#endif /* RYSQ_SHELL_H */
