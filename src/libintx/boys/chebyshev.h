#ifndef LIBINTX_BOYS_CHEBYSHEV_H
#define LIBINTX_BOYS_CHEBYSHEV_H

#include "libintx/boys/boys.h"
#include "libintx/boys/asymptotic.h"
#include "libintx/boys/reference.h"
#include "libintx/math/interpolate/chebyshev.h"
#include "libintx/math.h"
#include "libintx/forward.h"

namespace libintx::boys {

  template<typename T, typename R = long double>
  std::unique_ptr<T[]> chebyshev_interpolation_table(int Order, int M, int MaxT, int Segments) {
    // printf("chebyshev_interpolation_table: %i, %i, %i\n", Order, M, MaxT);
    R Delta = R(MaxT)/R(Segments);
    math::ChebyshevInterpolation<R> interpolate(Order);
    boys::Reference reference;
    std::unique_ptr<T[]> table(new (std::align_val_t{64}) T[(Order+1)*M*(Segments+1)]);
    for (int i = 0; i < Segments; ++i) {
      R a = i*Delta;
      R b = a + Delta;
      for (int m = 0; m < M; ++m) {
        auto f = [m,&reference](R x) {
          return (R)reference.maclaurin(x,m);
        };
        auto p = interpolate.generate(f, a, b);
        size_t idx = m*(Order+1) + i*M*(Order+1);
        for (int k = 0; k <= Order; ++k) {
          table[k+idx] = p[k];
        }
      }
    }
    // pad end with 0s
    for (int k = 0; k <= Order*M; ++k) {
      table[Segments*M*(Order+1) + k] = 0;
    }
    return table;
  }

  template<typename T, int Order, int M, int MaxT, int Segments>
  const auto& chebyshev_interpolation_table() {
    static auto table = std::shared_ptr(
      chebyshev_interpolation_table<T>(Order, M, MaxT, Segments)
    );
    return table;
  }

  template<int Order, int M, int MaxT, int Segments>
  struct Chebyshev : Boys {

    constexpr static double Delta = double(MaxT)/Segments;

    Chebyshev() {
      //printf("Chebyshev::Chebyshev()\n");
      table_ = chebyshev_interpolation_table<double,Order,M,MaxT,Segments>();
    }

    template<int N, typename T, size_t K>
    LIBINTX_ALWAYS_INLINE
    void compute(T x, T (&Fm)[K]) const {
      compute<N>(x, std::integral_constant<int,0>{}, Fm);
    }

    template<int N, typename T, size_t K>
    LIBINTX_ALWAYS_INLINE
    //__attribute__((noinline))
    void compute(const T &x, auto m, T (&Fm)[K]) const {

      static_assert(K == N+1);
      static_assert(N < M);
      assert(m+N < M);

      const auto* __restrict__ table = this->table_.get();

      // ---------------------------------------------
      // small and intermediate arguments => interpolate Fm and (optional) downward recursion
      // ---------------------------------------------
      // which interval does this x fall into?
      const T x_over_delta = x/Delta;

      if constexpr (std::is_scalar_v<T>) {
        assert(x >= 0);
        if (x_over_delta >= Segments) {
          asymptotic(x, m, Fm);
          return;
        }
        auto k = static_cast<int>(x_over_delta); // the interval index
        assert(k < Segments);
        int pidx = m*(Order+1) + k*M*(Order+1);
        double xd = x_over_delta - static_cast<double>(k) - 0.5;
#if __has_cpp_attribute(vector_size)
        //#pragma message("boys::Chebyshev::compute<double> vectorised")
        // assume we have gcc vector extensions
        static constexpr int W = 2;
        typedef double vdouble __attribute__((vector_size(W*8)));
        vdouble xi = { 1, 1 } ; // , xd*xd, xd*xd*xd };
#else
        static constexpr int W = 1;
        typedef double vdouble;
        double xi = 1;
#endif
        const auto* __restrict__ p = table + pidx;
        vdouble Fm2[N+1] = {};
        libintx_unroll(10)
        for (int i = 0; i <= Order; i += W) {
          libintx_unroll(10)
          for (int m = 0; m <= N; ++m) {
            auto C = *reinterpret_cast<const vdouble*>(&p[i + m*(Order+1)]);
            Fm2[m] += C*xi;
          }
          xi *= math::pow<W>(xd);
        }
        libintx_unroll(10)
        for (int m = 0; m <= N; ++m) {
          if constexpr (std::is_scalar_v<vdouble>) {
            Fm[m] = Fm2[m];
          }
          else {
            // *reinterpret_cast<vdouble*>(Fm+m) = (
            //   __builtin_shufflevector(Fm2[m], Fm2[m+1], 0, 0) +
            //   __builtin_shufflevector(Fm2[m], Fm2[m+1], 1, 1)
            // );
            Fm[m] = Fm2[m][0] + xd*Fm2[m][1];
          }
        }
      }
      else {

        auto asymptotic_mask = (x_over_delta >= Segments);

        if (any_of(asymptotic_mask)) {
          T x1 = 1/x;
          where(!asymptotic_mask, x1) = T(0);
          asymptotic_1_x(x1, m, Fm);
          if (all_of(asymptotic_mask)) return;
        }

        auto k = static_simd_cast<int64_t>(x_over_delta);
        //assert(all_of(k < Segments));
        T xi = T(1);
        T xd = x_over_delta - static_simd_cast<T>(k) - 0.5;
        auto pidx = m*(Order+1) + k*M*(Order+1);

        where(asymptotic_mask, xi) = T(0);
        where(asymptotic_mask, xd) = T(0);
        where(k >= Segments, pidx) = decltype(pidx)(0);

#ifdef __AVX2__
        {

          constexpr bool downward_recursion = false;//(N > 1);
          constexpr int m0 = (downward_recursion ? N : 0);

          constexpr int Lanes = T::size();

          static_assert(Order == 7);
          static_assert(Lanes%4 == 0);

          __m256d f[Lanes/2][N+1] = {};

libintx_unroll(16)
          for (int l = 0; l < Lanes; l += 2) {
            if (asymptotic_mask[l+0] && asymptotic_mask[l+1]) continue;

            using math::pow;

            double xd0 = pow<2,double>(xd[l+0]);
            double xd1 = pow<2,double>(xd[l+1]);

            __m256d x0_4 = _mm256_set1_pd(xd0*xd0);
            __m256d x1_4 = _mm256_set1_pd(xd1*xd1);

            __m256d x0 = { 1, 1, xd0, xd1 };
            __m256d x1 = x0*__m256d{ xd[l+0], xd[l+1], xd[l+0], xd[l+1] };

            //libintx_unroll(16
            for (int m = m0; m <= N; ++m) {
              int im = m*(Order+1);
              auto a0 = _mm256_load_pd(table + pidx[l+0] + 0 + im);
              auto a4 = _mm256_load_pd(table + pidx[l+0] + 4 + im);
              auto a = a0 + a4*x0_4;
              auto b0 = _mm256_load_pd(table + pidx[l+1] + 0 + im);
              auto b4 = _mm256_load_pd(table + pidx[l+1] + 4 + im);
              auto b = b0 + b4*x1_4;
              // [ a, b, a, b ]
              auto ab = (
                x0*_mm256_unpacklo_pd(a,b) +
                x1*_mm256_unpackhi_pd(a,b)
              );
              f[l/2][m] = ab;
            }

          } // Lanes

libintx_unroll(16)
          for (int l = 0; l < Lanes; l += 4) {
            for (int m = m0; m <= N; ++m) {
              const auto &f0 = f[l/2+0][m]; // abab
              const auto &f1 = f[l/2+1][m]; // cdcd
              // swap 128bit halves st p = f0[2:3], f1[0:1]
              auto p = _mm256_permute2f128_pd(f0, f1, 0x21);
              auto &xi4 = *(reinterpret_cast<const __m256d*>(&xi)+l/4);
              auto &Fm4 = *(reinterpret_cast<__m256d*>(Fm+m)+l/4);
              // blend f0,f1 ->  f0[0], f0[1], f1[2], f1[3]
              // xi4 acts as asymptotic mask
              Fm4 += xi4*(p + _mm256_blend_pd(f0, f1, 0b1100));
            }
          }

          return;

        } // avx2
#endif // __AVX2__

        libintx_unroll(16)
        for (int i = 0; i <= Order; ++i) {
          libintx_unroll(10)
          for (int m = 0; m <= N; ++m) {
            auto C = T(
              [&](auto Lane) {
                return table[pidx[Lane] + i + m*(Order+1)];
              }
            );
            Fm[m] += C*xi;
          }
          xi *= xd;
        }
      }

    }

    LIBINTX_ALWAYS_INLINE
    double compute(double x, int m) const override {
      double Fm[1] = {};
      compute<0>(x, m, Fm);
      return Fm[0];
    }

  private:
    std::shared_ptr<double[]> table_;

  };

  template<int L, int Lmax = L>
  auto& chebyshev() {
    static_assert(Lmax <= L);
    constexpr int M = (Lmax <= 8 ? std::min(8,L) : L) + 1;
    constexpr int MaxT = [] {
      if (M <= 9) return 36;
      return 117;
    }();
    static auto boys = boys::Chebyshev<7,M,MaxT,MaxT*7>{};
    return boys;
  }

}

#endif /* LIBINTX_BOYS_CHEBYSHEV_H */
