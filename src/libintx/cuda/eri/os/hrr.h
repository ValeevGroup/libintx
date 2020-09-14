#ifndef LIBINT_CUDA_OS_HRR_H
#define LIBINT_CUDA_OS_HRR_H

#include "libintx/array.h"
#include "libintx/shell.h"

namespace libintx::cuda::os::hrr {

  struct Index3 {
    uint32_t y:4,z:4;
    uint32_t idx1:12, idx2:12;
  };

  __constant__
  const auto index3_table = make_array<Index3>(
    [](auto idx) {
      constexpr auto o = cartesian::orbital(idx);
      uint32_t idx1 = cartesian::index(o.L()+1);
      uint32_t idx2 = cartesian::index(o.L()+2);
      return Index3{ o[1], o[2], idx1, idx2 };
    },
    cartesian::index_sequence_ab
  );


  struct Quanta {

    uint8_t y:4,z:4;

    Quanta() = default;

    __device__
    explicit Quanta(int axis) : y(axis == 1), z(axis == 2) {}

    __device__
    uint16_t index() const {
      const auto yz = y+z;
      return (yz*(yz+1))/2 + z;
    }

  };

  template<int N0, int N1, int N2, typename T>
  constexpr T bitstring(T arg0, T arg1, T arg2) {
    return T((arg0) | (arg1 << N0) | (arg2 << N0+N1));
  }

  template<int N0, int N1, typename T>
  constexpr T bitstring(T arg0, T arg1) {
    return T((arg0) | (arg1 << N0));
  }

  template<int N, typename T, typename ... Ts>
  constexpr T bitstring(T arg0, Ts ... args) {
    if constexpr (sizeof...(args)) {
        return T((arg0) | (bitstring<N,T>(args...) << (N)));
    }
    return arg0;
  }

  template<typename LHS>
  __device__
  Quanta operator+(LHS l, Quanta r) {
    Quanta q;
    q.y = l.y + r.y;
    q.z = l.z + r.z;
    return q;
  }

  __device__
  auto index1(int idx00, Quanta q) {
    auto index3 = index3_table[idx00];
    return (index3.idx1 + (index3 + q).index());
  }

  __device__
  auto index2(int idx00, Quanta q10, Quanta q01) {
    auto index3 = index3_table[idx00];
    auto q11 = q10 + q01;
    return std::tuple(
      index3.idx1 + (index3 + q01).index(),
      index3.idx1 + (index3 + q10).index(),
      index3.idx2 + (index3 + q11).index()
    );
  }

  template<int NX>
  __device__
  void hrr_axis_transfer1(double X, const double *V1, const double *V2, double *H) {
    for (int i = 0; i < NX; ++i) {
      double h = X*V1[i] + V2[i];
      H[i] = h;
    }
  }

  template<int NX>
  __device__
  void hrr_axis_transfer2(
    double x11, const double *V00,
    double x01, const double *V10,
    double x10, const double *V01,
    const double *V11,
    double *H)
  {
    for (int i = 0; i < NX; ++i) {
      H[i] = x11*V00[i] + x01*V10[i] + x10*V01[i] + V11[i];
    }
  }


  template<int L, int K>
  struct HRR;

  template<>
  struct HRR<1,3> {

    static constexpr int NV = 1;
    static constexpr int NB = 3;

    __device__
    static auto transfer(int b) {
      return std::tuple(0,b);
    }

  };

  template<>
  struct HRR<3,10> {

    static constexpr int L = 3;
    static constexpr int NB = 10;
    static constexpr int NV = 4;

    __device__
    static auto transfer(int b) {
      enum {
        XX = 0, X = 0,
        YY = 1, Y = 1,
        ZZ = 2, Z = 2,
        XY = 3
      };
      static constexpr auto transfers = bitstring<2,uint64_t>(
        XX, X, // (200,100)
        XX, Y, // (200,010)
        XX, Z, // (200,001)
        YY, X, // (020,100)
        XY, Z, // (110,001)
        ZZ, X, // (002,100)
        YY, Y, // (020,010)
        YY, Z, // (020,001)
        ZZ, Y, // (002,010)
        ZZ, Z  // (002,001)
      );
      int idx = (transfers >> (b*4+0)) &0b11;
      int axis = (transfers >> (b*4+2)) & 0b11;
      return std::tuple(idx,axis);
    }

  };

  template<int K>
  struct HRR<2,K> {

    static constexpr int NV = 1;
    static constexpr int NB = K;

    __device__
    static auto transfer(int b) {
      uint32_t transfer;
      enum { X=0, Y=1, Z=2 };
      if (K == 3) {
        transfer = bitstring<2,uint32_t>(X,X, Y,Y, Z,Z);
      }
      if (K == 4) {
        transfer = bitstring<2,uint32_t>(X,X, Y,Y, Z,Z, X,Y);
      }
      if (K == 6) {
        transfer = bitstring<2,uint32_t>(X,X, X,Y, X,Z, Y,Y, Y,Z, Z,Z);
      }
      return std::tuple(
        0, // s-index always 0
        ((transfer >> 4*b+0) & 0b11),
        ((transfer >> 4*b+2) & 0b11)
      );
    }

  };


  template<int K>
  struct HRR<4,K> {

    static constexpr int8_t NV = 3;
    static constexpr int8_t NB = K;

    __device__
    auto transfer(int b) const {
      enum {
        X = 0, XX = 0,
        Y = 1, YY = 1,
        Z = 2, ZZ = 2,
      };
      enum {
        // 0:7
        XXXX = bitstring<2,2,2>(XX, X, X), // (200,200)
        XXXY = bitstring<2,2,2>(XX, X, Y), // (200,110)
        XXXZ = bitstring<2,2,2>(XX, X, Z), // (200,101)
        XXYY = bitstring<2,2,2>(XX, Y, Y), // (200,020)
        XXYZ = bitstring<2,2,2>(XX, Y, Z), // (200,011)
        XXZZ = bitstring<2,2,2>(XX, Z, Z), // (200,002)
        XYYY = bitstring<2,2,2>(YY, X, Y), // (020,110)
        XYYZ = bitstring<2,2,2>(YY, X, Z), // (020,101)
        // 8:14
        XYZZ = bitstring<2,2,2>(ZZ, X, Y), // (002,110)
        XZZZ = bitstring<2,2,2>(ZZ, X, Z), // (002,101)
        YYYY = bitstring<2,2,2>(YY, Y, Y), // (020,020)
        YYYZ = bitstring<2,2,2>(YY, Y, Z), // (020,011)
        YYZZ = bitstring<2,2,2>(ZZ, Y, Y), // (002,020)
        YZZZ = bitstring<2,2,2>(ZZ, Y, Z), // (002,011)
        ZZZZ = bitstring<2,2,2>(ZZ, Z, Z)  // (002,002)
      };
      if (K == 6) {
        return transfer<XXXX,YYYY,ZZZZ,XXYY,XXZZ,YYZZ>(b);
      }
      if (K == 9) {
        return transfer<XXXX,YYYY,ZZZZ, XXYY,XXZZ,YYZZ, XXYZ,XYYZ,XYZZ>(b);
      }
      if (b < 8) {
        return transfer<XXXX,XXXY,XXXZ,XXYY,XXYZ,XXZZ,XYYY,XYYZ>(b%8);
      }
      else {
        return transfer<XYZZ,XZZZ,YYYY,YYYZ,YYZZ,YZZZ,ZZZZ>(b%8);
      }
    }

    template<int ... Args>
    __device__
    static auto transfer(int8_t t) {
      uint64_t transfer = bitstring<6,uint64_t>(Args...);
      transfer = transfer >> 6*t;
      return std::tuple(
        ((transfer >> 0) & 0b11), // index B-2
        ((transfer >> 2) & 0b11), // axis01
        ((transfer >> 4) & 0b11)  // axis10
      );
    }

  };

  template<>
  struct HRR<5,21> {

    static constexpr int8_t NV = 9;
    static constexpr int8_t NB = 21;

    __device__
    auto transfer(int b) const {
      enum {
        XXXX = 0, YYYY = 1, ZZZZ = 2,
        XXYY = 3, XXZZ = 4, YYZZ = 5,
        XXYZ = 6, XYYZ = 7, XYZZ = 8,
        X = 0, Y = 1, Z = 2
      };
      enum {
        // 1:8
        XXXXX = bitstring<4,2>(XXXX,X),
        XXXXY = bitstring<4,2>(XXXX,Y),
        XXXXZ = bitstring<4,2>(XXXX,Z),
        XXXYY = bitstring<4,2>(XXYY,X),
        XXXYZ = bitstring<4,2>(XXYZ,X),
        XXXZZ = bitstring<4,2>(XXZZ,X),
        XXYYY = bitstring<4,2>(XXYY,Y),
        XXYYZ = bitstring<4,2>(XXYY,Z),
        // 9:15
        XXYZZ = bitstring<4,2>(XXZZ,Y),
        XXZZZ = bitstring<4,2>(XXZZ,Z),
        XYYYY = bitstring<4,2>(YYYY,X),
        XYYYZ = bitstring<4,2>(XYYZ,Y),
        XYYZZ = bitstring<4,2>(YYZZ,X),
        XYZZZ = bitstring<4,2>(XYZZ,Z),
        XZZZZ = bitstring<4,2>(ZZZZ,X),
        YYYYY = bitstring<4,2>(YYYY,Y),
        // 16:21
        YYYYZ = bitstring<4,2>(YYYY,Z),
        YYYZZ = bitstring<4,2>(YYZZ,Y),
        YYZZZ = bitstring<4,2>(YYZZ,Z),
        YZZZZ = bitstring<4,2>(ZZZZ,Y),
        ZZZZZ = bitstring<4,2>(ZZZZ,Z)
      };
      uint64_t transfer;
#define BITS(...) bitstring<6,uint64_t>(__VA_ARGS__);
      if (b < 8) {
        transfer = BITS(XXXXX,XXXXY,XXXXZ,XXXYY,XXXYZ,XXXZZ,XXYYY,XXYYZ);
      }
      else if (b < 16) {
        transfer = BITS(XXYZZ,XXZZZ,XYYYY,XYYYZ,XYYZZ,XYZZZ,XZZZZ,YYYYY);
      }
      else {
        transfer = BITS(YYYYZ,YYYZZ,YYZZZ,YZZZZ,ZZZZZ);
      }
#undef BITS
      transfer = transfer >> 6*(b%8);
      return std::tuple(
        ((transfer >> 0) & 0b1111), // index B-1
        ((transfer >> 4) & 0b11)    // axis
      );
    }

  };


  template<>
  struct HRR<6,28> {

    static constexpr int8_t NV = 6;
    static constexpr int8_t NB = 28;

    __device__
    auto transfer(int b) const {
      enum {
        XXXX = 0, YYYY = 1, ZZZZ = 2,
        XXYY = 3, XXZZ = 4, YYZZ = 5,
        X = 0, Y = 1, Z = 2
      };
      enum {
        // 1:8
        XXXXXX = bitstring<3,2,2>(XXXX,X,X),
        XXXXXY = bitstring<3,2,2>(XXXX,X,Y),
        XXXXXZ = bitstring<3,2,2>(XXXX,X,Z),
        XXXXYY = bitstring<3,2,2>(XXXX,Y,Y),
        XXXXYZ = bitstring<3,2,2>(XXXX,Y,Z),
        XXXXZZ = bitstring<3,2,2>(XXXX,Z,Z),
        XXXYYY = bitstring<3,2,2>(XXYY,X,Y),
        XXXYYZ = bitstring<3,2,2>(XXYY,X,Z),
        // 9:16
        XXXYZZ = bitstring<3,2,2>(XXZZ,X,Y),
        XXXZZZ = bitstring<3,2,2>(XXZZ,X,Z),
        XXYYYY = bitstring<3,2,2>(XXYY,Y,Y),
        XXYYYZ = bitstring<3,2,2>(XXYY,Y,Z),
        XXYYZZ = bitstring<3,2,2>(XXYY,Z,Z),
        XXYZZZ = bitstring<3,2,2>(XXZZ,Y,Z),
        XXZZZZ = bitstring<3,2,2>(XXZZ,Z,Z),
        XYYYYY = bitstring<3,2,2>(YYYY,X,Y),
        // 17:24
        XYYYYZ = bitstring<3,2,2>(YYYY,X,Z),
        XYYYZZ = bitstring<3,2,2>(YYZZ,X,Y),
        XYYZZZ = bitstring<3,2,2>(YYZZ,X,Z),
        XYZZZZ = bitstring<3,2,2>(ZZZZ,X,Y),
        XZZZZZ = bitstring<3,2,2>(ZZZZ,X,Z),
        YYYYYY = bitstring<3,2,2>(YYYY,Y,Y),
        YYYYYZ = bitstring<3,2,2>(YYYY,Y,Z),
        YYYYZZ = bitstring<3,2,2>(YYYY,Z,Z),
        // 24:28
        YYYZZZ = bitstring<3,2,2>(YYZZ,Y,Z),
        YYZZZZ = bitstring<3,2,2>(ZZZZ,Y,Y),
        YZZZZZ = bitstring<3,2,2>(ZZZZ,Y,Z),
        ZZZZZZ = bitstring<3,2,2>(ZZZZ,Z,Z)
      };
      uint64_t transfer;
#define BITS(...) bitstring<7,uint64_t>(__VA_ARGS__);
      if (b < 8) {
        transfer = BITS(XXXXXX,XXXXXY,XXXXXZ,XXXXYY,XXXXYZ,XXXXZZ,XXXYYY,XXXYYZ);
      }
      else if (b < 16) {
        transfer = BITS(XXYZZZ,XXZZZZ,XXYYYY,XXYYYZ,XXYYZZ,XXYZZZ,XXZZZZ,XYYYYY);
      }
      else if (b < 24) {
        transfer = BITS(XYYYYZ,XYYYZZ,XYYZZZ,XYZZZZ,XZZZZZ,YYYYYY,YYYYYZ,YYYYZZ);
      }
      else {
        transfer = BITS(YYYZZZ,YYZZZZ,YZZZZZ,ZZZZZZ);
      }
#undef BITS
      transfer = transfer >> 7*(b%8);
      return std::tuple(
        ((transfer >> 0) & 0b111), // index B-1
        ((transfer >> 3) & 0b11),  // axis 01
        ((transfer >> 5) & 0b11)   // axis 10
      );
    }

  };


  template<int AB, int NX, int Block, int L, int K>
  __device__
  void hrr_transfer1(HRR<L,K> hrr, int A, const double *R, double *V) {

    __syncthreads();

    constexpr int NV = hrr.NV;
    constexpr int NB = hrr.NB;

    const auto &thread = this_thread_block().thread_rank();
    int active_a_threads = this_thread_block().size()/NB;

    double *H = V;

    int index0 = cartesian::index(A);
    V -= NX*NV*index0; // relative to L=0

    double h[Block][NX] = {};

    if (thread/NB < active_a_threads) {

      int B0 = thread%NB; // B index
      int A0 = thread/NB; // A+0 index
      A0 += index0;

      auto [vidx,axis] = hrr.transfer(B0);
      double r0 = R[axis];
      Quanta q(axis);

#pragma unroll
      for (int i = 0; i < Block; ++i) {
        if (A0 >= cartesian::index(AB+1-L)) break;
        auto A1 = index1(A0, q);
#define V(IDX) (V + NX*vidx + NX*NV*(IDX))
        hrr_axis_transfer1<NX>(r0, V(A0), V(A1), h[i]);
#undef V
        A0 += active_a_threads;
      }

    }

    __syncthreads();

    if (thread/NB < active_a_threads) {
#pragma unroll
      for (int i = 0; i < Block; ++i) {
        int idx0 = thread + i*NB*active_a_threads;
        if (idx0 >= NB*(cartesian::index(AB+1-L)-index0)) break;
        for (int x = 0; x < NX; ++x) {
          H[x+NX*idx0] = h[i][x];
          //printf("H@%p = %f\n", H+x+NX*idx0, h[i][x]);
        }
      }
    }

  }

  template<int AB, int NX, int Block, int L, int K>
  __device__
  void hrr_transfer2(HRR<L,K> hrr, int A, const double *R, double *V) {

    __syncthreads();

    constexpr int NV = hrr.NV;
    constexpr int NB = hrr.NB;

    const auto &thread = this_thread_block().thread_rank();
    int active_a_threads = this_thread_block().size()/NB;

    double *H = V;

    int index0 = cartesian::index(A);
    V -= NX*NV*index0; // relative to L=0

    double h[Block][NX];

    if (thread/NB < active_a_threads) {

      int B0 = thread%NB; // B index
      int a00 = thread/NB + index0; // A+0 index

      auto [vidx, axis01, axis10] = hrr.transfer(B0);
      Quanta q01 = Quanta(axis01);
      Quanta q10 = Quanta(axis10);
      double x01 = R[axis01];
      double x10 = R[axis10];

#pragma unroll
      for (int i = 0; i < Block; ++i) {
        if (a00 >= cartesian::index(AB+1-L)) break;
        auto [a10,a01,a11] = index2(a00, q01, q10);
#define V(IDX) (V + NX*vidx + NX*NV*(IDX))
        hrr_axis_transfer2<NX>(
          x01*x10, V(a00),
          x01, V(a10),
          x10, V(a01),
          V(a11),
          h[i]
        );
#undef V
        a00 += active_a_threads;
      }
    }

    __syncthreads();

    if (thread/NB < active_a_threads) {
#pragma unroll
      for (int i = 0; i < Block; ++i) {
        int idx0 = thread + i*NB*active_a_threads;
        if (idx0 >= NB*(cartesian::index(AB+1-L)-index0)) break;
        for (int x = 0; x < NX; ++x) {
          H[x+NX*idx0] = h[i][x];
          //printf("H@%p = %f\n", H+x+NX*idx0, h[i][x]);
        }
      }
    }

  }

  constexpr int Block(int AB, int NX) {
    if (AB <= 3) return 1;
    if (AB <= 5) return 2;
    if (NX <= 5) return 4;
    if (NX <= 7) return 3;
    return 2;
  }

  template<int AB, int NX>
  __device__
  double* hrr(int A, int B, const double *Xab, double *V) {
    if constexpr (AB == 0) { return V; }
    else {
      assert(blockDim.x == 32);
      static constexpr int Block = hrr::Block(AB,NX);
      if (AB > 1 && B == 1) {
        hrr_transfer1<AB,NX,Block>(HRR<1,3>(), AB-1, Xab, V);
        return V;
      }
      if (AB > 2 && B == 2) {
        hrr_transfer2<AB,NX,Block>(HRR<2,6>(), AB-2, Xab, V);
        return V;
      }
      if (AB > 3 && B == 3) {
        hrr_transfer2<AB,NX,Block>(HRR<2,4>(), AB-3, Xab, V);
        hrr_transfer1<AB,NX,Block>(HRR<3,10>(), AB-3, Xab, V);
        return V;
      }
      hrr_transfer2<AB,NX,Block>(HRR<2,3>(), A, Xab, V);
      if (AB > 4 && B == 4) {
        hrr_transfer2<AB,NX,Block>(HRR<4,15>(), AB-4, Xab, V);
        return V;
      }
      if (AB > 5 && B == 5) {
        hrr_transfer2<AB,NX,Block>(HRR<4,9>(), AB-5, Xab, V);
        hrr_transfer1<AB,NX,Block>(HRR<5,21>(), AB-5, Xab, V);
        return V;
      }
      if (AB > 6 && B == 6) {
        hrr_transfer2<AB,NX,Block>(HRR<4,6>(), AB-6, Xab, V);
        hrr_transfer2<AB,NX,Block>(HRR<6,28>(), AB-6, Xab, V);
        return V;
      }
    }
    __trap();
    return nullptr;
  }

  constexpr int num_threads(int A, int B, int X) {
    int intermediates = 0;
    // p,d require no intermediates
    if (B < 3) {
      intermediates = 0;
    }
    // f require 4 d-intermediates
    if (B == 3) {
      intermediates = 4*(ncart(A)+ncart(A+1));
    }
    if (B > 4) {
      // g (and above) require 3 d-intermediates
      intermediates = 3*(ncart(A)+ncart(A+1)+ncart(A+2));
    }
    if (B == 5) {
      intermediates = std::max(
        intermediates,
        9*(ncart(A)+ncart(A+1))
      );
    }
    if (B == 6) {
      intermediates = std::max(
        intermediates,
        6*(ncart(A)+ncart(A+1)+ncart(A+2))
      );
    }
    int Block = hrr::Block(A+B, npure(X));
    int threads = std::max(intermediates, ncart(A)*ncart(B));
    return (threads+Block-1)/Block;
  }

  int memory(int A, int B, int X) {
    int NA = ncart(A);
    int NB = ncart(B);
    int NX = npure(X);
    // p,d require no intermediates
    if (B < 3) return 0;
    // f require 4 d-intermediates
    if (B == 3) return NX*(ncartsum(A+1)-ncartsum(A-1))*4;
    // g (and above) return 3 d-intermediates
    int m4 = NX*(ncartsum(A+B-2)-ncartsum(A-1))*3; //
    if (B == 4) return m4;
    if (B == 5) return std::max(m4, NX*(ncartsum(A+1)-ncartsum(A-1))*9);  // d/g
    if (B == 6) return std::max(m4, NX*(ncartsum(A+2)-ncartsum(A-1))*6);  // d/g
    throw std::runtime_error("HRR: B > 6 not implemented");
  }

}

#endif /* LIBINT_CUDA_OS_HRR_H */
