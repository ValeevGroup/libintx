#include "rysq/config.h"
#include <math.h>

namespace rysq {

  template<int N, int K>
  struct Polynomial {
    double data[K][N];
  };

  template<int N, int Kmin, int Kmax>
  struct Hermite {
    double data[Kmax-Kmin][N];
  };


  template<int Kmin, int Kmax, int N>
  __device__
  void sumXk(double X, const double (&Ck)[Kmax-Kmin][N], double (&u)[N], int idx) {
    double Xk = 1;
#pragma unroll 1
    for (int k = 1; k <= -Kmin; ++ k) {
      Xk *= X;
      u[idx] += (Ck[-Kmin-k][idx])*(1/Xk);
    }

    Xk = 1 ;
    #pragma unroll 1
    for (int k = 0; k <= Kmax; ++k) {
      u[idx] += Ck[k-Kmin][idx]*Xk;
      Xk = Xk*X;
    }

    //u[idx] += uk;

  }

  template<int N, int ... Ku, int ... Kw>
  __device__
  void fit(
    double X, double (&u)[N], double (&w)[N], int idx,
    const Hermite<N,Ku...> &fu,
    const Hermite<N,Kw...> &fw)
  {
    sumXk<Ku...>(X, fu.data, u, idx);
    sumXk<Kw...>(X, fw.data, w, idx);
    double e = exp(X);
    {//for (int i = 0; i < N; ++i) {
      u[idx] = e*u[idx];
      w[idx] = e*w[idx];
    }
  }

  __device__
  void evaluate(double X, double (&u)[3], double (&W)[3], int idx) {
  }

  template<class F, class ... Fs>
  __device__
  void evaluate(double X, double (&u)[3], double (&W)[3], int idx, F &&f, Fs&& ...fs) {
    if (X < f.Xmax) {
      fit(X, u, W, idx, f.X, f.W);
      return;
    }
    evaluate(X, u, W, idx, fs...);
  }

}
