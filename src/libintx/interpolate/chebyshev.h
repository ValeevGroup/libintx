#ifndef BOYS_INTERPOLATE_H
#define BOYS_INTERPOLATE_H

#include "libintx/boys/reference.h"
#include <Eigen/Dense>
#include <vector>

namespace boys {

  template<typename T, int Rows = Eigen::Dynamic>
  using Vector = Eigen::Matrix<T, Rows, 1>;

  template<typename T, int Rows = Eigen::Dynamic, int Cols = Eigen::Dynamic>
  using Matrix = Eigen::Matrix<T, Rows, Cols>;

  template<typename Real>
  inline Real polyval(const Vector<Real> &p, Real x) {
    Real px = 0;
    Real xk = 1;
    for (int k = 0; k < p.size(); ++k) {
      px += xk*p[k];
      xk *= x;
    }
    return px;
  }

  template<typename Real = double>
  struct ChebyshevInterpolation {

    explicit ChebyshevInterpolation(int Order) {

      this->ChebyshevT_ = {
        Vector<Real,1>(1.0),
        Vector<Real,2>(0.0, 1.0)
      };

      for (int k = 2; k <= Order; ++k) {
        Vector<Real> pk(k+1);
        pk << 0, 2*ChebyshevT_[k-1];
        pk.head(k-1) -= ChebyshevT_[k-2];
        this->ChebyshevT_.push_back(pk);
        // for (int i = 0; i <= k; ++i) {
        //   printf("%f ", pk[i]);
        // }
        // printf("\n");
      }

      x_.resize(Order+1);
      for (int k = 1; k <= x_.size(); ++k) {
        x_[k-1] = std::cos(
          ((2*k-1)*EIGEN_PI)/
          (2*x_.size())
        )/2;
      }

      Matrix<Real> A(Order+1,Order+1);
      for (int i = 0; i <= Order; ++i) {
        for (int k = 0; k <= Order; ++k) {
          A(k,i) = boys::polyval(ChebyshevT_[i], x_[k]);
        }
      }
      this->lu_ = A.fullPivLu();
      //this->qr_ = A.fullPivHouseholderQr();

    }

    Vector<Real> generate(std::function<double(double)> f, double a, double b) const {

      const auto &x = this->x_;

      Vector<Real> fk(x.size());
      for (int k = 0; k < fk.size(); ++k) {
        double xk = x[k];
        xk = (b-a)*x[k]+(b+a)/2;
        fk[k] = f(xk);
      }

      Vector<Real> c = lu_.solve(fk);
      //Vector<Real> c = qr_.solve(fk);

      Vector<Real> p = Vector<Real>::Zero(c.size());
      for (int i = 0; i < c.size(); ++i) {
        p.head(i+1) += c[i]*ChebyshevT_[i];
      }

      return p;

    }

    static Real polyval(const Vector<Real> &p, Real x, Real a, Real b) {
      Real xd = (x - (b+a)/2)/(b-a);
      return boys::polyval(p,xd);
    }

  private:

    std::vector< Vector<Real> > ChebyshevT_;
    Vector<Real> x_;
    //Eigen::FullPivHouseholderQR< Matrix<Real> > qr_;
    Eigen::FullPivLU< Matrix<Real> > lu_;

  };

}

#endif /* BOYS_INTERPOLATE_H */
