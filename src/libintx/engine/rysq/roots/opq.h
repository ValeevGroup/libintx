#ifndef RYSQ_OPQ_H
#define RYSQ_OPQ_H

#include <math.h>

namespace rysq {
namespace opq {

    template<typename T>
    T pow2(T x) { return x*x; }

    template<int M, int N, typename T>
    static int stieltjes(T *x, T *w, T *a, T *b) {
	T p0[N], p1[N], p2[N];

	static const T TINY_ = 1.0E-40;
	static const T HUGE_ = 1.0E+40;

	if (M <= 0 || M > N) return 1;

	T sum0 = 0.0;
	T sum1 = 0.0;
	T sum2;

	for (int i = 0; i < N; ++i) {
	    sum0 = sum0 + w[i];
	    sum1 = sum1 + w[i]*x[i];
	}
	a[0] = sum1/sum0;
	b[0] = sum0;

	if (M == 1) return 0;

	for (int i = 0; i < N; ++i) {
	    p1[i] = 0.0;
	    p2[i] = 1.0;
	}

	for (int k = 0; k < M-1; ++k) {
	    sum1 = 0.0;
	    sum2 = 0.0;
	    for (int m=0; m < N; ++m) {

		if(w[m] == 0.0) continue;
		p0[m] = p1[m];
		p1[m] = p2[m];
		p2[m] = (x[m] - a[k])*p1[m] - b[k]*p0[m];

		if(fabs(p2[m]) > HUGE_ || fabs(sum2) > HUGE_) return k+1;

		T t = w[m]*p2[m]*p2[m];
		sum1 = sum1 + t;
		sum2 = sum2 + t*x[m];
	    }
	    if (fabs(sum1) < TINY_) {
		return -(k+1);
	    }

	    a[k+1] = sum2/sum1;
	    b[k+1] = sum1/sum0;
	    sum0 = sum1;
	}
	return 0;
    }

    template<typename T>
    static T pythag(T a, T b) {
	T absa,absb;
	absa=fabs(a);
	absb=fabs(b);
	if (absa > absb) return absa*sqrt(1.0+pow2(absb/absa));
	else return (absb == 0.0 ? 0.0 : absb*sqrt(1.0+pow2(absa/absb)));
    }

    template<typename T>
    static void eigsrt(int n, T *d, T *v){

	int k,j,i;
	T p;
	T *d1 = d - 1;
	T *v1 = v - 1;

	for (i=1;i<n;i++) {
	    p=d1[k=i];
	    for (j=i+1;j<=n;j++)
		if (d1[j] >= p) p=d1[k=j];
	    if (k != i) {
		d1[k]=d1[i];
		d1[i]=p;

		p=v1[i];
		v1[i]=v1[k];
		v1[k]=p;
	    }
	}
    }

    template<typename T>
    static int tqli(int n, T *d, T *e, T *z) {
	int m,l,iter,i;
	T s,r,p,g,f,dd,c,b;

	T *d1 = d - 1;
	T *e1 = e - 1;
	T *z1 = z - 1;


	//     for (i=2;i<=n;i++) e1[i-1]=e1[i];
	//     e1[n]=0.0;

	for (l=1;l<=n;l++) {
	    iter=0;
	    do {
		for (m=l;m<=n-1;m++) {
		    dd=fabs(d1[m])+fabs(d1[m+1]);
		    if ((T)(fabs(e1[m])+dd) == dd) break;
		}
		if (m != l) {
		    if (iter++ == 30) return 1;
		    g = (d1[l+1]-d1[l])/(2.0*e1[l]);
		    r = pythag(g, T(1));
		    g = d1[m]-d1[l]+e1[l]/(g+copysign(r,g));
		    s = c = 1.0;
		    p = 0.0;
		    for (i = m-1;i>=l;i--) {
			f=s*e1[i];
			b=c*e1[i];
			e1[i+1]=(r=pythag(f,g));
			if (r == 0.0) {
			    d1[i+1] -= p;
			    e1[m]=0.0;
			    break;
			}
			s=f/r;
			c=g/r;
			g=d1[i+1]-p;
			r=(d1[i]-g)*s+2.0*c*b;
			d1[i+1]=g+(p=s*r);
			g=c*r-b;

			// First element of eigenvector
			f = z1[i+1];
			z1[i+1] = s*z1[i] + c*f;
			z1[i] = c*z1[i] - s*f;
		    }

		    if (r == 0.0 && i >= l) continue;
		    d1[l] -= p;
		    e1[l]=g;
		    e1[m]=0.0;
		}
	    } while (m != l);
	}

	return 0;
    }

    template<int N, typename T>
    static int coefficients(const T *a, const T *b, T *x, T *w) {
	int i;
	T e[N];

	auto *a1 = a - 1;
	auto *b1 = b - 1;
	auto *x1 = x - 1;
	auto *w1 = w - 1;
	auto *e1 = e - 1;

	if (N < 1) return -1;
	if (b1[1] < 0.0) return -2;

	x1[1] = a1[1];
	w1[1] = b1[1];
	if (N == 1) return 0;

	w1[1] = 1.0;
	e1[N] = 0.0;

	for (i = 2; i <= N; ++i) {
	    x1[i] = a1[i];
	    if (b1[i] < 0.0) return -2;
	    e1[i-1] = sqrt(b1[i]);
	    w1[i] = 0.0;
	}


	int status = tqli(N, x, e, w);
	eigsrt(N, x, w);

	for (i = 1; i <= N; i++) {
	    w1[i] = b1[1]*pow2(w1[i]);
	}

	return status;
    }

}
}

#endif /* RYSQ_OPQ_H */
