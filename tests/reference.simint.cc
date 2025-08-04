#include "libintx/shell.h"
#include "simint/simint.h"
#include "test.h"


#include <memory>
#include <string>

namespace libintx::simint {

  auto init(simint_multi_shellpair &P, auto &a, auto &b) {
    simint_initialize_multi_shellpair(&P);
    simint_create_multi_shellpair(a.size(), a.data(), b.size(), b.data(), &P, 0);
  }

  double time(
    std::array<int,4> Ls,
    std::array<int,4> Ks,
    std::array<int,4> dims)
  {

    simint_init();

    int maxam = 6;
    int nbf = 1;
    std::vector<simint_shell> basis[4];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < dims[i]; ++j) {
        simint_shell s;
        auto g = test::gaussian(Ls[i], Ks[i]);
        //simint_create_shell(1, 1, g.r[0], g.r[1], g.r[2], alpha, coef, &s);
        simint_initialize_shell(&s);
        simint_allocate_shell(g.K, &s);
        s.am = g.L;
        s.nprim = g.K;
        s.x = g.r[0];
        s.y = g.r[1];
        s.z = g.r[2];
        for (size_t idx = 0; auto &[a,c] : g.prims) {
          s.alpha[idx] = a;
          s.coef[idx] = c;
          ++idx;
        }
        //simint_create_shell(1, 1, 0,0,0, alpha, coef, s);
        //simint_free_shell(s);
        basis[i].push_back(s);
      }
      nbf *= ncart(Ls[i])*dims[i];
      maxam = std::max(maxam,Ls[i]);
    }

    std::vector<double> res_ints(nbf);
    double* simint_work = (double*)SIMINT_ALLOC(simint_ostei_workmem(0,maxam));

    struct simint_multi_shellpair P, Q;
    init(P, basis[0], basis[1]);
    init(Q, basis[2], basis[3]);

    double time = 0;
    timer t;
    simint_compute_eri(&P, &Q, 0.0, simint_work, res_ints.data());
    time = t;
    //println("ok");

    SIMINT_FREE(simint_work);

    simint_free_multi_shellpair(&P);
    simint_free_multi_shellpair(&Q);

    for (int i = 0; i < 4; ++i) {
      for (auto &s : basis[i]) {
        simint_free_shell(&s);
      }
    }

    return time;

  }

}
