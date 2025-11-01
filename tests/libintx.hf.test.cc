#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"
using libintx::test::zeros;

#include "libintx/hf/engine.h"
#include "libintx/hf/guess.h"
#include "libintx/ao/engine.h"

using namespace libintx;
using libintx::hf::FockEngine;

struct Ref {
  struct Matrix : decltype(zeros(0,0)) {
    using InitializerList = std::initializer_list<std::initializer_list<double> >;
    Matrix(const InitializerList& args = {}) {
      *this = args;
    }
    Matrix& operator=(const InitializerList& args) {
      Matrix::Index n = args.size();
      this->resize({n,n});
      this->setValues(args);
      return *this;
    }
  };
  std::string name;
  std::string basis;
  Matrix G;
  Matrix H;
  Matrix J;
  Matrix K;
  Matrix D;
};


using Atom = std::tuple<int, std::array<double,3> >;

const std::map< std::string, std::vector<Atom> > mol = {
  {
    "h2", {
#include "data/mol/h2.h"
    }
  },
  {
    "h2o", {
#include "data/mol/h2o.h"
    }
  },
  {
    "hno3", {
#include "data/mol/hno3.h"
    }
  },
  {
    "kno3", {
#include "data/mol/kno3.h"
    }
  },
  {
    "znso4", {
#include "data/mol/znso4.h"
    }
  },
  {
    "aspirin", {
#include "data/mol/aspirin.h"
    }
  },
  {
    "disiloxane", {
#include "data/mol/disiloxane.h"
    }
  }
};

using BasisSet = std::map<
  int,
  std::vector<
    std::tuple<int, std::vector<Gaussian::Primitive> >
  >
>;

std::map<std::string,BasisSet> gto = {
  {
    "6-31g", {
#include "data/gto/6-31g.h"
    }
  },
  {
    "6-31g*", {
#include "data/gto/6-31g*.h"
    }
  },
  {
    "def2-svp", {
#include "data/gto/def2-svp.h"
    }
  },
  {
    "def2-tzvp", {
#include "data/gto/def2-tzvp.h"
    }
  },
  {
    "cc-pvdz", {
#include "data/gto/cc-pvdz.h"
    }
  }
};

void test_subcase(Ref ref) {

  auto subcase = ref.name + "." + ref.basis;

  SUBCASE(subcase.c_str()) {

    auto wfn = libintx::Wfn<Gaussian>(
      ::mol.at(ref.name),
      ::gto.at(ref.basis)
    );

    printf("- %s", subcase.c_str());
    if (wfn.basis()->max().L > LMAX) {
      printf(" skipped\n");
      return;
    }
    printf("\n");

    auto engine = hf::FockEngine(wfn, ao::screening::norm<2>, 1e-16);
    engine.num_threads = { 1 };

    size_t N = wfn.basis()->nbf();

    auto H = zeros(N,N);
    engine.T({ H.data(), N });
    engine.V({ H.data(), N });
    test::check2(
      [&](auto ref, auto ... idx) {
        CHECK(H(idx...) == ref.epsilon(1e-9));
      },
      ref.H
    );

    auto G = zeros(N,N);
    hf::SOAD guess;
    guess.precision = 1e-16;
    guess.fock(wfn, { G.data(), N });
    test::check2(
      [&](auto ref, auto ... idx) {
        CHECK(G(idx...) == ref.epsilon(1e-10));
      },
      ref.G
    );

    auto X = zeros(N,N);
    auto [NX,s_cond,xtx_cond] = engine.X({ X.data(), N });

    auto F = zeros(N,N);
    auto D = zeros(N,N);
    F = H + G;
    engine.D(
      { F.data(), N },
      { X.data(), N }, NX,
      { D.data(), N }
    );
    test::check2(
      [&](auto ref, auto ... idx) {
        CHECK(D(idx...) == ref.epsilon(1.5e-9));
      },
      ref.D
    );

  }

}

TEST_CASE("libintx.hf") {

  // 6-31g

  test_subcase(
#include "data/h2.6-31g/hf.h"
  );

  test_subcase(
#include "data/h2o.6-31g/hf.h"
  );

  test_subcase(
#include "data/hno3.6-31g/hf.h"
  );

  test_subcase(
#include "data/kno3.6-31g/hf.h"
  );

  test_subcase(
#include "data/znso4.6-31g/hf.h"
  );

  // 6-31g*

  test_subcase(
#include "data/h2.6-31g*/hf.h"
  );

  test_subcase(
#include "data/h2o.6-31g*/hf.h"
  );

  test_subcase(
#include "data/hno3.6-31g*/hf.h"
  );

  test_subcase(
#include "data/kno3.6-31g*/hf.h"
  );

  test_subcase(
#include "data/znso4.6-31g*/hf.h"
  );

  // def2-svp

  test_subcase(
#include "data/aspirin.def2-svp/hf.h"
  );

  test_subcase(
#include "data/disiloxane.def2-svp/hf.h"
  );

  test_subcase(
#include "data/znso4.def2-svp/hf.h"
  );

  // def2-tzvp

  test_subcase(
#include "data/h2.def2-tzvp/hf.h"
  );

  test_subcase(
#include "data/h2o.def2-tzvp/hf.h"
  );

  test_subcase(
#include "data/hno3.def2-tzvp/hf.h"
  );

  test_subcase(
#include "data/kno3.def2-tzvp/hf.h"
  );

  // cc-pvdz

  test_subcase(
#include "data/h2.cc-pvdz/hf.h"
  );

  test_subcase(
#include "data/h2o.cc-pvdz/hf.h"
  );

  test_subcase(
#include "data/hno3.cc-pvdz/hf.h"
  );

}
