#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "test.h"

#include "libintx/gpu/jengine.h"

TEST_CASE("gpu.jengine") {

  auto basis = std::get<0>(libintx::test::basis1(0, 1, 10));
  auto df_basis = std::get<0>(libintx::test::basis1(0, 1, 20));
  auto jengine = libintx::gpu::make_jengine(basis, df_basis, nullptr, nullptr);

}
