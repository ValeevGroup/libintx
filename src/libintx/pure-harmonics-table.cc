#include "libintx/pure.h"
#include "libintx/orbital.h"
#include <cstdio>
#include <cstdlib>

using namespace libintx;

void pure3(int x, int a, int b) {
  for (int mb = -b; mb <= b; ++mb) {
    for (int ma = -a; ma <= a; ++ma) {
      for (int mx = -x; mx <= x; ++mx) {
        printf("f(Pure{%i,%i}, Pure{%i,%i}, Pure{%i,%i}, ", x,mx, a,ma, b,mb);
        for (auto [ib,jb,kb] : cartesian::shell(b)) {
          for (auto [ia,ja,ka] : cartesian::shell(a)) {
            for (auto [ix,jx,kx] : cartesian::shell(x)) {
              double c = 1;
              c *= pure::coefficient(x,mx,ix,jx,kx);
              c *= pure::coefficient(a,ma,ia,ja,ka);
              c *= pure::coefficient(b,mb,ib,jb,kb);
              if (!c) continue;
              printf(
                "%f*c(Cart{%i,%i,%i},Cart{%i,%i,%i},Cart{%i,%i,%i}) + ",
                c, ix,jx,kx, ia,ja,ka, ib,jb,kb
              );
            }
          }
        }
        printf(";\n");
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc == 4) {
    pure3(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
  }
}
