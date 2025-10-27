#!/usr/bin/env python

import sys
sys.path.insert(0, "../../../python")

assert(sys.argv[1])
path = sys.argv[1]

import pywfn.gto

with open(path) as fh:
  basis = pywfn.gto.parse(fh.read(), normalize=False)
  for (k,v) in basis.items():
    print("{ %i, {" % k)
    for (l,gs) in v:
      gs = [ "{ %.16e, %.16e }" % (e,c) for (e,c) in gs ]
      print("  { %i, { %s } }," % (l, ", ".join(gs)))
    print("} },")
