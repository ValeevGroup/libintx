#!/usr/bin/env python

import sys
sys.path.insert(0, "../../../python")

assert(sys.argv[1])
path = sys.argv[1]

import pywfn.mol

with open(path) as fh:
  mol = pywfn.mol.parse(fh.read())
  for (a,Z,r) in mol:
    print("{ %i, { %.16e, %.16e, %.16e } }," % (Z, *r))
