#!/usr/bin/env python

import unittest
from pywfn import Wavefunction, mol
import pywfn
from pywfn.hf import HF
import numpy
import timeit

import numpy as np

#import matplotlib.pyplot as plot

num_threads = 4

# pywfn api

# # basis by name (load from pywfn/lib/gto)
# basis = '6-31g*'

# # or load basis from url.  NB need trim zeros, test better
# basis = pywfn.gto.basis(
#   "cc-pvdz",
#   "https://www.basissetexchange.org/api/basis/cc-pvdz/format/json/?version=1"
# )

# make your own molecule, it's but a list !
mol = [
  #['x', 2, [4,0,0]],
  #['x', 4, [2,0,0]],
  # ['x', 3, [-1,0,0]],
  # ['x', 3, [1,0,0]],
  # ['x', 3, [0,0,0]],
  # ['x', 6, [2,0,0]],
  ['x', 6, [3,0,0]],
]

# # load molecule from pywfn/lib/mol lib path
# mol = pywfn.mol.library.taxol

# # load molecule from url
# mol = pywfn.mol.load("https://raw.githubusercontent.com/evaleev/libint/refs/heads/master/tests/hartree-fock/h2o.xyz")


class FockEngineTest(unittest.TestCase):

  def test_hf(self):

    # load molecule from pywfn/mol lib path
    # mol = pywfn.mol.library.hno3
    # mol = pywfn.mol.library.taxol
    # mol = pywfn.mol.library.olestra
    # mol = pywfn.mol.library.gly120
    # mol = pywfn.mol.library.crambin
    # mol = pywfn.mol.library.ubiquitin

    # mols = [
    #   pywfn.mol.library.aspirin,
    #   pywfn.mol.library.taxol,
    #   pywfn.mol.library.olestra,
    #   pywfn.mol.library.gly120,
    #   pywfn.mol.library.crambin,
    #   pywfn.mol.library.ubiquitin,
    # ]

    mols = [
      pywfn.mol.library.water,
      #pywfn.mol.library.hno3,
      #pywfn.mol.library.aspirin,
      #pywfn.mol.library.taxol,
      #pywfn.mol.library.olestra,
      #pywfn.mol.library.gly30,
      #pywfn.mol.library.znglud,
      pywfn.mol.library.gly120,
      #pywfn.mol.library.at08,
      # pywfn.mol.library.crambin,
    ]

    basis_sets = [
      #"sto-6g",
      #"6-31g*",
      "def2-tzvp"
    ]

    precision = 1e-7

    pywfn.num_threads = 6

    for basis in basis_sets:
      for mol in mols:

        #basis = 'sto-3g'
        print ("\n***",mol.name,basis,precision)

        wfn = Wavefunction(mol, basis)

        hf = HF(wfn,precision=1e-10,num_threads=0)
        hf.D
        hf.X

if __name__ == '__main__':
  unittest.main()
