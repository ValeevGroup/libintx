import unittest
import libintx
from pywfn import Wavefunction, mol
import cupy as cp

def make_basis(mol, basis):
  wfn = Wavefunction(mol, basis)
  basis = [(g,r) for (r,s) in wfn.basis for g in s]
  return basis

class GpuIntegralEngineTest(unittest.TestCase):

  def test_eri0(self):
    eri = libintx.gpu.eri(0, [], [])
    self.assertTrue(eri is None)

  def test_eri3(self):
    basis = make_basis(mol.library.water, '6-31g')
    nbf = sum(g.nbf for (g,r) in basis)
    basis = [ (g.L, g.primitives, r) for (g,r) in basis ]
    eri = libintx.gpu.eri(3, basis, basis, 0)
    dst = cp.ndarray([nbf,nbf**2])
    eri.compute([0], [(0,0)], dst.data.ptr, dst.shape)

  def test_eri4(self):
    basis = make_basis(mol.library.water, '6-31g')
    nbf = sum(g.nbf for (g,r) in basis)
    basis = [ (g.L, g.primitives, r) for (g,r) in basis ]
    eri = libintx.gpu.eri(4, basis, basis, 0)
    dst = cp.ndarray([nbf**2,nbf**2])
    eri.compute([(0,0)], [(0,0)], dst.data.ptr, dst.shape)

if __name__ == '__main__':
  unittest.main()
