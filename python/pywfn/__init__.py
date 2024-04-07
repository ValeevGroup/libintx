from . import gto

class Wavefunction:

  def __init__(self, mol, basis):
    basis = gto.basis(basis)
    self.basis = [ (r,basis[Z]) for (a,Z,r) in mol ]
    #setattr(self.basis, "nbf", property(lambda x: x))
