from . import gto

num_threads = None

def get_num_threads():
  if num_threads: return num_threads
  from os import cpu_count
  return cpu_count()

class Wavefunction:

  def __init__(self, mol, basis):
    basis = gto.basis(basis)
    self.basis = [ (r,basis[Z]) for (a,Z,r) in mol ]
    self._atoms =  [ (Z,r) for (a,Z,r) in mol ]
    #setattr(self.basis, "nbf", property(lambda x: x))

  @property
  def atoms(self):
    return self._atoms

  @property
  def nelectrons(self):
    return sum([z for z,r in self.atoms])

  @property
  def nbf(self):
    return sum([g.nbf for (r,shells) in self.basis for g in shells])
