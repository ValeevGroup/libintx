@staticmethod
def nbf(L):
  return 2*L+1

class Gaussian(tuple):
  def __new__(self, L, *args):
    return tuple.__new__(Gaussian, (L, *args))
  @property
  def L(self): return self[0]
  @property
  def primitives(self): return self[1]
  @property
  def nbf(self):
    return (2*self.L+1)
  @property
  def normalized(self):
    L = self.L
    sqrt_Pi_cubed = float(5.56832799683170784528481798212)
    # df_Kminus1[k] = (k-1)!!
    factorial2_Kminus1 = (1, 1, 1, 2, 3, 8, 15, 48, 105, 384, 945, 3840, 10395)
    assert(2*L <= len(factorial2_Kminus1))
    pn = []
    from math import sqrt
    NL = 2**L / (sqrt_Pi_cubed * factorial2_Kminus1[2*L])
    for (alpha,C) in self.primitives:
      if (alpha == 0): continue
      assert(alpha > 0)
      two_alpha = 2*alpha;
      two_alpha_to_am32 = two_alpha**(L+1)*sqrt(two_alpha)
      N = sqrt(two_alpha_to_am32*NL)
      pn.append((alpha,N*C))
    return Gaussian(L, pn)

def parse(basis, format="json"):
  if isinstance(basis, str):
    from json import loads as load
    basis = load(basis)
  elif hasattr(basis, "read"):
    from json import load as load
    basis = load(basis)
  else:
    pass
  elements = basis['elements']
  basis = {}
  for Z in map(int,elements):
    basis[Z] = []
    # print(Z)
    for f in (elements[str(Z)]['electron_shells']):
      angular_momentum = f['angular_momentum']
      exponents = list(map(float, f['exponents']))
      coefficients = [list(map(float,c)) for c in f['coefficients']]
      for i,L in enumerate(angular_momentum):
        primitives = list(zip(exponents, coefficients[i]))
        basis[Z].append(Gaussian(L, primitives).normalized)
  return basis

def basis(name, file=None, format="json"):
  data = None
  if not file:
    from . import resources
    file = resources.file("gto", ("%s.%s" % (name,format)).lower())
  with open(file) as fh:
    return parse(fh.read(),format)
