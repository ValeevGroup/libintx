import urllib.request

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
      alpha2 = 2*alpha;
      N = sqrt(NL**2*alpha2**(L+1.5))
      pn.append((alpha,N*C))
    # normalise to unity
    # compute the self-overlap, scale coefficients by its inverse square root
    overlap = 0.0
    for i,(ai,Ci) in enumerate(pn):
      for j,(aj,Cj) in enumerate(pn):
        gamma = ai + aj;
        overlap += (
          (factorial2_Kminus1[2*L] * sqrt_Pi_cubed * Ci*Cj) /
          (2**L * (ai+aj)**(L+1.5))
        )
    pn = [ (a,C/sqrt(overlap)) for (a,C) in pn ]
    return Gaussian(L, pn)

def parse(basis, format="json", normalize=True, keep_zeros=False):
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
      if len(angular_momentum) == 1:
        angular_momentum *= len(coefficients)
      for (L,cs) in zip(angular_momentum,coefficients):
        primitives = [ (e,c) for (e,c) in zip(exponents, cs) if c != 0.0 or keep_zeros ]
        # print (coefficients[i])
        # print (primitives)
        # print()
        g = Gaussian(L, primitives)
        if normalize: g = g.normalized
        basis[Z].append(g)
        # print("Gaussian L=%i,%s normalized -> %s" % (L,primitives,basis[Z][-1]))
  # print ("// { Z, { L, { { alpha, coeff }, ... } } }")
  # for z in basis.keys():
  #   print ("{",z,",\n  {")
  #   for (l,g) in basis[z]:
  #     print("   { %i, { %s } }," % (l,", ".join(["{ %17.12f, %17.12f }" % (a,c) for (a,c) in g])))
  #   print("  }\n},")
  return basis

def basis(name, url=None, format="json", keep_zeros=False):
  data = None
  if not url:
    from . import resources
    url = "file://" + str(resources.file("lib/gto", ("%s.%s" % (name,format)).lower()))
  with urllib.request.urlopen(url) as fh:
    data = fh.read().decode()
    #print(data)
    return parse(data,format,keep_zeros)
