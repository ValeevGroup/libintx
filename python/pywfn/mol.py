import urllib.request

elements = (
  "",
  "H", "He",
  "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr"
  )

bohr_to_angstrom = 0.529177210903

def parse(mol, format="xyz", name=None):
  lines = mol.splitlines()
  nxyz = int(lines[0])
  class Atoms(list): pass
  atoms = Atoms()
  atoms.name = name
  for line in lines[2:2+nxyz]:
    [a,x,y,z] = line.split()
    Z = elements.index(a.capitalize())
    r = tuple(float(r)*(1/bohr_to_angstrom) for r in (x,y,z))
    atoms.append((a, Z, r))
  return atoms

def load(url, format="xyz", name=None):
  with urllib.request.urlopen(url) as fh:
    data = fh.read().decode()
    #print(data)
    return parse(data,format,name)

class Library():
  def __init__(self):
    pass
  def __getattr__(self,name):
    from . import resources
    url = "file://%s" % resources.file("lib/mol", ("%s.xyz" % name).lower())
    return load(url,name=name)

library = Library()
