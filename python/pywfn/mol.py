elements = (
  "",
  "H", "He",
  "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
  "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr"
  )

bohr_to_angstrom = 0.529177210903

def parse(mol,format="xyz"):
  lines = mol.splitlines()
  nxyz = int(lines[0])
  atoms = []
  for line in lines[2:2+nxyz]:
    [a,x,y,z] = line.split()
    Z = elements.index(a.capitalize())
    r = (float(r)*(1/bohr_to_angstrom) for r in (x,y,z))
    atoms.append((a, Z, tuple(r)))
  return atoms

def load(name, file=None, format="xyz"):
  text = None
  if not file:
    from . import resources
    file = resources.file("mol", ("%s.%s" % (name,format)).lower())
  with open(file) as f:
    return parse(f.read(),format)

class Library():
  def __init__(self):
    pass
  def __getattr__(self,name):
    return load(name)

library = Library()
