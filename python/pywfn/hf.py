import warnings
warnings.simplefilter("ignore", UserWarning)

import libintx
import itertools
from timeit import default_timer as timer

#import scipy.linalg
import numpy as np

class HF:

  def __init__(self, wfn, precision=1e-10, num_threads = 0, smin=1e-12):
    self._wfn = wfn

    if not num_threads:
      import pywfn
      num_threads = pywfn.get_num_threads()

    from . import gto
    basis = [ gto.Gaussian(*g,r) for (r,s) in self._wfn.basis for g in s ]

    atoms = wfn.atoms
    self._ehf = 0
    self._enuc = np.float128(0)
    for i,(qi,ri) in enumerate(atoms):
      for j,(qj,rj) in enumerate(atoms[i+1:]):
      #for j,(qj,rj) in enumerate(atoms[:i]):
        self._enuc += qi*qj/np.linalg.norm(np.array(ri)-np.array(rj), ord=2)

    print("# threads:", num_threads)
    print("#atoms = %i" % len(atoms))
    print("#shells = %i" % len(basis))
    print("#N(ao) = %i" % self._wfn.nbf)
    print("#N(e) = %i" % self._wfn.nelectrons)
    print("#E(nuclear) = %.16f" % self._enuc)

    print("* Initializing HF.engine")
    engine = libintx.hf.fock_engine(wfn.atoms, basis, precision=precision)
    engine.num_threads = num_threads
    print("* ao screening ...", end='')
    t0 = timer()
    [n_pairs,n_total] = engine.init2(smin)
    print(" %.2f s" % (timer()-t0))
    print(
      "  %i pairs > %e, %.2f%% sparse" % (
        n_pairs, precision,
        (1-float(n_pairs)/n_total)*100
      )
    )

    self._engine = engine
    self._H = None
    self._X = None
    self._D = None
    self._F = None

  @property
  def nbf(self):
    return self._wfn.nbf

  @property
  def enuc(self):
    #return 0
    return self._enuc

  @property
  def engine(self):
    return self._engine;

  @property
  def S(self):
    return self.compute1("S")

  @property
  def T(self):
    return self.compute1("T")

  @property
  def V(self):
    return self.compute1("V")

  @property
  def H(self):
    if self._H is None:
      t0 = timer()
      print("* HF.H ...", end='')
      nbf = self.nbf
      H = np.zeros([nbf,nbf], order="f")
      self.engine.T(H.ctypes.data, nbf)
      self.engine.V(H.ctypes.data, nbf)
      self._H = H
      print(" %.2f s" % (timer()-t0))
    return self._H

  @property
  def X(self):
    if self._X is None:
      t0 = timer()
      print("* HF.X ...", end='')
      nbf = self.nbf
      X = np.zeros([nbf,nbf], order="f")
      (nx,snorm,xnorm) = self.engine.X(X.ctypes.data, nbf)
      X.resize([nbf,nx])
      self._X = X
      print(" %.2f s" % (timer()-t0))
    return self._X

  @property
  def D(self):
    if self._D is None:
      F = np.copy(self.H)
      t0 = timer()
      print("* HF.D.initial_guess ...", end='')
      self.engine.initial_guess(F.ctypes.data, F.shape[0], 1e-16)
      print(" %.2f s" % (timer()-t0))
      self._D = np.ndarray([self.nbf,self.nbf], order="f")
      self.compute_density(F,self._D)
    return self._D

  def compute1(self, op):
    nbf = self.nbf
    A = np.zeros([nbf,nbf])
    getattr(self.engine, op)(A.ctypes.data, nbf)
    #A[np.abs(A) < 1e-12] = 0
    return A

  def compute_density(self, F, D):
    X = self.X
    print("* HF.compute_density ...", end='')
    t0 = timer()
    self.engine.D(
      F.ctypes.data, F.shape[0],
      X.ctypes.data, X.shape[0], X.shape[1],
      D.ctypes.data, D.shape[0],
      0, 0 # C
    )
    print(" %.2f s" % (timer()-t0))
    return D
