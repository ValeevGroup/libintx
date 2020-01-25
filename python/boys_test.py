import unittest
import boys

class BoysTest(unittest.TestCase):

  def test_chebyshev(self):
    chebyshev = boys.chebyshev()
    reference = boys.reference()
    for x in [ float(x)/5 for x in range(120*5) ]:
      self.assertAlmostEqual(chebyshev.compute(x, 0), reference.compute(x,0), places=15)

if __name__ == '__main__':
  unittest.main()
