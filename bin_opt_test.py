
import unittest
from bin_opt import *

class TestBinOpt(unittest.TestCase):

  def test_vector_len_six(self):
    def f(x):
      return sum([(x[i] + 1) ** i for i in range(len(x))])
    x0 = np.array([0, 0, 0, 1, 1, 1])
    res = bin_opt(f, x0, sum(x0), niter=1000)
    print(res)
    self.assertTrue(np.array_equal(res['x'], np.array([1, 1, 1, 0, 0, 0])))

  def test_vector_len_ten_many_opt(self):
    def f(x):
      return sum([(x[i] + 1) ** i for i in range(len(x))])
    x0 = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    res = run_many_bin_opt(f, x0, sum(x0), niter=100, ntries=10)
    print(res)
    self.assertEqual(sum(res['x'][:sum(x0)]), sum(x0))
    self.assertEqual(res['value'], 36)

if __name__ == '__main__':
  unittest.main()