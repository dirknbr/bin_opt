
import unittest
from lin_model_interact import *

class TestLinModelInteract(unittest.TestCase):

  def test_end_to_end(self):
    pairs0 = [(1, 2), (3, 4), (6, 7)]
    np.random.seed(33)
    X, y = sim(n=1000, p=10, pairs=pairs0)
    res, pairs1 = find_model(X, y, 5)
    pairs_matched = count_matching_pairs(pairs0, pairs1)
    # test if at least 2 of the 3 pairs are found
    self.assertTrue(pairs_matched >= 2)
    self.assertTrue(res['value'] < -.9)


if __name__ == '__main__':
  unittest.main()
