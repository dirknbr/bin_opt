
"""
Minimise f(x) s.t. sum(x) == n, where x is binary vector and n is integer
"""

import numpy as np

def bin_opt(f, x0, n, seed=1, niter=100):
  """
  Args:
    f: function to minimise
    x0: binary vector of 0s and 1s
    n: sum constraint on x
    seed: random seed
    niter: number of iterations

  Returns:
    dict of best value and x
  """
  assert sum(x0) == n
  assert set(x0) == set([0, 1])
  np.random.seed(seed)
  res = {'value': f(x0), 'x': x0}
  # use local search and then randomly flip one unit
  for _ in range(niter):
    x1 = 1 * res['x']
    idx0 = np.random.choice(np.where(x1 == 0)[0], 1)[0]
    idx1 = np.random.choice(np.where(x1 == 1)[0], 1)[0]
    x1[idx0] = 1 - x1[idx0]
    x1[idx1] = 1 - x1[idx1]
    if f(x1) < res['value']:
      res = {'value': f(x1), 'x': x1}
  return res

def run_many_bin_opt(f, x0, n, niter=100, ntries=100):
  """
  Args:
    f: function to minimise
    x0: binary vector of 0s and 1s
    n: sum constraint on x
    niter: number of iterations in each search
    ntries: number of outer search starts

  Returns:
    dict of best value and x
  """
  res = {'value': f(x0), 'x': x0}
  for i in range(ntries):
    # randomise x0 and also change seed
    np.random.shuffle(x0)
    res1 = bin_opt(f, x0, n, 1 + i, niter)
    if res1['value'] < res['value']:
      res = res1
  return res 
