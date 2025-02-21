"""
Use bin_opt to find the features in a linear model that should be combined

We have feature (n, p) matrix X and target y. We have (n ** 2 - n) / 2 possible x pairs that could
be interacted, we impose constraint that number of interactions == p
"""

import numpy as np
from sklearn import linear_model, metrics
from bin_opt import *

def sim(n=1000, p=10, pairs=[(1, 2), (3, 4), (4, 5)]):
  X = np.random.normal(10, 1, (n, p))
  beta = np.random.normal(0, .5, p + len(pairs))
  y = 100 + X.dot(beta[:p]) + np.random.normal(0, 1.5, n)
  for i, pair in enumerate(pairs):
    y += X[:, pair[0]] * X[:, pair[1]] * beta[p + i]
  return X, y

def index_to_coord(d, idx):
  # find key given value
  for k, v in d.items():
    if v == idx:
      return k
  return None

def find_model(X, y, ninteract, niter=100):
  """
  Args:
    X: (n, p) feature matrix
    y: target vectior
    ninteract: number of allowed interactions
    niter: number of bin_opt iterations

  Returns:
    tuple of (best combo vector and R2, selected pairs)
  """
  n, p = X.shape
  # TODO: add train and test split
  # TODO: add classification
  combos = interact_dict(X.shape[1])
  print(len(combos), 'possible combos')
  x0 = np.zeros(len(combos))
  x0[:ninteract] = 1 # initialise
  def f(x0):
    reg = linear_model.LinearRegression()
    Xinteract = np.zeros((n, ninteract))
    # find the index where x0 = 1
    indices = np.where(x0 == 1)[0]
    for i, idx in enumerate(indices):
      coord = index_to_coord(combos, idx)
      Xinteract[:, i] = X[:, coord[0]] * X[:, coord[1]]
    Xall = np.hstack((X, Xinteract))
    reg.fit(Xall, y)
    r2 = metrics.r2_score(y, reg.predict(Xall))
    return -r2
  res = bin_opt(f, x0, ninteract, niter=niter)
  # translate the index back to coordinates
  pairs = [index_to_coord(combos, i) for i in np.where(res['x'] == 1)[0]] 
  return res, pairs


def interact_dict(n):
  m = 0
  d = {}
  for i in range(n):
    for j in range(i + 1, n):
      d[i, j] = m
      m += 1
  return d

def count_matching_pairs(pairs0, pairs1):
  return sum([1 for p0 in pairs0 if p0 in pairs1])

