# Binary integer optimisation

Minimise f(x) s.t. sum(x) == n, where x is binary vector and n is integer

Method is local search.

## Application

We have a linear model y = f(X), we want to allow certain pairwise interactions between X
columns with a constraint on the number of pairs. We use bin_opt to find the model with best R2. 

