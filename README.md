# maxentropy: Maximum entropy and minimum divergence models in Python

## History
This package previously lived in SciPy 
(http://scipy.org) as ``scipy.maxentropy`` from versions v0.5 to v0.10. It was under-maintained and removed
from SciPy v0.11. It is now being refactored to use ``scikit-learn``'s estimator interface.

## Background

This package helps you to follow the maximum entropy principle (or the closely related principle of minimum divergence)
to construct a probability distribution (Bayesian prior) from prior information that you encode as generalized moment constraints.

The maximum entropy principle has been shown [Cox 1982, Jaynes 2003] to be the unique consistent approach to
constructing a discrete probability distribution from prior information that is available as "testable information".

If the constraints have the form of linear moment constraints:

$$
E(f_1(X)) = k_1
...
E(f_m(X)) = k_m
$$

then the principle gives rise to a unique probability distribution of **exponential form**. Most well-known probability
distributions are special cases of maximum entropy distributions. This includes uniform, geometric, exponential, Pareto,
normal, von Mises, Cauchy, and others: see https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution.

## Quickstart example: constructing a prior subject to known constraints

See this notebook: https://github.com/PythonCharmers/maxentropy/blob/master/notebooks/Loaded%20die%20example%20-%20skmaxent.ipynb

