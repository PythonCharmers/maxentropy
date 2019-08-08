# maxentropy: Maximum entropy and minimum divergence models in Python

## Purpose

This package helps you to construct a probability distribution
(Bayesian prior) from prior information that you encode as
generalized moment constraints.

You can use it to either:

1. find the flattest distribution that meets your constraints, using the
   maximum entropy principle (discrete distributions only)

2. or find the "closest" model to a given prior model (in a KL divergence
   sense) that also satisfies your additional constraints.

## Background

The maximum entropy principle has been shown [Cox 1982, Jaynes 2003] to be the unique consistent approach to
constructing a discrete probability distribution from prior information that is available as "testable information".

If the constraints have the form of linear moment constraints, then
the principle gives rise to a unique probability distribution of
**exponential form**. Most well-known probability distributions are
special cases of maximum entropy distributions. This includes
uniform, geometric, exponential, Pareto, normal, von Mises, Cauchy,
and others: see
[here](https://en.wikipedia.org/wiki/Maximum_entropy_probability_distribution).

## Examples: constructing a prior subject to known constraints

See the [notebooks folder](https://github.com/PythonCharmers/maxentropy/tree/master/notebooks).

### Quickstart guide
This is a good place to start: [Loaded die example (scikit-learn estimator API)](https://github.com/PythonCharmers/maxentropy/blob/master/notebooks/Loaded%20die%20example%20-%20skmaxent.ipynb)

## History
This package previously lived in SciPy 
(http://scipy.org) as ``scipy.maxentropy`` from versions v0.5 to v0.10.
It was under-maintained and removed from SciPy v0.11. It has since been
resurrected and refactored to use the scikit-learn Estimator inteface.

## Copyright
(c) Ed Schofield, 2003-2019
