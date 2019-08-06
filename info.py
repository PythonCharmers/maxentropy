"""
================================================
Maximum entropy models (:mod:`maxentropy`)
================================================

.. currentmodule:: maxentropy

Package content
===============

Models:

.. autosummary::
   :toctree: generated/

   Model
   BigModel
   BaseModel
   ConditionalModel

Utilities:

.. autosummary::
   :toctree: generated/

   arrayexp
   arrayexpcomplex
   columnmeans
   columnvariances
   densefeaturematrix
   densefeatures
   dotprod
   flatten
   innerprod
   innerprodtranspose
   logsumexp
   logsumexp_naive
   robustlog
   rowmeans
   sample_wr
   sparsefeaturematrix
   sparsefeatures


Usage information
=================

Contains two classes for fitting maximum entropy models and minimum
KL-divergence models (also known as "exponential family" models) subject
to linear constraints on the expectations of arbitrary feature
statistics.  One class, "Model", is for small discrete sample spaces,
using explicit summation. The other, "BigModel", is for sample spaces
that are either continuous (and perhaps high-dimensional) or discrete but
too large to sum over, and uses importance sampling.

The resulting maximum entropy model always has exponential form

..
   p(x) = exp(theta^T f(x)) / Z(theta)

.. math::
   p\\left(x\\right)=\\exp\\left(\\frac{\\theta^{T}f\\left(x\\right)}
                                  {Z\\left(\\theta\\right)}\\right)

with a real parameter vector theta of the same length as the feature
statistic f(x), For more background, see, for example, Cover and
Thomas (1991), *Elements of Information Theory*.

See the file examples/berger_example.py for a simple walk-through of how to use
these routines when the sample space is small enough to be enumerated.

See examples/berger_example_simulated.py for a a similar walk-through using
simulation.

"""

# Copyright: Ed Schofield, 2003-2019
# License: BSD-style (see LICENSE.txt in main source directory)

