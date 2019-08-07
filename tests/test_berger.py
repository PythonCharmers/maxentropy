# -*- coding: utf-8 -*-
"""Tests for the maxentropy package:

    Machine translation example -- English to French -- from the paper 'A
    maximum entropy approach to natural language processing' by Berger et
    al., 1996.

    Consider the translation of the English word 'in' into French.  We
    notice in a corpus of parallel texts the following facts:

        (1)    p(dans) + p(en) + p(a) + p(au cours de) + p(pendant) = 1
        (2)    p(dans) + p(en) = 3/10
        (3)    p(dans) + p(a)  = 1/2

    This code finds the probability distribution with maximal entropy
    subject to these constraints.
"""

import numpy as np

import maxentropy
from maxentropy.utils import dictsampler


def f0(x):
    return x in samplespace

def f1(x):
    return x=='dans' or x=='en'

def f2(x):
    return x=='dans' or x=='à'

features = [f0, f1, f2]

samplespace = ['dans', 'en', 'à', 'au cours de', 'pendant']

# Now set the desired feature expectations
target_expectations = [1.0, 0.3, 0.5]

X = np.atleast_2d(target_expectations)


def test_berger(algorithm='CG'):

    model = maxentropy.MinDivergenceModel(features, samplespace,
                                          vectorized=False,
                                          verbose=False,
                                          algorithm=algorithm)


    # Fit the model
    model.fit(X)

    # How well are the constraints satisfied?
    assert np.allclose(X[0, :], model.expectations())

    # Manually test if the constraints are satisfied:
    p = model.probdist()
    assert np.isclose(p.sum(), target_expectations[0])
    assert np.isclose(p[0] + p[1], target_expectations[1])
    assert np.isclose(p[0] + p[2], target_expectations[2])


def test_berger_simulated(algorithm='CG'):
    # Define a uniform instrumental distribution for sampling.
    # This can be unnormalized.
    samplefreq = {e: 1 for e in samplespace}

    n = 10**5

    # Now create a function that will be used for importance sampling.
    # When called with no arguments it should return a tuple
    # (xs, log_q_xs) representing:

    #     xs: a sample x_1,...,x_n to use for importance sampling
    # 
    #     log_q_xs: an array of length n containing the (natural) log
    #               probability density (pdf or pmf) of each point under the
    #               auxiliary sampling distribution.

    auxiliary_sampler = dictsampler(samplefreq, size=n)

    model = maxentropy.MCMinDivergenceModel(features, auxiliary_sampler,
                                            vectorized=False,
                                            verbose=False,
                                            algorithm=algorithm)

    model.fit(X)

    # How well does the model estimate that the constraints satisfied?
    assert np.allclose(X[0, :], model.expectations())

    # Manually test if the constraints are satisfied:
    F = model.features(samplespace)
    p = model.pdf(F)

    assert np.isclose(p.sum(), target_expectations[0], atol=1e-2)
    assert np.isclose(p[0] + p[1], target_expectations[1], atol=1e-2)
    assert np.isclose(p[0] + p[2], target_expectations[2], atol=1e-2)

