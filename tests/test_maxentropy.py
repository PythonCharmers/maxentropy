#!/usr/bin/env python

""" Test functions for maximum entropy module.

Author: Ed Schofield, 2003-2005
Copyright: Ed Schofield, 2003-2005
"""

from numpy.testing import assert_allclose
import numpy as np
import scipy.stats
from scipy.special import logsumexp

import maxentropy
import maxentropy.utils as utils


def test_logsumexp():
    """
    Test whether logsumexp() correctly handles large
    inputs.
    """

    a = np.arange(200)
    desired = np.log(np.sum(np.exp(a)))
    assert_allclose(logsumexp(a), desired)

    # Now test with large numbers
    b = [1000, 1000]
    desired = 1000.0 + np.log(2.0)
    assert_allclose(logsumexp(b), desired)

    n = 1000
    b = np.ones(n) * 10000
    desired = 10000.0 + np.log(n)
    assert_allclose(logsumexp(b), desired)


def test_entropy_loaded_die():
    """
    Fit some discrete models and ensure that the entropy and entropy are equal
    (when the constraints are satisfied after fitting) and both these equal the
    result produced by scipy.stats.entropy().
    """
    samplespace = np.arange(6) + 1

    def f0(x):
        return x

    features = [f0]

    # Now set the desired feature expectations
    target_expectations = [4.5]

    # X = np.atleast_2d(target_expectations)
    model = maxentropy.MinDivergenceModel(features, samplespace)

    # Fit the model
    model.fit(target_expectations)

    H = model.entropy()
    assert_allclose(H, model.entropydual())
    assert_allclose(H, scipy.stats.entropy(model.probdist()))


def test_evaluate_feature_matrix():
    # Define 3 functions, vectorize them,
    # and use them with a sampler of continuous things

    def f0(x):
        return x

    def f1(x):
        return x**2

    lower, upper = (-3, 3)

    def f2(x):
        return (lower < x) & (x < upper)

    features = [f0, f1, f2]

    auxiliary = scipy.stats.norm(loc=0.0, scale=2.0)
    sampler = utils.auxiliary_sampler_scipy(auxiliary)
    xs, _ = next(sampler)

    # Test dense ndarray:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=True, matrix_format="ndarray"
    )
    assert isinstance(F, np.ndarray)
    assert F.shape == (len(xs), len(features))

    # Test csc sparse:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=True, matrix_format="csc_matrix"
    )
    assert scipy.sparse.issparse(F)
    assert scipy.sparse.isspmatrix_csc(F)
    assert F.shape == (len(xs), len(features))

    # Test csr sparse:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=True, matrix_format="csr_matrix"
    )
    assert scipy.sparse.issparse(F)
    assert scipy.sparse.isspmatrix_csr(F)
    assert F.shape == (len(xs), len(features))

    # Test that it still works if your functions are vectorized but you pass
    # vectorized=False:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, matrix_format="ndarray"
    )
    assert isinstance(F, np.ndarray)
    assert F.shape == (len(xs), len(features))

    # Test that the dtype argument works:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, matrix_format="ndarray", dtype=np.float32
    )
    assert F.dtype == np.float32

    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, matrix_format="ndarray", dtype=np.float64
    )
    assert F.dtype == np.float64

    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, matrix_format="ndarray", dtype=np.int64
    )
    assert F.dtype == np.int64


"""
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


def test_berger(algorithm="CG"):
    def f0(x):
        return x in samplespace

    def f1(x):
        return x == "dans" or x == "en"

    def f2(x):
        return x == "dans" or x == "à"

    features = [f0, f1, f2]
    samplespace = ["dans", "en", "à", "au cours de", "pendant"]

    # Now set the desired feature expectations
    target_expectations = [1.0, 0.3, 0.5]

    model = maxentropy.MinDivergenceModel(
        features, samplespace, vectorized=False, verbose=False, algorithm=algorithm
    )

    # Fit the model
    model.fit(target_expectations)

    # How well are the constraints satisfied?
    assert_allclose(target_expectations, model.expectations())

    # Manually test if the constraints are satisfied:
    p = model.probdist()
    assert_allclose(p.sum(), target_expectations[0])
    assert_allclose(p[0] + p[1], target_expectations[1])
    assert_allclose(p[0] + p[2], target_expectations[2])

    # Output the distribution
    print("\nFitted model parameters are:\n" + str(model.params))
    print("\nFitted distribution is:")
    for j, x in enumerate(model.samplespace):
        print(f"\tx = {x:15s}: p(x) = {p[j]:.4f}")

    # Now show how well the constraints are satisfied:
    print()
    print("Desired constraints:")
    print("\tp['dans'] + p['en'] = 0.3")
    print("\tp['dans'] + p['à']  = 0.5")
    print()
    print("Actual expectations under the fitted model:")
    print("\tp['dans'] + p['en'] =", p[0] + p[1])
    print("\tp['dans'] + p['à']  =", p[0] + p[2])


def test_dictsampler():
    def f0(x):
        return x in samplespace

    def f1(x):
        return x == "dans" or x == "en"

    def f2(x):
        return x == "dans" or x == "à"

    features = [f0, f1, f2]
    samplespace = ["dans", "en", "à", "au cours de", "pendant"]

    # Now set the desired feature expectations
    target_expectations = [1.0, 0.3, 0.5]

    # Define a uniform instrumental distribution for sampling.
    # This can be unnormalized.
    samplefreq = {e: 1 for e in samplespace}

    n = 10**5

    # Now create a generator that will be used for importance sampling.
    # When called with no arguments it should return a tuple
    # (xs, log_q_xs) representing:

    #     xs: a sample x_1,...,x_n to use for importance sampling
    #
    #     log_q_xs: an array of length n containing the (natural) log
    #               probability density (pdf or pmf) of each point under the
    #               auxiliary sampling distribution.

    auxiliary_sampler = utils.dictsampler(samplefreq, size=n)

    model = maxentropy.MCMinDivergenceModel(
        features,
        auxiliary_sampler,
        vectorized=False,
        verbose=False,
        algorithm="CG",
    )

    model.fit(target_expectations)

    # How well does the model estimate that the constraints satisfied?
    assert_allclose(target_expectations, model.feature_expectations())

    # Manually test if the constraints are satisfied:
    F = model.features(samplespace)
    p = model.pdf_from_features(F)

    assert_allclose(p.sum(), target_expectations[0], atol=1e-2)
    assert_allclose(p[0] + p[1], target_expectations[1], atol=1e-2)
    assert_allclose(p[0] + p[2], target_expectations[2], atol=1e-2)
