#!/usr/bin/env python

"""Test functions for maximum entropy module.

Author: Ed Schofield, 2003-2005
Copyright: Ed Schofield, 2003-2005
"""

from numpy.testing import assert_allclose
import numpy as np
import pytest
import scipy.stats
from scipy.special import logsumexp
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import toolz as tz

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
    Fit some discrete models and ensure that the entropy its Lagrangian dual are
    equal (after fitting, when the constraints are satisfied) and both these
    equal the result produced by scipy.stats.entropy().
    """
    samplespace = np.arange(6) + 1

    def f0(x):
        return x

    features = [f0]

    # Now set the desired feature expectations
    target_expectations = [4.5]

    # X = np.atleast_2d(target_expectations)
    model = maxentropy.DiscreteMinDivergenceDensity(features, samplespace)

    # Fit the model
    model.fit_expectations(target_expectations)

    H = model.entropy()
    assert_allclose(H, model.entropydual())
    assert_allclose(H, scipy.stats.entropy(model.probdist()))


def test_kl_div_loaded_die():
    """
    Fit some discrete models with priors and ensure that the KL divergence is
    equal to that computed by scipy.stats.entropy(px, qx).
    """
    samplespace = np.arange(6) + 1

    def f0(x):
        return x in samplespace

    uniform_model = maxentropy.DiscreteMinDivergenceDensity(
        [f0], samplespace, vectorized=False
    )
    uniform_model.fit_expectations([1.0])

    def f1(x):
        return x

    features = [f0, f1]

    # Now set the desired feature expectations
    target_expectations = [1.0, 4.5]

    # X = np.atleast_2d(target_expectations)
    model = maxentropy.DiscreteMinDivergenceDensity(
        features,
        samplespace,
        vectorized=False,
        prior_log_pdf=uniform_model.predict_log_proba,
    )

    # Fit the model
    model.fit_expectations(target_expectations)

    KL = model.kl_divergence()
    assert KL >= 0
    assert_allclose(KL, scipy.stats.entropy(model.probdist(), uniform_model.probdist()))


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
        features, xs, vectorized=True, array_format="ndarray"
    )
    assert isinstance(F, np.ndarray)
    assert F.shape == (len(xs), len(features))

    # Test csc sparse:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=True, array_format="csc_array"
    )
    assert scipy.sparse.issparse(F)
    assert isinstance(F, scipy.sparse.csc_array)
    assert F.shape == (len(xs), len(features))

    # Test csr sparse:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=True, array_format="csr_array"
    )
    assert scipy.sparse.issparse(F)
    assert isinstance(F, scipy.sparse.csr_array)
    assert F.shape == (len(xs), len(features))

    # Test that it still works if your functions are vectorized but you pass
    # vectorized=False:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, array_format="ndarray"
    )
    assert isinstance(F, np.ndarray)
    assert F.shape == (len(xs), len(features))

    # Test that the dtype argument works:
    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, array_format="ndarray", dtype=np.float32
    )
    assert F.dtype == np.float32

    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, array_format="ndarray", dtype=np.float64
    )
    assert F.dtype == np.float64

    F = utils.evaluate_feature_matrix(
        features, xs, vectorized=False, array_format="ndarray", dtype=np.int64
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

    model = maxentropy.DiscreteMinDivergenceDensity(
        features, samplespace, vectorized=False, verbose=False, algorithm=algorithm
    )

    # Fit the model
    model.fit_expectations(target_expectations)

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

    model = maxentropy.MinDivergenceDensity(
        features,
        auxiliary_sampler,
        vectorized=False,
        verbose=False,
        algorithm="CG",
    )

    model.fit_expectations(target_expectations)

    # How well does the model estimate that the constraints satisfied?
    assert_allclose(target_expectations, model.feature_expectations())

    # Manually test if the constraints are satisfied:
    F = model.features(samplespace)
    # p = model.pdf_from_features(F)
    p = model.predict_proba(samplespace)

    assert_allclose(p.sum(), target_expectations[0], atol=1e-2)
    assert_allclose(p[0] + p[1], target_expectations[1], atol=1e-2)
    assert_allclose(p[0] + p[2], target_expectations[2], atol=1e-2)


def test_classifier():
    iris = load_iris()

    iris.keys()

    X = iris.data
    y = iris.target

    def f0(X):
        return X[:, 0] ** 2

    def f1(X):
        return X[:, 1] ** 2

    def f2(X):
        return X[:, 2] ** 2

    def f3(X):
        return X[:, 3] ** 2

    def f4(X):
        """
        Petal length * petal width
        """
        return X[:, 1] * X[:, 2]

    features = [f0, f1, f2, f3, f4]

    stretched_minima, stretched_maxima = utils.bounds_stretched(
        X, stretch_factor=1.0
    )  # i.e. 100%
    uniform_dist = scipy.stats.uniform(
        stretched_minima, stretched_maxima - stretched_minima
    )
    sampler = utils.auxiliary_sampler_scipy(
        uniform_dist, n_dims=len(iris["feature_names"]), n_samples=10_000
    )
    clf = maxentropy.MinDivergenceClassifier(
        feature_functions=features, auxiliary_sampler=sampler, verbose=True
    )
    # For added fun, we test whether `predict` etc. can handle labels that don't start at 0 and non-consecutive labels:
    target_mapping = np.array(
        [3, 6, 9]
    )  # i.e. map iris target class 0 to 3, class 1 to 6, class 2 to 9
    y = target_mapping[y]
    clf.fit(X, y)
    log_proba = clf.predict_log_proba(X)
    proba = clf.predict_proba(X)
    assert_allclose(np.log(proba), log_proba)
    pred = clf.predict(X)
    assert sorted(set(pred)) == sorted(target_mapping)
    assert clf.score(X, y) > 0.9


@pytest.mark.xfail(reason="need to figure out this test!")
def test_current_api_fixme():

    cancer = load_breast_cancer(as_frame=True)

    df_cancer = cancer["data"]
    X_cancer = df_cancer.values
    y_cancer = cancer["target"]

    def non_neg(x):
        return x >= 0

    auxiliary = scipy.stats.uniform(-0.2, 1.2)  # i.e. from -0.2 to 1.0

    sampler = maxentropy.utils.auxiliary_sampler_scipy(auxiliary, n_samples=10_000)

    model = maxentropy.MinDivergenceDensity(
        [non_neg],
        sampler,
        prior_log_pdf=prior_model.logpdf,
        array_format="ndarray",
    )

    k = model.features(np.array([df_cancer["mean concavity"].mean()]))

    model.fit_expectations(k)
    print(f"Log likelihood of original model: {model.predict_log_proba(X_cancer)}")


# def test_old_idea_of_an_ideal_api():
#     from sklearn.datasets import load_breast_cancer
#
#     cancer = load_breast_cancer(as_frame=True)
#     df_cancer = cancer["data"]
#     X_cancer = df_cancer.values
#     y_cancer = cancer["target"]
#
#     feature = "mean concavity"
#
#     # We constrain all the values to be non-negative
#     feature_functions = [non_neg] * X_cancer.shape[1]
#
#     # This looks nice, but the API would be a mess with different samplers. It's
#     # inflexible and messy. Much cleaner to just pass in a single iterator as
#     # the sampler:
#     model = maxentropy.MinDivergenceDensity(
#         sampler="uniform",
#         array_format="ndarray",
#         sampling_stretch_factor=0.1,
#         n_samples=10_000,
#     )
#     model.fit(X_cancer, feature_functions=feature_functions)


def test_classifier():

    wine = load_wine()

    X_wine = wine["data"]
    y_wine = wine["target"]

    net = MLPClassifier(
        hidden_layer_sizes=(100,),
        learning_rate_init=0.01,
        max_iter=1000,
        random_state=7,
    )
    net.fit(X_wine, y_wine)
    print(f"Score of MLPClassifier = {net.score(X_wine, y_wine)}")

    # None of the values in the wine data can be negative, so define constraints
    # on these feature functions:

    @tz.curry
    def non_neg(column, x):
        return x[:, column] >= 0

    feature_functions = [non_neg(i) for i in range(len(wine["feature_names"]))]

    y_freq = np.bincount(y_wine)
    y_freq = y_freq / np.sum(y_freq)

    minima = X_wine.min(axis=0) - 10 * X_wine.std(axis=0)
    maxima = X_wine.max(axis=0) + 10 * X_wine.std(axis=0)
    sampler = utils.make_uniform_sampler(minima, maxima, n_samples=100_000)

    clf = maxentropy.MinDivergenceClassifier(
        feature_functions,
        auxiliary_sampler=sampler,
        prior_clf=net,
        prior_class_probs=y_freq,
        # prior_log_proba_fn=lambda xs: forward_pass_centered(net, slice(None), xs),
        array_format="ndarray",
        vectorized=True,
    )
    clf.fit(X_wine, y_wine)
    print(f"Score of the MinKLClassifier: {clf.score(X_wine, y_wine)}")
