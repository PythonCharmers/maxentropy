#!/usr/bin/env python

"""Tests for the maximum entropy package.

"""

import numpy as np

import maxentropy


def test_loaded_die_with_feature_expectations():
    """
    Unfair die example from Jaynes, Probability Theory: The Logic of Science, 2006
    
    Suppose you know that the long-run average number on the face of a 6-sided die
    tossed many times is 4.5.
    
    What probability p(x) would you assign to rolling x on the next roll?
    
    This code finds the probability distribution with maximal entropy
    subject to the single constraint:
    
    1.    E f(X) = 4.5
    
    where f(x) = x.
    """
    samplespace = np.arange(6) + 1
    
    def f0(x):
        return x
    
    
    features = [f0]
    
    # Now set the desired feature expectations
    target_expectations = [4.5]
    
    model = maxentropy.DiscreteMinDivergenceDensity(features, samplespace) # , algorithm='Nelder-Mead')

    # Fit the model
    model.fit_expectations(target_expectations)

    # How well are the constraints satisfied?
    assert np.allclose(target_expectations, model.feature_expectations())

    # Manually test if the constraints are satisfied:
    p = model.probdist()
    mean = np.sum(p * samplespace)  # E(X) = \sum_{j=1,...,n} p(x_j) x_j
    assert np.isclose(mean, target_expectations[0])

    # Output the distribution
    print("\nFitted model parameters are:\n" + str(model.params))
    print("\nFitted distribution is:")

    for j, x in enumerate(model.samplespace):
        print(f"\tx = {x}: p(x) = {p[j]:0.3f}")

    # Now show how well the constraints are satisfied:
    print()
    print("Desired constraints:")
    print("\tE f(X)   = [4.5]")
    print()
    print("Actual expectations under the fitted model:")
    print("\t\\hat{X} = ", model.expectations())


def test_loaded_die_with_data():
    r"""
    Unfair die example with data---adapted from Jaynes, Probability Theory:
    The Logic of Science, 2006
    
    Suppose we have data with many rolls of an unfair 6-sided die. We observe
    the long-run average (which happens to be 4.5).

    What probability p(x) would you assign to rolling x on the next roll?
    
    This code finds the probability distribution with maximal entropy
    subject to the single constraint on the model, that the expectation
    
        E f(X) = 1 / n . \sum_{k=1}^n f(x_k)     over the data samples x_k
    
    where f(x) = x

    We don't use any other features of the data, just this
    one observation as a constraint.
    """
    samplespace = np.arange(6) + 1
    
    # One moment constraint on E(X):

    def f0(x):
        return x
    
    features = [f0]
 
    model = maxentropy.DiscreteMinDivergenceDensity(features, samplespace) # , algorithm='Nelder-Mead')

    # Simulated rolls of an unfair die with mean 4.5:
    rolls = np.random.randint(low=3, high=7, size=1000)

    # Make 2d with 1 column (1 feature):
    X = rolls[:, None]

    # Fit the model to constraints given by features and their empirical
    # frequencies in the data:
    model.fit(X)

    # How well are the constraints satisfied?
    empirical_expectations = np.array([f(X).mean() for f in features])
    assert np.allclose(empirical_expectations, model.feature_expectations())

    # Manually test if the constraints are satisfied:
    p = model.probdist()
    mean = np.sum(p * samplespace)  # E(X) = \sum_{j=1,...,n} p(x_j) x_j
    assert np.isclose(mean, X.mean(axis=0))

    # Output the distribution
    print("\nFitted model parameters are:\n" + str(model.params))
    print("\nFitted distribution is:")

    for j, x in enumerate(model.samplespace):
        print(f"\tx = {x}: p(x) = {p[j]:0.3f}")

    # Now show how well the constraints are satisfied:
    print()
    print("Desired constraints:")
    print("\tE f(X)   = [4.5]")
    print()
    print("Actual expectations under the fitted model:")
    print("\t\\hat{X} = ", model.expectations())



    

