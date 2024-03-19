#!/usr/bin/env python

"""Tests for the maximum entropy package:

    Unfair die example from Jaynes, Probability Theory: The Logic of Science, 2006

    Suppose you know that the long-run average number on the face of a 6-sided die
    tossed many times is 4.5.

    What probability p(x) would you assign to rolling x on the next roll?

    This code finds the probability distribution with maximal entropy
    subject to the single constraint:

    1.    E f(X) = 4.5

    where f(x) = x

"""
import numpy as np

import maxentropy


samplespace = np.arange(6) + 1

def f0(x):
    return x

features = [f0]

# Now set the desired feature expectations
target_expectations = [4.5]

X = np.atleast_2d(target_expectations)


def test_loaded_die():
    model = maxentropy.MinDivergenceModel(features, samplespace)

    # Fit the model
    model.fit(X)

    # How well are the constraints satisfied?
    assert np.allclose(X[0, :], model.expectations())

    # Manually test if the constraints are satisfied:
    p = model.probdist()
    mean = np.sum(p * samplespace)   # E(X) = \sum_{j=1,...,n} p(x_j) x_j
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
