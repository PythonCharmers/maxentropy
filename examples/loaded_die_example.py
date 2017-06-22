#!/usr/bin/env python

""" Example use of the maximum entropy module:

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

f = [f0]

model = maxentropy.Model(f, samplespace)

# Now set the desired feature expectations
K = [4.5]

model.verbose = True

# Fit the model
model.fit(K)

# Output the distribution
print("\nFitted model parameters are:\n" + str(model.params))
print("\nFitted distribution is:")
p = model.probdist()
print(type(p))
print(p.dtype)
print(p[0])
print(p[1])
print(model.samplespace)

for j, x in enumerate(model.samplespace):
    print("\tx = {0:}: p(x) = {1:0.3f}".format(x, p[j]))


# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tE(X) = 4.5")
print()
print("Actual expectations under the fitted model:")
print("\t\hat{X} = ", model.expectations())
