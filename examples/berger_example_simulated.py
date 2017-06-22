#!/usr/bin/env python

""" Example use of the maximum entropy module fit a model using
    simulation:

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

    This problem is small enough to solve analytically, but this code
    shows the steps one would take to fit a model on a continuous or
    large discrete sample space.
"""
from __future__ import print_function

import sys

import maxentropy
from maxentropy.maxentutils import dictsample, sampleFgen


try:
    algorithm = sys.argv[1]
except IndexError:
    algorithm = 'CG'
else:
    assert algorithm in ['CG', 'BFGS', 'LBFGSB', 'Powell', 'Nelder-Mead']

a_grave = u'\u00e0'

samplespace = ['dans', 'en', a_grave, 'au cours de', 'pendant']

def f0(x):
    return x in samplespace

def f1(x):
    return x == 'dans' or x == 'en'

def f2(x):
    return x == 'dans' or x == a_grave

f = [f0, f1, f2]

model = maxentropy.BigModel()

# Now set the desired feature expectations
K = [1.0, 0.3, 0.5]

# Define a uniform instrumental distribution for sampling
samplefreq = {e: 1 for e in samplespace}

n = 10**4
m = 3

# Now create a generator of features of random points and their logprobs

print("Generating an initial sample ...")
model.setsampleFgen(sampleFgen(samplefreq, f, n))

model.verbose = True

# Fit the model
model.avegtol = 1e-4
model.fit(K, algorithm=algorithm)

# Output the true distribution
print("\nFitted model parameters are:\n" + str(model.params))
smallmodel = maxentropy.Model(f, samplespace)
smallmodel.setparams(model.params)
print("\nFitted distribution is:")
p = smallmodel.probdist()
for j, x in enumerate(smallmodel.samplespace):
    x = smallmodel.samplespace[j]
    print("\tx = %-15s" %(x + ":",) + " p(x) = "+str(p[j]))


# Now show how well the constraints are satisfied:
print()
print("Desired constraints:")
print("\tp['dans'] + p['en'] = 0.3")
print("\tp['dans'] + p['" + a_grave + "']  = 0.5")
print()
print("Actual expectations under the fitted model:")
print("\tp['dans'] + p['en'] =", p[0] + p[1])
print("\tp['dans'] + p['" + a_grave + "']  = " +
        str(p[0]+p[2]))
# (Or substitute "x.encode('latin-1')" if you have a primitive terminal.)

print("\nEstimated error in constraint satisfaction (should be close to 0):\n"
        + str(abs(model.expectations() - K)))
print("\nTrue error in constraint satisfaction (should be close to 0):\n" +
        str(abs(smallmodel.expectations() - K)))
