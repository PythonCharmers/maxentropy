#!/usr/bin/env python

""" Test functions for maximum entropy module.

Author: Ed Schofield, 2003-2005
Copyright: Ed Schofield, 2003-2005
"""

from numpy.testing import assert_allclose, TestCase
from numpy import arange, log, exp, ones
from scipy.special import logsumexp

class TestMaxentropy(TestCase):
    """Test whether logsumexp() function correctly handles large
    inputs.
    """
    def test_logsumexp(self):
        a = arange(200)
        desired = log(sum(exp(a)))
        assert_allclose(logsumexp(a), desired)

        # Now test with large numbers
        b = [1000,1000]
        desired = 1000.0 + log(2.0)
        assert_allclose(logsumexp(b), desired)

        n = 1000
        b = ones(n)*10000
        desired = 10000.0 + log(n)
        assert_allclose(logsumexp(b), desired)

