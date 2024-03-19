"""
Utility routines for the maxentropy package.

License: BSD-style (see LICENSE.md in main source directory)
"""

import random
import math
import cmath
import numpy as np
#from numpy import log, exp, asarray, ndarray, empty
import scipy.sparse
from scipy.special import logsumexp


__all__ = [
           'DivergenceError'
           ]



class DivergenceError(Exception):
    """Exception raised if the entropy dual has no finite minimum.
    """
    def __init__(self, message):
        self.message = message
        Exception.__init__(self)

    def __str__(self):
        return repr(self.message)

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
