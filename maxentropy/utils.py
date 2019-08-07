"""
Utility routines for the maximum entropy module.

Most of them are either Python replacements for the corresponding Fortran
routines or wrappers around matrices to allow the maxent module to
manipulate ndarrays, scipy sparse matrices, and PySparse matrices a
common interface.

Now the logsumexp() function, which was here, has been moved into
scipy.special.

License: BSD-style (see LICENSE.txt in main source directory)

"""

# Future imports must come before any code in 2.5
from __future__ import division
from __future__ import print_function

from builtins import range
__author__ = "Ed Schofield"
__version__ = '2.0'

import random
import math
import cmath
import numpy as np
#from numpy import log, exp, asarray, ndarray, empty
import scipy.sparse
from scipy.special import logsumexp


__all__ = ['feature_sampler',
           'dictsample',
           'dictsampler',
           'auxiliary_sampler_scipy',
           'evaluate_feature_matrix',
           'innerprod',
           'innerprodtranspose',
           'DivergenceError']



def feature_sampler(vec_f, auxiliary_sampler):
    """
    A generator function for tuples (F, log_q_xs, xs)

    Parameters
    ----------
    vec_f : function
        Pass `vec_f` as a (vectorized) function that operates on a vector of
        samples xs = {x1,...,xn} and returns a feature matrix (m x n), where m
        is some number of feature components.

    auxiliary_sampler : function
        Pass `auxiliary_sampler` as a function that returns a tuple
        (xs, log_q_xs) representing a sample to use for sampling (e.g.
        importance sampling) on the sample space of the model.

        xs : list, 1d ndarray, or 2d matrix (n x d)
            We require len(xs) == n.


    Yields
    ------
        tuples (F, log_q_xs, xs)

        F : matrix (m x n)
        log_q_xs : as returned by auxiliary_sampler
        xs : as returned by auxiliary_sampler

    """
    while True:
        xs, log_q_xs = auxiliary_sampler()
        F = vec_f(xs)  # compute feature matrix from points
        yield F, log_q_xs, xs


def dictsample(freq, size=None, return_probs=None):
    """
    Create a sample of the given size from the specified discrete distribution.

    Parameters
    ----------
    freq : a dictionary
        A mapping from values x_j in the sample space to probabilities (or
        unnormalized frequencies).

    size : a NumPy size parameter (like a shape tuple)
        Something passable to NumPy as a size argument to np.random.choice(...)

    return_probs : int, optional (default 0)
        None:     don't return pmf values at each sample point
        'prob':    return pmf values at each sample point
        'logprob': return log pmf values at each sample point

    Returns
    -------
    Returns a sample of the given size from the keys of the given
    dictionary `freq` with probabilities given according to the
    values (normalized to 1). Optionally returns the probabilities
    under the distribution of each observation.

    Example
    -------
    >>> freq = {'a': 10, 'b': 15, 'c': 20}
    >>> dictsample(freq, size=1)
    array([c, b, b, b, b, b, c, b, b, b], dtype=object)
    """
    n = len(freq)
    probs = np.fromiter(freq.values(), float)
    probs /= probs.sum()
    indices = np.random.choice(np.arange(n), size=size, p=probs)

    labels = np.empty(n, dtype=object)
    for i, label in enumerate(freq.keys()):
        labels[i] = label
    sample = labels[indices]

    if return_probs is None:
        return sample
    sampleprobs = probs[indices]
    if return_probs == 'prob':
        return sample, sampleprobs
    elif return_probs == 'logprob':
        return sample, np.log(sampleprobs)
    else:
        raise ValueError('return_probs must be "prob", "logprob", or None')


def dictsampler(freq, size=None, return_probs=None):
    """
    A generator of samples of the given size from the specified discrete
    distribution.

    Parameters
    ----------
    freq : a dictionary
        A mapping from values x_j in the sample space to probabilities (or
        unnormalized frequencies).

    size : a NumPy size parameter (like a shape tuple)
        Something passable to NumPy as a size argument to np.random.choice(...)

    return_probs : int, optional (default 0)
        None:     don't return pmf values at each sample point
        'prob':    return pmf values at each sample point
        'logprob': return log pmf values at each sample point

    Returns
    -------
    Returns a sample of the given size from the keys of the given
    dictionary `freq` with probabilities given according to the
    values (normalized to 1). Optionally returns the probabilities
    under the distribution of each observation.

    Example
    -------
    >>> freq = {'a': 10, 'b': 15, 'c': 20}
    >>> g = dictsample_gen(freq, size=1)
    >>> next(g)
    array([c, b, b, b, b, b, c, b, b, b], dtype=object)
    """
    while True:
        yield dictsample(freq, size=size, return_probs=return_probs)


def auxiliary_sampler_scipy(auxiliary, dimensions=1, n=10**5):
    """
    Sample (once) from the given scipy.stats distribution

    Parameters
    ----------
    auxiliary : a scipy.stats distribution object (rv_frozen)

    Returns
    -------
    sampler : function

        sampler(), when called with no parameters, returns a tuple
        (xs, log_q_xs), where:
            xs : matrix (n x d): [x_1, ..., x_n]: a sample
            log_q_xs: log pdf values under the auxiliary sampler for each x_j
    """
    def sampler():
        xs = auxiliary.rvs(size=(n, dimensions))
        log_q_xs = np.log(auxiliary.pdf(xs.T)).sum(axis=0)
        return (xs, log_q_xs)
    return sampler


def _logsumexpcomplex(values):
    """A version of logsumexp that should work if the values passed are
    complex-numbered, such as the output of robustarraylog().  So we
    expect:

    cmath.exp(logsumexpcomplex(robustarraylog(values))) ~= sum(values,axis=0)

    except for a small rounding error in both real and imag components.
    The output is complex.  (To recover just the real component, use
    A.real, where A is the complex return value.)
    """
    if len(values) == 0:
        return 0.0
    iterator = iter(values)
    # Get the first element
    while True:
        # Loop until we have a value greater than -inf
        try:
            b_i = next(iterator) + 0j
        except StopIteration:
            # empty
            return float('-inf')
        if b_i.real != float('-inf'):
            break

    # Now the rest
    for a_i in iterator:
        a_i += 0j
        if b_i.real > a_i.real:
            increment = robustlog(1.+cmath.exp(a_i - b_i))
            # print "Increment is " + str(increment)
            b_i = b_i + increment
        else:
            increment = robustlog(1.+cmath.exp(b_i - a_i))
            # print "Increment is " + str(increment)
            b_i = a_i + increment

    return b_i


def logsumexp_naive(values):
    """For testing logsumexp().  Subject to numerical overflow for large
    values (e.g. 720).
    """

    s = 0.0
    for x in values:
        s += math.exp(x)
    return math.log(s)


def robustlog(x):
    """Returns log(x) if x > 0, the complex log cmath.log(x) if x < 0,
    or float('-inf') if x == 0.
    """
    if x == 0.:
        return float('-inf')
    elif type(x) is complex or (type(x) is float and x < 0):
        return cmath.log(x)
    else:
        return math.log(x)


def _robustarraylog(x):
    """ An array version of robustlog.  Operates on a real array x.
    """
    arraylog = empty(len(x), np.complex64)
    for i in range(len(x)):
        xi = x[i]
        if xi > 0:
            arraylog[i] = math.log(xi)
        elif xi == 0.:
            arraylog[i] = float('-inf')
        else:
            arraylog[i] = cmath.log(xi)
    return arraylog


# def arrayexp(x):
#     """
#     OBSOLETE?
#     
#     Returns the elementwise antilog of the real array x.
#
#     We try to exponentiate with np.exp() and, if that fails, with
#     python's math.exp().  np.exp() is about 10 times faster but throws
#     an OverflowError exception for numerical underflow (e.g. exp(-800),
#     whereas python's math.exp() just returns zero, which is much more
#     helpful.
#     """
#     try:
#         ex = np.exp(x)
#     except OverflowError:
#         print("Warning: OverflowError using np.exp(). Using slower Python"\
#               " routines instead!")
#         ex = np.empty(len(x), float)
#         for j in range(len(x)):
#             ex[j] = math.exp(x[j])
#     return ex
#
# def arrayexpcomplex(x):
#     """
#     OBSOLETE?
#     
#     Returns the elementwise antilog of the vector x.
#
#     We try to exponentiate with np.exp() and, if that fails, with python's
#     math.exp().  np.exp() is about 10 times faster but throws an
#     OverflowError exception for numerical underflow (e.g. exp(-800),
#     whereas python's math.exp() just returns zero, which is much more
#     helpful.
#
#     """
#     try:
#         ex = np.exp(x).real
#     except OverflowError:
#         ex = np.empty(len(x), float)
#         try:
#             for j in range(len(x)):
#                 ex[j] = math.exp(x[j])
#         except TypeError:
#             # Perhaps x[j] is complex.  If so, try using the complex
#             # exponential and returning the real part.
#             for j in range(len(x)):
#                 ex[j] = cmath.exp(x[j]).real
#     return ex


def sample_wr(population, k):
    """Chooses k random elements (with replacement) from a population.
    (From the Python Cookbook).
    """
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [population[_int(_random() * n)] for i in range(k)]


def evaluate_feature_matrix(feature_functions,
                            xs,
                            vectorized=True,
                            format='csc_matrix',
                            dtype=float,
                            verbose=False):
    """Evaluate a (m x n) matrix of features `F` of the sample `xs` as:

        F[i, :] = f_i(xs[:])

    if xs is 1D, or as:

        F[i, j] = f_i(xs[:, j])

    if xs is 2D, for each feature function `f_i` in `feature_functions`.

    Parameters
    ----------
    feature_functions : a list of m feature functions f_i.

    xs : either:
        1. a (n x d) matrix representing n d-dimensional
           observations xs[j, :] for j=1,...,n.
        2. a 1d array or sequence (e.g list) of observations xs[j]
           for j=1,...,n.

    vectorized : bool (default True)
        If True, the feature functions f_i are assumed to be vectorized;
        then these will be passed all observations xs at once, in turn.

        If False, the feature functions f_i will be evaluated one at a time.

    format : str (default 'csc_matrix')
        Options: 'ndarray', 'csc_matrix', 'csr_matrix', 'dok_matrix'.
        If you have enough memory, it may be faster to create a dense
        ndarray and then construct a e.g. CSC matrix from this.

    Returns
    -------
    F : (m x n) matrix (in the given format: ndarray / csc_matrix / etc.)
        Matrix of evaluated features.

    """
    m = len(feature_functions)

    if isinstance(xs, np.ndarray) and xs.ndim == 2:
        n, d = xs.shape
        if d == 1 and vectorized:
            # xs may be a column vector, i.e. (n x 1) array.
            # In this case, reshape it to a 1d array. This
            # makes it easier to define functions that
            # operate on only one variable (the usual case)
            # given that sklearn's interface now forces 2D
            # arrays X when calling .transform(X) and .fit(X).
            xs = np.reshape(xs, n)
    else:
        n, d = len(xs), 1

    if format in ('dok_matrix', 'csc_matrix', 'csr_matrix'):
        F = scipy.sparse.dok_matrix((m, n), dtype=dtype)
    elif format == 'ndarray':
        F = np.empty((m, n), dtype=dtype)
    else:
        raise ValueError('matrix format not recognized')

    for i, f_i in enumerate(feature_functions):
        if verbose:
            print('Computing feature {i} of {m} ...'.format(i=i, m=m))
        if vectorized:
            F[i::m, :] = f_i(xs)
        else:
            for j in range(n):
                f_i_x = f_i(xs[j])
                if f_i_x != 0:
                    F[i,j] = f_i_x

    if format == 'csc_matrix':
        return F.tocsc()
    elif format == 'csr_matrix':
        return F.tocsr()
    else:
        return F


# def densefeatures(f, x):
#     """Returns a dense array of non-zero evaluations of the vector
#     functions fi in the list f at the point x.
#     """
#
#     return np.array([fi(x) for fi in f])


# def densefeaturematrix(f, sample, verbose=False):
#     """Compute an (m x n) dense array of non-zero evaluations of the
#     scalar functions fi in the list f at the points x_1,...,x_n in the
#     list sample.
#     """
#
#     # Was: return np.array([[fi(x) for fi in f] for x in sample])
#
#     m = len(f)
#     n = len(sample)
#
#     F = np.empty((m, n), float)
#     for i in range(m):
#         f_i = f[i]
#         for j in range(n):
#             x = sample[j]
#             F[i,j] = f_i(x)
#     return F


# def sparsefeatures(f, x, format='csc_matrix'):
#     """Compute an mx1 sparse matrix of non-zero evaluations of the
#     scalar functions f_1,...,f_m in the list f at the point x.
#
#     """
#     m = len(f)
#     if format in ('dok_matrix', 'csc_matrix', 'csr_matrix'):
#         sparsef = scipy.sparse.dok_matrix((m, 1))
#     else:
#         raise ValueError("sparse matrix format not recognized")
#
#     for i in range(m):
#         f_i_x = f[i](x)
#         if f_i_x != 0:
#             sparsef[i, 0] = f_i_x
#
#     if format == 'csc_matrix':
#         print("Converting to CSC matrix ...")
#         return sparsef.tocsc()
#     elif format == 'csr_matrix':
#         print("Converting to CSR matrix ...")
#         return sparsef.tocsr()
#     else:
#         return sparsef


# def sparsefeaturematrix(f, sample, format='csc_matrix', verbose=False):
#     """Compute an (m x n) sparse matrix of non-zero evaluations of the
#     scalar functions f_1,...,f_m in the list f at the points x_1,...,x_n
#     in the sequence 'sample'.
#
#     """
#     m = len(f)
#     n = len(sample)
#     if format in ('dok_matrix', 'csc_matrix', 'csr_matrix'):
#         sparseF = scipy.sparse.dok_matrix((m, n))
#     else:
#         raise ValueError("sparse matrix format not recognized")
#
#     for i in range(m):
#         if verbose:
#             print('Computing feature {i} of {m}'.format(i=i, m=m))
#         f_i = f[i]
#         for j in range(n):
#             x = sample[j]
#             f_i_x = f_i(x)
#             if f_i_x != 0:
#                 sparseF[i,j] = f_i_x
#
#     if format == 'csc_matrix':
#         return sparseF.tocsc()
#     elif format == 'csr_matrix':
#         return sparseF.tocsr()
#     else:
#         return sparseF


# def sparsefeaturematrix_vectorized(feature_functions, xs, format='csc_matrix'):
#     """
#     Evaluate a (m x n) matrix of features `F` of the sample `xs` as:
#
#         F[i, j] = f_i(xs[:, j])
#
#     Parameters
#     ----------
#     feature_functions : a list of feature functions f_i.
#
#     xs : either:
#         1. a (d x n) matrix representing n d-dimensional
#            observations xs[: ,j] for j=1,...,n.
#         2. a 1d array or sequence (e.g list) of observations xs[j]
#            for j=1,...,n.
#
#     The feature functions f_i are assumed to be vectorized. These will be
#     passed all observations xs at once, in turn.
#
#     Note: some samples may be more efficient / practical to compute
#     features one sample observation at a time (e.g. generated). For these
#     cases, use sparsefeaturematrix().
#
#     Only pass sparse=True if you need the memory savings. If you want a
#     sparse matrix but have enough memory, it may be faster to
#     pass dense=True and then construct a CSC matrix from the dense NumPy
#     array.
#
#     """
#     m = len(feature_functions)
#
#     if isinstance(xs, np.ndarray) and xs.ndim == 2:
#         d, n = xs.shape
#     else:
#         n = len(xs)
#     if not sparse:
#         F = np.empty((m, n), float)
#     else:
#         import scipy.sparse
#         F = scipy.sparse.lil_matrix((m, n), dtype=float)
#
#     for i, f_i in enumerate(feature_functions):
#         F[i::m, :] = f_i(xs)
#
#     if format == 'csc_matrix':
#         return F.tocsc()
#     elif format == 'csr_matrix':
#         return F.tocsr()
#     else:
#         return F


def old_vec_feature_function(feature_functions, sparse=False):
    """
    Create and return a vectorized function `features(xs)` that
    evaluates an (n x m) matrix of features `F` of the sample `xs` as:

        F[j, i] = f_i(xs[:, j])

    Parameters
    ----------
    feature_functions : a list of feature functions f_i.

    `xs` will be passed to these functions as either:
        1. an (n x d) matrix representing n d-dimensional
           observations xs[j, :] for j=1,...,n.
        2. a 1d array or sequence (e.g list) of observations xs[j]
           for j=1,...,n.

    The feature functions f_i are assumed to be vectorized. These will be
    passed all observations xs at once, in turn.

    Note: some samples may be more efficient / practical to compute
    features of one sample observation at a time (e.g. generated).

    Only pass sparse=True if you need the memory savings. If you want a
    sparse matrix but have enough memory, it may be faster to
    pass sparse=False and then construct a CSC matrix from the dense NumPy
    array.
    """
    if sparse:
        import scipy.sparse

    m = len(feature_functions)

    def vectorized_features(xs):
        if isinstance(xs, np.ndarray) and xs.ndim == 2:
            n, d = xs.shape
        else:
            n = len(xs)
        if not sparse:
            F = np.empty((n, m), float)
        else:
            F = scipy.sparse.lil_matrix((n, m), dtype=float)

        # Equivalent:
        # for i, f_i in enumerate(feature_functions):
        #     for k in range(len(xs)):
        #         F[len(feature_functions)*k+i, :] = f_i(xs[k])
        for i, f_i in enumerate(feature_functions):
            F[:, i::m] = f_i(xs)
        if not sparse:
            return F
        else:
            return scipy.sparse.csc_matrix(F)
    return vectorized_features


def dotprod(u,v):
    """
    This is a wrapper around general dense or sparse dot products.

    It is not necessary except as a common interface for supporting
    ndarray, scipy spmatrix, and PySparse arrays.

    Returns the dot product of the (1 x m) sparse array u with the
    (m x 1) (dense) numpy array v.

    """
    #print "Taking the dot product u.v, where"
    #print "u has shape " + str(u.shape)
    #print "v = " + str(v)

    try:
        dotprod = np.array([0.0])  # a 1x1 array.  Required by spmatrix.
        u.matvec(v, dotprod)
        return dotprod[0]               # extract the scalar
    except AttributeError:
        # Assume u is a dense array.
        return np.dot(u,v)


def innerprod(A,v):
    """
    This is a wrapper around general dense or sparse dot products.

    It is not necessary except as a common interface for supporting
    ndarray, scipy spmatrix, and PySparse arrays.

    Returns the inner product of the (m x n) dense or sparse matrix A
    with the n-element dense array v.  This is a wrapper for A.dot(v) for
    dense arrays and spmatrix objects, and for A.matvec(v, result) for
    PySparse matrices.

    """

    # We assume A is sparse.
    (m, n) = A.shape
    vshape = v.shape
    try:
        (p,) = vshape
    except ValueError:
        (p, q) = vshape
    if n != p:
        raise TypeError("matrix dimensions are incompatible")
    if isinstance(v, np.ndarray):
        try:
            # See if A is sparse
            A.matvec
        except AttributeError:
            # It looks like A is dense
            return np.dot(A, v)
        else:
            # Assume A is sparse
            if scipy.sparse.isspmatrix(A):
                innerprod = A.matvec(v)   # This returns a float32 type. Why???
                return innerprod
            else:
                # Assume PySparse format
                innerprod = np.empty(m, float)
                A.matvec(v, innerprod)
                return innerprod
    elif scipy.sparse.isspmatrix(v):
        return A * v
    else:
        raise TypeError("unsupported types for inner product")


def innerprodtranspose(A,v):
    """
    This is a wrapper around general dense or sparse dot products.

    It is not necessary except as a common interface for supporting
    ndarray, scipy spmatrix, and PySparse arrays.

    Computes A^T V, where A is a dense or sparse matrix and V is a numpy
    array.  If A is sparse, V must be a rank-1 array, not a matrix.  This
    function is efficient for large matrices A.  This is a wrapper for
    A.T.dot(v) for dense arrays and spmatrix objects, and for
    A.matvec_transp(v, result) for pysparse matrices.

    """

    (m, n) = A.shape
    #pdb.set_trace()
    if hasattr(A, 'matvec_transp'):
        # A looks like a PySparse matrix
        if len(v.shape) == 1:
            innerprod = np.empty(n, float)
            A.matvec_transp(v, innerprod)
        else:
            raise TypeError("innerprodtranspose(A,v) requires that v be "
                    "a vector (rank-1 dense array) if A is sparse.")
        return innerprod
    elif scipy.sparse.isspmatrix(A):
        return (A.conj().transpose() * v).transpose()
    else:
        # Assume A is dense
        if isinstance(v, np.ndarray):
            # v is also dense
            if len(v.shape) == 1:
                # We can't transpose a rank-1 matrix into a row vector, so
                # we reshape it.
                vm = v.shape[0]
                vcolumn = np.reshape(v, (1, vm))
                x = np.dot(vcolumn, A)
                return np.reshape(x, (n,))
            else:
                #(vm, vn) = v.shape
                # Assume vm == m
                x = np.dot(np.transpose(v), A)
                return np.transpose(x)
        else:
            raise TypeError("unsupported types for inner product")


def rowmeans(A):
    """
    This is a wrapper for general dense or sparse dot products.

    It is only necessary as a common interface for supporting ndarray,
    scipy spmatrix, and PySparse arrays.

    Returns a dense (m x 1) vector representing the mean of the rows of A,
    which be an (m x n) sparse or dense matrix.

    >>> a = np.array([[1,2],[3,4]], float)
    >>> rowmeans(a)
    array([ 1.5,  3.5])

    """
    if type(A) is np.ndarray:
        return A.mean(1)
    else:
        # Assume it's sparse
        try:
            n = A.shape[1]
        except AttributeError:
            raise TypeError("rowmeans() only works with sparse and dense "
                            "arrays")
        rowsum = innerprod(A, np.ones(n, float))
        return rowsum / float(n)

def columnmeans(A):
    """
    This is a wrapper for general dense or sparse dot products.

    It is only necessary as a common interface for supporting ndarray,
    scipy spmatrix, and PySparse arrays.

    Returns a dense (1 x n) vector with the column averages of A, which can
    be an (m x n) sparse or dense matrix.

    >>> a = np.array([[1,2],[3,4]],'d')
    >>> columnmeans(a)
    array([ 2.,  3.])

    """
    if type(A) is np.ndarray:
        return A.mean(0)
    else:
        # Assume it's sparse
        try:
            m = A.shape[0]
        except AttributeError:
            raise TypeError("columnmeans() only works with sparse and dense "
                            "arrays")
        columnsum = innerprodtranspose(A, np.ones(m, float))
        return columnsum / float(m)

def columnvariances(A):
    """
    This is a wrapper for general dense or sparse dot products.

    It is not necessary except as a common interface for supporting ndarray,
    scipy spmatrix, and PySparse arrays.

    Returns a dense (1 x n) vector with unbiased estimators for the column
    variances for each column of the (m x n) sparse or dense matrix A.  (The
    normalization is by (m - 1).)

    >>> a = np.array([[1,2], [3,4]], 'd')
    >>> columnvariances(a)
    array([ 2.,  2.])

    """
    if type(A) is np.ndarray:
        return np.std(A,0)**2
    else:
        try:
            m = A.shape[0]
        except AttributeError:
            raise TypeError("columnvariances() only works with sparse "
                            "and dense arrays")
        means = columnmeans(A)
        return columnmeans((A-means)**2) * (m/(m-1.0))

def flatten(a):
    """Flattens the sparse matrix or dense array/matrix 'a' into a
    1-dimensional array
    """
    if scipy.sparse.isspmatrix(a):
        return a.A.flatten()
    else:
        return np.asarray(a).flatten()

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
