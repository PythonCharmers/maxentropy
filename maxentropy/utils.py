"""
Utility routines for the maxentropy package.

License: BSD-style (see LICENSE.md in main source directory)
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator

import numpy as np
import toolz as tz

# from numpy import log, exp, asarray, ndarray, empty
import scipy.sparse
from sklearn.base import BaseEstimator, TransformerMixin, DensityMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils import check_array


__all__ = ["DivergenceError"]


class DivergenceError(Exception):
    """Exception raised if the entropy dual has no finite minimum."""

    def __init__(self, message):
        self.message = message
        Exception.__init__(self)

    def __str__(self):
        return repr(self.message)


def bounds_stretched(X, stretch_factor=0.1):
    """
    Returns (min, max) pairs for each column of X, with each interval stretched
    by the factor (1 + stretch_factor).

    Parameters
    ----------
        X: a 2d array of size (n, m)

    Returns
    -------
        tuple:
            (stretched_minima, stretched_maxima)

        where each of stretched_minima and stretched_maxima is a length-m array
        with the minima (or maxima) of each column, stretched as requested.
    """
    check_array(X)
    minima = X.min(axis=0)
    maxima = X.max(axis=0)
    widths = maxima - minima
    stretched_widths = widths * (1 + stretch_factor)
    stretched_minima = minima - (stretched_widths - widths) / 2
    stretched_maxima = maxima + (stretched_widths - widths) / 2
    return (stretched_minima, stretched_maxima)


def auxiliary_sampler_gaussian_mixture(
    model: GaussianMixture, n_samples=1, output_format="numpy"
):
    """
    A generator function for samples from the given scipy.stats distribution.

    Parameters
    ----------
    model : a fitted sklearn.mixture.GaussianMixture model

    n_samples: the number of samples to generate and yield in each iteration

    output_format: either 'numpy' or 'pytorch'

    Returns
    -------
    A generator that yields tuples of length 2:
        (xs, log_q_xs)
    where:
        xs : 2d array (n_samples x n_dims): [x_1, ..., x_n]: a sample
    and:
        log_q_xs: log pdf values under the auxiliary sampler q for each x_j (for j = 1 through n_samples)
    """
    while True:
        xs, _ = model.sample(n_samples=n_samples)
        log_q_xs = model.score_samples(xs)
        if output_format == "numpy":
            yield (xs, log_q_xs)
        elif output_format == "pytorch":
            import torch

            yield (
                torch.tensor(xs, dtype=torch.float32),
                torch.tensor(log_q_xs, dtype=torch.float32),
            )


def auxiliary_sampler_scipy(distribution, n_dims=1, n_samples=1, output_format="numpy"):
    """
    A generator function for samples from the given scipy.stats distribution.

    Parameters
    ----------
    distribution : a scipy.stats distribution object (rv_frozen)

        Note: distribution.rvs(size=(n_samples, n_dims) must return an array (n_samples x n_dims).

    n_dims: int or tuple of ints
        the number of dimensions we want in our samples (each using the same
        distribution object). This can be a tuple (n_dims0, n_dims1) to sample
        3d tensors of shape (n_samples, n_dims, n_features) from the given
        distribution.

    n_samples: the number of samples to generate and yield in each iteration

    output_format: either 'numpy' or 'pytorch'

    Returns
    -------
    A generator that yields tuples of length 2:
        (xs, log_q_xs)
    where:
        xs : 2d array (n_samples x n_dims): [x_1, ..., x_n]: a sample
    or:
        xs : 3d array (n_samples x n_dims[0] x n_dims[1]): [x_1, ..., x_n_samples]: a sample
    and:
        log_q_xs: log pdf values under the auxiliary sampler q for each x_j (for j = 1 through n_samples)
    """
    if isinstance(n_dims, tuple):
        size = (n_samples,) + n_dims
    else:
        assert isinstance(n_dims, int)
        size = (n_samples, n_dims)
    reduce_axes = -1 if len(size) == 2 else (-1, -2)
    while True:
        xs = distribution.rvs(size=size)
        log_q_xs = np.log(distribution.pdf(xs)).sum(reduce_axes)
        if output_format == "numpy":
            yield (xs, log_q_xs)
        elif output_format == "pytorch":
            import torch

            yield (
                torch.tensor(xs, dtype=torch.float32),
                torch.tensor(log_q_xs, dtype=torch.float32),
            )


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform observations into a 2d array of real-valued features
    suitable for fitting e.g. a MinDivergenceModel.

    The observations X can be given as a 2d array or as a sequence of n Python
    objects representing points in some arbitrary sample space. For example,
    X can comprise n strings representing natural-language sentences.

    The result of calling the `transform(X)` method is an (n x m) array of
    each of the m features evaluated for each of the n observations.

    Parameters
    ----------
    features : either (a) list of functions or (b) array

        (a) list of functions: [f_1, ..., f_m]

        (b) array: 2d array of shape (n, m)
            Matrix representing evaluations of f_i(x) from i=1 to i=m on all
            points or observations x_1,...,x_n in the sample space.

    samplespace : sequence
        an enumerable sequence of values x in some discrete sample space X.

    vectorized : bool (default True)
        If True, the functions f_i(xs) are assumed to be "vectorized", meaning
        that each is assumed to accept a sequence of values xs = (x_1, ...,
        x_n) at once and each return a vector of length n.

        If False, the functions f_i(x) take individual values x on the sample
        space and return real values. This is likely to be slow down computing
        the features significantly.

    array_format : string
             Currently 'csr_array', 'csc_array', and 'ndarray' are recognized.


    Example usage:
    --------------
    # Fit a model with the constraint E(X) = 18 / 4
    def f0(x):
        return x

    features = [f0]

    samplespace = np.arange(6) + 1
    transformer = FeatureTransformer(features, samplespace)
    X = np.array([[5, 6, 6, 1]]).T
    >>> transformer.transform(X)

    """

    def __init__(
        self,
        feature_functions,
        samplespace,
        *,
        array_format="csr_array",
        vectorized=True,
        verbose=0,
    ):
        if array_format in {
            "csr_array",
            "csc_array",
            "ndarray",
        }:
            self.array_format = array_format
        else:
            raise ValueError("array format not understood")
        self.feature_functions = feature_functions
        self.samplespace = samplespace
        self.array_format = array_format
        self.vectorized = vectorized
        self.verbose = verbose

    def fit(self, X, y=None):
        """Unused.

        Parameters
        ----------
        X : Unused.

        y : Unused.

        These are placeholders to allow for usage in a Pipeline.

        Returns
        -------
        self

        """
        return self

    def transform(self, X, y=None):
        """
        Apply features to a sequence of observations X

        Parameters
        ----------
        X : a sequence (list or array) of observations.
            These can be arbitrary objects: strings, row vectors, etc.

        Returns
        -------
        (n x d) array of features.
        """

        n_samples = len(X)
        # if isinstance(X, list):
        #     n_samples = len(X)
        # else:
        X = check_array(X, accept_sparse=["csr", "csc"], ensure_2d=False, dtype=None)
        # n_samples = X.shape[0]
        # if not X.shape[1] == 1:
        #     raise ValueError('X must have only one column')
        F = evaluate_feature_matrix(
            self.feature_functions, X, vectorized=self.vectorized, verbose=self.verbose
        )
        assert n_samples, len(self.feature_functions) == F.shape
        return F


def force_array(matrix):
    """
    Takes an array or matrix (either numpy ndarray or scipy.sparse array /
    matrix) and returns an array (either numpy ndarray or scipy.sparse array).

    Doesn't copy if possible.
    """
    usage_error = (
        "Only ndarray, csr, csc, and dok (array or matrix) are currently accepted."
    )
    if scipy.sparse.issparse(matrix):
        if not scipy.sparse.isspmatrix(matrix):
            # scipy.sparse array
            return matrix
        if isinstance(matrix, scipy.sparse.csr_matrix):
            return scipy.sparse.csr_array(matrix)
        elif isinstance(matrix, scipy.sparse.csc_matrix):
            return scipy.sparse.csc_array(matrix)
        elif isinstance(matrix, scipy.sparse.dok_matrix):
            return scipy.sparse.dok_array(matrix)
        else:
            raise ValueError(usage_error)
    else:
        if not isinstance(matrix, np.ndarray):
            raise ValueError(usage_error)
        return np.asarray(matrix)


# Previously sampleFgen, with different argument order
def feature_sampler(
    features: list[Callable],
    sampler: Iterator,
    *,
    array_format: str = "csc_array",
    vectorized: bool = True,
    dtype=float,
    omit_samples=False,
):
    """
    A generator function that yields features of random points.

    Parameters
    ----------
        features : list of functions
            Pass `features` as a list of d feature functions f_i to apply to the
            outputs yielded by `sampler`.

        sampler : generator
            `sampler` must be a generator that yields tuples (xs, log_q_xs)
            representing a sample to use for sampling (e.g. importance sampling)
            on the sample space of the model, which must be:

            - xs : list, 1d ndarray, or 2d array (n x m)
                   The samples generated by `sampler`. We require len(xs) == n.

            - log_q_xs : list or 1d ndarray of shape (n,)
                         The log pdf values of the samples xs under the
                         probability distribution q(x) that governs `sampler`.


        array_format: str
            The output format for the matrices F we yield. Either 'ndarray' for a
            dense array or an array format string understood by scipy.sparse --
            currently just 'csc_array' or 'csr_array' -- for constructing a
            scipy.sparse array of features.

        vectorized: bool
            If True, the feature functions f_i are assumed to be vectorized;
            then these will be passed all observations xs at once, in turn.

            If False, the feature functions f_i will be evaluated on each x in the
            sample xs, one at a time.

        dtype: a NumPy-compatible dtype for the output array.

        omit_samples: bool
            Whether to yield values of None for the sample xs.
            Passing omit_samples = True saves memory, since fitting the model
            doesn't require these, only the features. But having the samples makes
            the model easier to inspect / understand / debug.

    Yields
    ------
    If omit_samples is False, the generator yields the tuple:
        (F, logprobs, xs)
    Or if omit_samples is True, the generator yields the tuple:
        (F, logprobs)

    where:
        - F is the computed dense or sparse feature matrix (of size n x d)
        - logprobs is the same vector of log probs yielded by sampler (length n)
        - xs is the sample yielded by auxiliary_sampler (length n)

    """
    # We require an array (not matrix) type since we need consistency for elementwise multiplication versus matrix products in density.py:
    while True:
        xs, logprobs = next(sampler)
        F = evaluate_feature_matrix(
            features,
            xs,
            vectorized=vectorized,
            array_format=array_format,
            dtype=dtype,
        )
        if omit_samples:
            yield F, logprobs
        else:
            yield F, logprobs, xs


def evaluate_feature_matrix(
    feature_functions: list,
    xs: list | np.ndarray,
    *,
    vectorized: bool = True,
    array_format: bool = "csc_array",
    dtype=float,
    verbose: bool = False,
):
    """Evaluate a (n x d) array of features `F` of the sample `xs` as:

        F[:, i] = f_i(xs)

    if xs is 1D, or as:

        F[j, i] = f_i(xs[j, :])

    if xs is 2D, for each feature function `f_i` in `feature_functions`.

    Parameters
    ----------
    feature_functions :
        a list of d feature functions f_i. These will be passed xs and must return:
            - a 1d array f_i(x) = (f_i(x_1), ..., f_i(x_n)) for j=1,...,n if vectorized is True; or
            - a scalar f_i(x_j) if vectorized is False

    xs : either:
        1. a 1d array or sequence (e.g list) of observations [x_j
           for j=1,...,n].
        2. a (n x m) array representing n m-dimensional
           observations xs[j, :] for j=1,...,n.

    vectorized : bool (default True)
        If True, the feature functions f_i are assumed to be vectorized;
        then these will be passed all observations xs at once, in turn.

        If False, the feature functions f_i will be evaluated on each x_j in the
        sample xs, one at a time, for j=1,...,n.

    array_format : str (default 'csc_array')
        Options: 'ndarray', 'csc_array', 'csr_array', 'dok_array'. If you have
        enough memory, it may be faster to create a dense ndarray and then
        construct a e.g. CSC array from this.

    Returns
    -------
    F : (n x d) array (in the given array format: ndarray / csc_array /
        csr_array). Array of evaluated features.

    """
    d = len(feature_functions)

    if isinstance(xs, np.ndarray) and xs.ndim == 2:
        n, m = xs.shape
        # if m == 1 and vectorized:
        #     # xs may be a column vector, i.e. (n x 1) array.
        #     # In this case, reshape it to a 1d array. This
        #     # makes it easier to define functions that
        #     # operate on only one variable (a common case)
        #     # given that sklearn's interface now forces 2D
        #     # arrays X when calling .transform(X) and .fit(X).
        #     xs = np.reshape(xs, n)
    else:
        n, m = len(xs), 1

    if array_format in ("csc_array", "csr_array"):
        F = scipy.sparse.dok_array((n, d), dtype=dtype)
    elif array_format == "ndarray":
        F = np.zeros((n, d), dtype=dtype)
    else:
        raise ValueError("array format not recognized")

    for i, f_i in enumerate(feature_functions):
        if verbose:
            print(f"Computing feature {i=} of {d=} ...")
        if vectorized:
            output = f_i(xs)
            ndim = np.ndim(output)
            if ndim == 0:
                raise ValueError(
                    f"Your feature function (for feature {i}) is returning scalar output. Change it to return 1-dimensional output or call this function with vectorized=False"
                )
            # elif ndim == 2:
            #     vec_output = np.ravel(output)
            #     if vec_output.ndim != 1:
            #         raise ValueError(
            #             f"Your feature function (for feature {i}) is returning 2d output. Change it to return 1d output."
            #         )
            try:
                F[:, i] = output
            except ValueError:
                raise ValueError(
                    f"Something is wrong with the output from your feature function (for feature {i}). Debug this! We want 1d output but we are getting:\n{output=}"
                )
        else:
            try:
                for j in range(n):
                    f_i_xj = f_i(xs[j])
                    if f_i_xj != 0:
                        F[j, i] = np.squeeze(
                            f_i_xj
                        )  # squeeze in case f_i is vectorized but the user passed vectorize=False
            except TypeError:
                raise TypeError(
                    "Failed to evaluating feature functions on individual "
                    "samples in xs. Are your feature functions already vectorized "
                    "(i.e. they don't take individual values x_j in xs)? If so, "
                    "pass vectorized=True."
                )
    if verbose:
        print("Finished computing features.")

    if array_format == "csc_array":
        return F.tocsc()
    elif array_format == "csr_array":
        return F.tocsr()
    else:
        assert isinstance(F, np.ndarray)
        return F


def dictsample(freq, size=(), return_probs="logprob"):
    """
    Create a sample of the given size from the specified discrete distribution.

    Parameters
    ----------
    freq : a dictionary
        A mapping from values x_j in the sample space to probabilities (or
        unnormalized frequencies).

    size : a NumPy size parameter (like a shape tuple)
        Something passable to NumPy as a size argument to np.random.choice(...)

    return_probs : string or None
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
    >>> dictsample(freq, size=10)
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
    if return_probs == "prob":
        return sample, sampleprobs
    elif return_probs == "logprob":
        return sample, np.log(sampleprobs)
    else:
        raise ValueError('return_probs must be "prob", "logprob", or None')


def dictsampler(freq, size=()):
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

    return_probs : string or None
        None:     don't return pmf values at each sample point
        'prob':    return pmf values at each sample point
        'logprob': return log pmf values at each sample point

    Returns
    -------
    Returns a tuple
        (xs, logprob_xs)

    defined as:

        xs : a sample of the given `size` from the keys of the given
             dictionary `freq` with probabilities given according to its
             values (normalized to 1).

        logprob_xs : a vector of the log probabilities of each observation x in
                     xs.

    Example
    -------
    >>> freq = {'a': 10, 'b': 15, 'c': 20}
    >>> g = dictsampler(freq, size=10)
    >>> next(g)
    array([c, b, b, b, b, b, c, b, b, b], dtype=object)
    """
    while True:
        yield dictsample(freq, size=size, return_probs="logprob")


def make_uniform_sampler(
    minima, maxima, n_samples=100_000, output_format="numpy"
) -> Generator:
    """
    Returns a generator suitable for passing into MinDivergenceDensity and
    D2GDensity models.

    Pass minima and maxima, where each has length equal to X.shape[1].
    """
    minima = np.ravel(minima)
    maxima = np.ravel(maxima)
    assert minima.shape == maxima.shape and minima.ndim == 1
    n_dims = len(minima)
    uniform_dist = scipy.stats.uniform(minima, maxima - minima)
    sampler = auxiliary_sampler_scipy(
        uniform_dist, n_dims=n_dims, n_samples=n_samples, output_format=output_format
    )
    return sampler


def make_uniform_sampler_3d(
    minima, maxima, n_dims=1, n_samples=10_000, output_format="numpy"
) -> Generator:
    """
    Returns a generator suitable for passing into MinDivergenceDensity and
    D2GDensity models.

    Pass minima and maxima, where each has length equal to X.shape[1].
    """
    uniform_dist = scipy.stats.uniform(minima, maxima - minima)
    sampler = auxiliary_sampler_scipy(
        uniform_dist,
        n_dims=(n_dims, len(minima)),
        n_samples=n_samples,
        output_format=output_format,
    )
    return sampler


@tz.curry
def combine_posterior_densities(posterior_densities: list[DensityMixin], X: np.ndarray):
    """
    Pass a list of class-specific posterior densities p(x | k)
    for classes k=0, 1, ..., K.

    Returns a predict_log_proba(X) function that returns a N x K matrix
    of log probabilities for classes k=1, ..., K.
    """
    values = np.vstack(
        [posterior.predict_log_proba(X) for posterior in posterior_densities]
    ).T
    return values


@tz.curry
def evaluate_fn_and_extract_column(fn, k, X):
    """
    A curried function to evaluate fn(X)[:, k].

    Calling `evaluate_fn_and_extract_column(fn, k)` returns a partial function
    that expects X and returns the kth column of the result.

    Use this to prevent NASTY bugs that arise when creating lambdas in a for loop
    that depend on the loop variable.

    Example of the bug:

        >>> functions = [lambda x: k * x for k in range(10)]   # WRONG!
        >>> functions[0](5)
        45

    What to do instead:
        >>> @tz.curry
        >>> def mul(k, x):
        ...     return k * x
        >>> functions = [mul(k) for k in range(10)]
        >>> functions[0](5)
        0
    """
    return fn(X)[:, k]


@tz.curry
def cutoff_low(cutoff, column, X):
    """
    Adding this as a constraint forces our model for each class
    to assign the same probability of being above this cutoff as the
    per-class empirical frequency in the dataset.
    """
    return X[:, column] >= cutoff


@tz.curry
def cutoff_high(cutoff, column, X):
    """
    Adding this as a constraint forces our model for each class
    to assign the same probability of being below this cutoff as the
    per-class empirical frequency in the dataset.
    """
    return X[:, column] <= cutoff


def _test():
    import doctest

    doctest.testmod()


if __name__ == "__main__":
    _test()
