import math

from scipy.special import logsumexp
import numpy as np

from .basemodel import BaseModel
from .utils import evaluate_feature_matrix


class Model(BaseModel):
    """A maximum-entropy or minimum-divergence
    (exponential-form) model on a discrete sample space.

    Parameters
    ----------
    features : either (a) list of functions or (b) array

        (a) list of functions: [f_1, ..., f_m]
            Each function is expected to return either a vector or a
            scalar, depending on the `vectorized` keyword argument
            (below).

        (b) array: 2d array of shape (m, n)
            Matrix representing evaluations of features f_i(x) on all
            points x_1,...,x_n in the sample space.

    samplespace : sequence
        an enumerable (iterable and finite) sequence of values x in X that the model is
        defined over.

    vectorized : bool (default True)
        If True, the functions f_i(xs) accept a sequence of values
        xs = (x_1, ..., x_n) at once and return a vector of length n.

        If False, the functions f_i(x) take values x on the sample space and
        return real values.

    Algorithms
    ----------
    The optimization algorithm can be 'CG', 'BFGS', 'LBFGSB', 'Powell', or
    'Nelder-Mead'.

    The surface is guaranteed to be convex, so optimization should be
    relatively unproblematic with Model instances.

    The CG (conjugate gradients) method is the default; it is quite fast
    and requires only linear space in the number of parameters, (not
    quadratic, like Newton-based methods).

    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm is a
    variable metric Newton method.  It is perhaps faster than the CG
    method but requires O(N^2) instead of O(N) memory, so it is
    infeasible for more than about 10^3 parameters.

    The Powell algorithm doesn't require gradients.  For small models
    it is slow but robust.  For big models (where func and grad are
    simulated) with large variance in the function estimates, this
    may be less robust than the gradient-based algorithms.
    """
    def __init__(self,
                 features,
                 samplespace,
                 *,
                 vectorized=True,
                 format='csc_matrix',
                 verbose=False):
        super(Model, self).__init__()

        self.samplespace = samplespace
        self.max_output_lines = 20
        self.verbose = verbose
        self.vectorized = vectorized

        if isinstance(features, np.ndarray):
            self.F = features
        else:
            self.f = features
            self.F = evaluate_feature_matrix(features, samplespace,
                                             format=format,
                                             vectorized=vectorized,
                                             verbose=verbose)
        self._check_features()

    def fit(self, K):
        """Fit the maxent model p whose feature expectations <f_i(X)> are given
        by the vector K_i.

        Parameters
        ----------
        K : array
            desired expectation values <f_i(X)> to set as constraints
            on the model p(X).

        Notes
        -----
        Model expectations are computed either exactly, by summing
        over the given sample space.  If the sample space is continuous or too
        large to iterate over, use the 'BigModel' class instead.

        """
        super(Model, self).fit(K)

    # def setfeatures(self, f):
    #     """
    #     Create a new matrix self.F of features f of all points in the sample
    #     space.
    #
    #     Computes f(x) for each x in the sample space and stores them as self.F.
    #     This uses lots of memory but is much faster than re-evaluating them at
    #     each iteration.

    #     This is only appropriate when the sample space is finite.

    #     Parameters
    #     ----------
    #     f : list of functions
    #         f is a list of feature functions f_i(x) that operate on values x on
    #         the sample space, returning real values.
    #     """
    #     self.f = f
    #     self.F = evaluate_feature_matrix(f, self.samplespace,
    #                                      format='csr_matrix',
    #                                      vectorized=self.vectorized
    #                                      verbose=self.verbose)

    def _check_features(self):
        """
        Validation of whether the feature matrix has been set properly
        """
        # Ensure the feature matrix for the sample space has been set
        if not hasattr(self, 'F'):
            raise AttributeError('missing feature matrix')
        assert self.F.ndim == 2
        try:
            assert self.F.shape[1] == len(self.samplespace)
        except:
            raise AttributeError('the feature matrix is incompatible with the sample space. The number of columns must equal len(self.samplespace)')
        blank = False
        if hasattr(self.F, 'nnz'):
            if self.F.nnz == 0:
                blank = True
        else:
            if (self.F == 0).all():
                blank = True
        if blank:
            raise ValueError('the feature matrix is zero. Check whether your feature functions are vectorized and, if not, pass vectorized=False')

        # Watch out: if self.F is a dense NumPy matrix, its dot product
        # with a 1d parameter vector comes out as a 2d array, whereas if self.F
        # is a SciPy sparse matrix or dense NumPy array, its dot product
        # with the parameters is 1d. If it's a matrix, cast it to an array.
        if isinstance(self.F, np.matrix):
            self.F = np.asarray(self.F)

    def log_norm_constant(self):
        """Compute the log of the normalization constant (partition
        function) Z=sum_{x \in samplespace} p_0(x) exp(params . f(x)).
        The sample space must be discrete and finite.
        """
        # See if it's been precomputed
        if hasattr(self, 'logZ'):
            return self.logZ

        # Has F = {f_i(x_j)} been precomputed?
        if not hasattr(self, 'F'):
            raise AttributeError("first create a feature matrix F")

        # Good, assume the feature matrix exists
        # Calculate the dot product of F^T and the parameter vector:
        log_p_dot = self.F.T.dot(self.params)

        # Are we minimizing KL divergence?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs

        self.logZ = logsumexp(log_p_dot)
        return self.logZ

    def expectations(self):
        """The vector E_p[f(X)] under the model p_params of the vector of
        feature functions f_i over the sample space.
        """
        # For discrete models, use the representation E_p[f(X)] = p . F
        if not hasattr(self, 'F'):
            raise AttributeError("first set the feature matrix F")

        # A pre-computed matrix of features exists
        p = self.probdist()
        #return innerprod(self.F, p)
        return self.F.dot(p)

    def logprobdist(self):
        """Returns an array indexed by integers representing the
        logarithms of the probability mass function (pmf) at each point
        in the sample space under the current model (with the current
        parameter vector self.params).
        """
        # Have the features already been computed and stored?
        if not hasattr(self, 'F'):
            raise AttributeError("first set the feature matrix F")

        # Yes:
        # p(x) = exp(params.f(x)) / sum_y[exp params.f(y)]
        #      = exp[log p_dot(x) - logsumexp{log(p_dot(y))}]

        #log_p_dot = innerprodtranspose(self.F, self.params)
        # Calculate the dot product of F^T and the parameter vector:
        log_p_dot = self.F.T.dot(self.params)

        # Do we have a prior distribution p_0?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs
        if not hasattr(self, 'logZ'):
            # Compute the norm constant (quickly!)
            self.logZ = logsumexp(log_p_dot)
        return log_p_dot - self.logZ


    def probdist(self):
        """Returns an array indexed by integers representing the values
        of the probability mass function (pmf) at each point in the
        sample space under the current model (with the current parameter
        vector self.params).

        Equivalent to exp(self.logprobdist())
        """
        return np.exp(self.logprobdist())

    def pmf_function(self, f=None):
        """Returns the pmf p_params(x) as a function taking values on the
        model's sample space.  The returned pmf is defined as:

            p_params(x) = exp(params.f(x) - log Z)

        where params is the current parameter vector self.params.  The
        returned function p_params also satisfies
            all([p(x) for x in self.samplespace] == pmf()).

        The feature statistic f should be a list of functions
        [f1(),...,fn(x)].  This must be passed unless the model already
        contains an equivalent attribute 'model.f'.

        Requires that the sample space be discrete and finite, and stored
        as self.samplespace as a list or array.
        """

        if hasattr(self, 'logZ'):
            logZ = self.logZ
        else:
            logZ = self.log_partition_function()

        if f is None:
            try:
                f = self.f
            except AttributeError:
                raise AttributeError("either pass a list f of feature"
                           " functions or set this as a member variable self.f")

        # Do we have a prior distribution p_0?
        priorlogpmf = None
        if self.priorlogprobs is not None:
            try:
                priorlogpmf = self.priorlogpmf
            except AttributeError:
                raise AttributeError("prior probability mass function not set")

        def p(x):
            f_x = np.array([f[i](x) for i in range(len(f))], float)

            # Do we have a prior distribution p_0?
            if priorlogpmf is not None:
                priorlogprob_x = priorlogpmf(x)
                return math.exp(np.dot(self.params, f_x) + priorlogprob_x \
                                - logZ)
            else:
                return math.exp(np.dot(self.params, f_x) - logZ)
        return p

    def show(self):
        """
        Output the distribution
        """
        def show_x_and_px_values(n1, n2):
            """
            Output values x and their probabilities p_n from n=n1 to n=n2
            """
            for j in range(n1, n2):
                x = self.samplespace[j]
                print("\tx = {0:15s} \tp(x) = {1:.4f}".format(str(x), p[j]))

        p = self.probdist()
        n = len(self.samplespace)
        if n < self.max_output_lines:
            show_x_and_px_values(0, n)
        else:
            # Show the first e.g. 10 values, then ..., then the last 10 values
            show_x_and_px_values(0, self.max_output_lines // 2)
            print("\t...")
            show_x_and_px_values(n - self.max_output_lines // 2, n)

    # For backwards compatibility:
    showdist = show
