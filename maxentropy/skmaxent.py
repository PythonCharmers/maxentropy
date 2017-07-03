import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, DensityMixin
from sklearn.utils import check_array
from scipy.misc import logsumexp
from scipy.stats import entropy

from maxentropy.maxentutils import evaluate_feature_matrix
from maxentropy.model import BaseModel


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transform observations into a matrix of real-valued features
    suitable for fitting e.g. a MaxEntPrior model.

    The observations X can be given as a matrix or as a sequence of n Python
    objects representing points in some arbitrary sample space. For example,
    X can comprise n strings representing natural-language sentences.

    The result of calling the `transform(X)` method is an (n x m) matrix of
    each of the m features evaluated for each of the n observations.

    Parameters
    ----------
    features : either (a) list of functions or (b) array

        (a) list of functions: [f_1, ..., f_m]

        (b) array: 2d array of shape (m, n)
            Matrix representing evaluations of f_i(x) on all points
            x_1,...,x_n in the sample space.

    samplespace : sequence
        an enumerable sequence of values x in some discrete sample space X.

    vectorized : bool (default True)
        If True, the functions f_i(xs) are assumed to be "vectorized", meaning
        that each is assumed to accept a sequence of values xs = (x_1, ...,
        x_n) at once and each return a vector of length n.

        If False, the functions f_i(x) take individual values x on the sample
        space and return real values. This is likely to be slow down computing
        the features significantly.

    format : string
             Currently 'csr_matrix', 'csc_matrix', and 'ndarray'
             are recognized.


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
    def __init__(self,
                 features,
                 samplespace,
                 format='csr_matrix',
                 vectorized=True,
                 verbose=0):
        """

        """
        if format in ('csr_matrix', 'csc_matrix', 'ndarray'):
            self.format = format
        else:
            raise ValueError('matrix format not understood')
        self.features = features
        self.samplespace = samplespace
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
        X : a sequence (list or matrix) of observations
            These can be arbitrary objects: strings, row vectors, etc.

        Returns
        -------
        (n x d) array of features.
        """
        
        if isinstance(X, list):
            n_samples = len(X)
        else:
            X = check_array(X, accept_sparse=['csr', 'csc'])
            n_samples = X.shape[0]
            # if not X.shape[1] == 1:
            #     raise ValueError('X must have only one column')
        return evaluate_feature_matrix(self.features,
                                       X,
                                       vectorized=self.vectorized,
                                       verbose=self.verbose).T


class MinDivergenceModel(BaseEstimator, DensityMixin, BaseModel):
    """
    A discrete model with minimum Kullback-Leibler (KL) divergence from
    a given prior distribution subject to defined moment constraints.

    This includes models of maximum entropy ("MaxEnt") as a special case.
    
    This provides a principled method of assigning initial probabilities from
    prior information for Bayesian inference.

    Minimum divergence models and maximum entropy models have exponential form.
    The majority of well-known discrete and continuous probability
    distributions are special cases of maximum entropy models subject to moment
    constraints. This includes the following discrete probability
    distributions:
        
    - Uniform
    - Bernoulli
    - Geometric
    - Binomial
    - Poisson

    In the continuous case, models of maximum entropy also include
    distributions such as:

    - Lognormal
    - Gamma
    - Pareto
    - Cauchy
    - von Mises

    The information entropy of continuous probability distributions is
    sensitive to the choice of probability measure, where as the divergence is not.
    
    This makes continuous models easier to construct by minimizing divergence
    than by maximizing entropy.

    Parameters
    ----------

    features : either (a) list of functions or (b) array

        (a) list of functions: [f_1, ..., f_m]

        (b) array: 2d array of shape (m, n)
            Matrix F representing evaluations of f_i(x) on all points
            x_1,...,x_n in the sample space, such as the output of
            FeatureTransformer(...).transform(X)

    samplespace : sequence
        an enumerable sequence of values x in X that the model is
        defined over.

    vectorized : bool (default True)
        If True, the functions f_i(xs) are assumed to be "vectorized", meaning
        that each is assumed to accept a sequence of values xs = (x_1, ...,
        x_n) at once and each return a vector of length n.

        If False, the functions f_i(x) take individual values x on the sample
        space and return real values. This is likely to be slow down computing
        the features significantly.

    priorlogprobs : None (default) or 1d ndarray
        If not None, fitting the model minimizes Kullback-Leibler (KL)
        divergence between the prior distribution p_0 whose log probabilities
        are given by `priorlogprobs`. This is expected to be a 1d ndarray of
        length n = len(samplespace).
        
        If None, fitting the model maximizes Shannon information entropy H(p).

        In both cases the minimization / maximization are done subject to the same
        constraints on feature expectations.

    format : string
        Currently 'csr_matrix', 'csc_matrix', and 'ndarray'
        are recognized.

    algorithm : string (default 'CG')
        The algorithm can be 'CG', 'BFGS', 'LBFGSB', 'Powell', or
        'Nelder-Mead'.

        The CG (conjugate gradients) method is the default; it is quite fast
        and requires only linear space in the number of parameters, (not
        quadratic, like Newton-based methods).

        The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm is a
        variable metric Newton method.  It is perhaps faster than the CG
        method but requires O(N^2) instead of O(N) memory, so it is
        infeasible for more than about 10^3 parameters.

        The Powell algorithm doesn't require gradients.  For exact models
        it is slow but robust.  For big models (where func and grad are
        simulated) with large variance in the function estimates, this
        may be less robust than the gradient-based algorithms.

    verbose : int, (default=0)
        Enable verbose output.


    Example usage:
    --------------
    >>> # Fit a model p(x) for dice probabilities (x=1,...,6) with the
    >>> # single constraint E(X) = 18 / 4
    >>> def f0(x):
    >>>     return x

    >>> features = [f0]
    >>> k = [18 / 4]

    >>> samplespace = list(range(1, 7))
    >>> model = MinDivergenceModel(features, samplespace)
    >>> X = np.atleast_2d(k)
    >>> model.fit(X)

    """
    def __init__(self,
                 features,
                 samplespace,
                 priorlogprobs=None,
                 vectorized=True,
                 format='csr_matrix',
                 algorithm='CG',
                 verbose=0):
        """

        Parameters
        ----------
        """
        super(MinDivergenceModel, self).__init__()

        if format in ('csr_matrix', 'csc_matrix', 'ndarray'):
            self.format = format
        else:
            raise ValueError('matrix format not understood')

        if isinstance(features, np.ndarray):
            self.F = features
        else:
            self.F = evaluate_feature_matrix(features, samplespace,
                                             format=format,
                                             vectorized=vectorized,
                                             verbose=verbose)
            self.features = features

        self.samplespace = samplespace
        self.vectorized = vectorized
        self.priorlogprobs = priorlogprobs
        self.algorithm = algorithm
        self.verbose = verbose
        self.resetparams()

    def fit(self, X, y=None):
        """Fit the model of minimum divergence / maximum entropy subject to constraints E(U) = X

        Parameters
        ----------
        X : ndarray (dense) of shape [1, n_features]
            A row vector representing desired expectations of features.
            This is deliberate: models of minimum divergence / maximum entropy
            depend on the data only through the feature expectations.

        y : is not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        self

        """

        X = check_array(X)
        n_samples = X.shape[0]
        if n_samples != 1:
            raise ValueError('X must have only one row')
        # Explicitly call the BaseModel fit method:
        BaseModel.fit(self, X[0])
        # self.params = self.model.params.copy()
        return self

    def log_partition_function(self):
        """Compute the log of the normalization term (partition
        function) Z=sum_{x \in samplespace} p_0(x) exp(params . f(x)).
        The sample space must be discrete and finite.
        """
        # See if it's been precomputed
        if hasattr(self, 'logZ_'):
            return self.logZ_

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
        """The vector E_p[f(X)] under the model p_theta of the vector of
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
        # p(x) = exp(params . f(x)) / sum_y[exp params . f(y)]
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

    def divergence(self):
        """Return the Kullback-Leibler (KL) divergence between the model and
        the prior p0 (whose log probabilities were specified when constructing
        the model).

        This is defined as:

        D_{KL} (P || Q) = \sum_i P(x_i) log ( P(x_i) / Q(x_i) )
                        = \sum_i P(x_i) [ log P(x_i) - log Q(x_i) ]
        """

        if self.priorlogprobs is None:
            raise ValueError('divergence cannot be computed because no prior'
                             'distribution was defined when creating the model')

        p = self.probdist()
        log_p = self.logprobdist()
        divergence = np.sum(p * (log_p - self.priorlogprobs))

        # To verify with SciPy:
        # D = entropy(self.probdist(), np.exp(self.priorlogprobs))
        # assert np.allclose(D, divergence)

        return divergence


    def _check_features(self):
        """Validate whether the feature matrix has been set properly.
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
        # with the parameters is 1d. So, if it's a matrix, we cast it to an array.
        if isinstance(self.F, np.matrix):
            self.F = np.asarray(self.F)


