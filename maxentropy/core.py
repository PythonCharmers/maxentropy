from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import types

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, DensityMixin  # , _fit_context
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from scipy.special import logsumexp

from maxentropy.utils import evaluate_feature_matrix, feature_sampler
from maxentropy.base import BaseMinKLDensity


class MinKLDensity(BaseEstimator, DensityMixin, BaseMinKLDensity):
    """
    A discrete model with minimum Kullback-Leibler (KL) divergence from
    a given prior distribution subject to defined moment constraints.

    This includes models of maximum entropy ("MaxEnt") as a special case, with
    a flat prior distribution.

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
    sensitive to the choice of probability measure, whereas the divergence is not.

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

    prior_log_pdf : None (default) or function
        Do you seek to minimize the KL divergence between the model and a
        prior density p_0?  If not, set this to None; then we maximize
        the Shannon information entropy H(p).

        If so, set this to a function that can take an array of values X of
        shape (k x m) and return an array of the log probability densities
        p_0(x) under the prior p_0 for each (row vector) x in the sample space.
        The output of calling this function, if it has shape (k, 1), will be
        squeezed to be a vector of shape (k,).

        For models involving simulation, set this to a function
        that should return p_0(x) for each x in the random sample produced by
        the auxiliary distribution.

        In both cases the minimization / maximization are done subject to the
        same constraints on feature expectations.

    For other parameters, see notes in the BaseModel docstring:
    - algorithm
    - matrix_format
    - verbose

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

    def __init__(
        self,
        feature_functions: list[types.FunctionType],
        samplespace,
        prior_log_pdf=None,
        *,
        vectorized=True,
        matrix_format="csr_matrix",
        algorithm="CG",
        max_iter=1000,
        verbose=0,
        warm_start=False,
    ):
        super().__init__(
            feature_functions,
            prior_log_pdf=prior_log_pdf,
            vectorized=vectorized,
            matrix_format=matrix_format,
            algorithm=algorithm,
            max_iter=max_iter,
            verbose=verbose,
            warm_start=warm_start,
        )

        # TODO: reinstate this in the future for a large speedup opportunity if
        # there are many functions:
        # if isinstance(features, np.ndarray):
        #     self.F = features
        # else:

        self.F = evaluate_feature_matrix(
            feature_functions,
            samplespace,
            matrix_format=matrix_format,
            vectorized=vectorized,
            verbose=verbose,
        )

        self.samplespace = samplespace
        if prior_log_pdf is not None:
            lp = prior_log_pdf(self.samplespace)
            self.priorlogprobs = np.reshape(lp, len(self.samplespace))
        self.resetparams(len(feature_functions))

    def entropy(self):
        """
        Compute the entropy of the model with the current parameters self.params.
        """
        # H = -sum(pk * log(pk))
        H = -np.sum(self.probdist() * self.log_probdist())
        # or simpler but less efficient:
        # H = scipy.stats.entropy(self.probdist())
        return H

    def kl_divergence(self):
        """
        Compute the Kullback-Leibler divergence (often called relative entropy)
        K(p || q) of the model p with current parameters self.params versus the
        prior probability distribution q specified as self.log_prior_x.
        """
        if self.prior_log_pdf is None:
            raise Exception(
                "Model needs to be initialized with a prior distribution to calculate KL divergence"
            )
        # prior_log_pdf = prior_log_pdf(self.samplespace)
        kl_div = np.sum(self.probdist() * (self.log_probdist() - self.priorlogprobs))
        return kl_div

    def log_norm_constant(self):
        r"""Compute the log of the normalization term (partition
        function) Z=sum_{x \in samplespace} p_0(x) exp(params . f(x)).
        The sample space must be discrete and finite.
        """
        # See if it's been precomputed
        if hasattr(self, "logZ_"):
            return self.logZ_

        # Has F = {f_i(x_j)} been precomputed?
        if not hasattr(self, "F"):
            raise AttributeError("first create a feature matrix F")

        # Good, assume the feature matrix exists
        # Calculate the dot product of F and the parameter vector:
        log_p_dot = self.F.dot(self.params)

        # Are we minimizing KL divergence?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs

        self.logZ = logsumexp(log_p_dot)
        return self.logZ

    def feature_expectations(self):
        """The vector E_p[f(X)] under the model p_theta of the vector of
        feature functions f_i over the sample space.
        """
        # For discrete models, use the representation E_p[f(X)] = p . F
        if not hasattr(self, "F"):
            raise AttributeError("first set the feature matrix F")

        # A pre-computed matrix of features exists
        p = self.probdist()
        return self.F.T.dot(p)

    # For compatibility with older versions:
    expectations = feature_expectations

    def log_probdist(self):
        """Returns an array indexed by integers representing the
        logarithms of the probability mass function (pmf) at each point
        in the sample space under the current model (with the current
        parameter vector self.params).
        """
        # Have the features already been computed and stored?
        if not hasattr(self, "F"):
            raise AttributeError("first set the feature matrix F")

        # Yes:
        # p(x) = exp(params . f(x)) / sum_y[exp params . f(y)]
        #      = exp[log p_dot(x) - logsumexp{log(p_dot(y))}]

        # Calculate the dot product of F and the parameter vector:
        log_p_dot = self.F.dot(self.params)

        # Do we have a prior distribution p_0?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs
        if not hasattr(self, "logZ"):
            # Compute the norm constant (quickly!)
            self.logZ = logsumexp(log_p_dot)
        return log_p_dot - self.logZ

    def probdist(self):
        """Returns an array indexed by integers representing the values
        of the probability mass function (pmf) at each point in the
        sample space under the current model (with the current parameter
        vector self.params).

        Equivalent to exp(self.log_probdist())
        """
        return np.exp(self.log_probdist())

    def divergence(self):
        r"""Return the Kullback-Leibler (KL) divergence between the model and
        the prior p0 (whose log pdf was specified when constructing
        the model).

        This is defined as:

        D_{KL} (P || Q) = \sum_i P(x_i) log ( P(x_i) / Q(x_i) )
                        = \sum_i P(x_i) [ log P(x_i) - log Q(x_i) ]
        """

        if self.priorlogprobs is None:
            raise ValueError(
                "divergence cannot be computed because no prior "
                "distribution was defined when creating the model"
            )

        p = self.probdist()
        log_p = self.log_probdist()
        divergence = np.sum(p * (log_p - self.priorlogprobs))

        # To verify with SciPy:
        # D = entropy(self.probdist(), np.exp(self.priorlogprobs))
        # assert np.allclose(D, divergence)
        return divergence

    def _check_features(self):
        """Validate whether the feature matrix has been set properly."""
        # Ensure the feature matrix for the sample space has been set
        if not hasattr(self, "F"):
            raise AttributeError("missing feature matrix")
        assert self.F.ndim == 2
        try:
            assert self.F.shape[0] == len(self.samplespace)
        except Exception:
            raise AttributeError(
                "the feature matrix is incompatible with the sample space. The number of columns must equal len(self.samplespace)"
            )

        blank = False
        if hasattr(self.F, "nnz"):
            if self.F.nnz == 0:
                blank = True
        else:
            if (self.F == 0).all():
                blank = True
        if blank:
            raise ValueError(
                "the feature matrix is zero. Check whether your feature functions are vectorized and, if not, pass vectorized=False"
            )

        # Watch out: if self.F is a dense NumPy matrix, its dot product
        # with a 1d parameter vector comes out as a 2d array, whereas if
        # self.F is a SciPy sparse matrix or dense NumPy array, its dot
        # product with the parameters is 1d. So, if it's a matrix, we cast
        # it to an array.
        if isinstance(self.F, np.matrix):
            self.F = np.asarray(self.F)

    def show_dist(self, max_output_lines=20):
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
        if n < max_output_lines:
            show_x_and_px_values(0, n)
        else:
            # Show the first e.g. 10 values, then ..., then the last 10 values
            show_x_and_px_values(0, max_output_lines // 2)
            print("\t...")
            show_x_and_px_values(n - max_output_lines // 2, n)


class SamplingMinKLDensity(BaseEstimator, DensityMixin, BaseMinKLDensity):
    """
    A minimum KL-divergence / maximum-entropy (exponential-form) model
    on a continuous or large discrete sample space requiring Monte Carlo
    simulation.

    Model feature expectations are computed iteratively using importance
    sampling.

    The model feature expectations are not computed exactly (by summing or
    integrating over a sample space) but approximately (by Monte Carlo
    estimation).  Approximation is necessary when the sample space is too
    large to sum or integrate over in practice, like a continuous sample
    space in more than about 4 dimensions or a large discrete space like
    all possible sentences in a natural language.

    Approximating the expectations by sampling requires an instrumental
    distribution that should be close to the model for fast convergence.
    The tails should be fatter than the model.

    This auxiliary or instrumental distribution is specified in the
    constructor.

    Sets up a generator for feature matrices internally from a list of feature
    functions.

    Parameters
    ----------
    feature_functions : list of functions
        Each feature function f_i (from i=1 to i=m) must operate on a vector of
        samples xs = {x1,...,xn}, either real data or samples generated by the
        auxiliary sampler.

        If your feature functions are not vectorized, you can wrap them in calls
        to np.vectorize(f_i) or pass vectorized=False, but beware the
        performance overhead.

    auxiliary_sampler : generator

        Pass auxiliary_sampler as a generator that will be used for importance
        sampling. Calling `next(auxiliary_sampler)` should return a tuple
            (xs, log_q_xs)
        representing:

            xs: a sample x_1,...,x_n to use for importance sampling

            log_q_xs: an array of length n containing the (natural) log
                      probability density (pdf or pmf) of each point under the
                      auxiliary sampling distribution.

    prior_log_pdf : None (default) or function
        Do you seek to minimize the KL divergence between the model and a
        prior density p_0?  If not, set this to None; then we maximize
        the Shannon information entropy H(p).

        If so, set this to a function that can take an array of values xs
        and return an array of the log probability densities p_0(x) for
        each x in the sample space.  For models involving simulation, set
        this to a function that should return p_0(x) for each x in the
        random sample from the auxiliary distribution.

        In both cases the minimization / maximization are done subject to the
        same constraints on feature expectations.

    For other parameters, see notes in the BaseModel docstring:
    - algorithm
    - matrix_format
    - verbose


    Algorithms
    ----------
    The algorithm can be 'CG', 'BFGS', 'LBFGSB', 'Powell', or
    'Nelder-Mead'.

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

    def __init__(
        self,
        feature_functions,
        auxiliary_sampler,
        prior_log_pdf=None,
        *,
        vectorized=True,
        matrix_format="csc_matrix",
        algorithm="CG",
        max_iter=1000,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            feature_functions=feature_functions,
            prior_log_pdf=prior_log_pdf,
            vectorized=vectorized,
            matrix_format=matrix_format,
            algorithm=algorithm,
            max_iter=max_iter,
            warm_start=warm_start,
            verbose=verbose,
        )
        self.auxiliary_sampler = auxiliary_sampler

        # We require that auxiliary_sampler be a generator:
        assert isinstance(auxiliary_sampler, types.GeneratorType)

        self.sampleFgen = feature_sampler(
            feature_functions,
            auxiliary_sampler,
            vectorized=vectorized,
            matrix_format=matrix_format,
        )

        # Number of sample matrices to generate and use each iteration to estimate E and logZ.
        # Normally this will be 1. Setting this > 1 would be much slower but could offer
        # more accurate estimates toward the end of the fitting process, when we
        # are near convergence.
        self.matrixtrials = 1

        # Store the lowest dual estimate observed so far in the fitting process
        self.bestdual = float("inf")

        # Whether or not to use the same sample for all iterations
        self.staticsample = True
        # If matrixtrials > 1 and staticsample = True, (which is useful for
        # estimating variance between the different feature estimates),
        # next(self.samplerFgen) will be called once for each trial
        # (0,...,matrixtrials) for each iteration.  This allows using a set
        # of feature matrices, each of which stays constant over all
        # iterations.

        # Test for convergence every 'testevery' iterations, using one or
        # more external samples. If 0, don't test.
        self.testevery = 0

        # self.priorlogprobs will be set by resample()
        self.resample()

    def _check_features(self):
        """
        Validation of whether the feature matrix has been set properly
        """
        # Ensure the sample matrix has been set
        if not (hasattr(self, "sample_F") and hasattr(self, "sample_log_probs")):
            raise AttributeError("first specify a sample feature matrix")

    def resample(self):
        """
        (Re)sample the matrix F of sample features, sample log probs, and
        (optionally) sample points too.
        """

        if self.verbose >= 2:
            print("Sampling...")

        # First delete the existing sample matrix to save memory
        # This matters, since these can be very large
        if hasattr(self, "sample_F"):
            del self.sample_F
        if hasattr(self, "sample_log_probs"):
            del self.sample_log_probs
        if hasattr(self, "sample"):
            del self.sample

        # Now generate a new sample. Assume the format is (F, lp, sample):
        (self.sample_F, self.sample_log_probs, self.sample) = next(self.sampleFgen)

        # Evaluate the prior log probabilities on the sample (for KL div
        # minimization)
        if self.prior_log_pdf is not None:
            lp = self.prior_log_pdf(self.sample)
            self.priorlogprobs = np.reshape(lp, len(self.sample))

        n, m = self.sample_F.shape
        if self.params is None:
            self.params = np.zeros(m, np.float64)
        else:
            # Check whether the number m of features and the dimensionalities are correct.
            # The number of features is defined as the length of # self.params.
            if m != len(self.params):
                raise ValueError(
                    "the sample feature generator returned"
                    " a feature matrix of incorrect dimensions."
                    " The number of rows must equal the number of model parameters."
                )

        # Check the dimensionality of sample_log_probs is correct. It should be 1d, of length n
        if not (
            isinstance(self.sample_log_probs, np.ndarray)
            and self.sample_log_probs.shape == (n,)
        ):
            raise ValueError(
                "Your sampler appears to be spitting out logprobs of the wrong dimensionality."
            )

        if self.verbose >= 2:
            print("Finished sampling.")

        # Now clear the temporary variables that are no longer correct for this
        # sample
        self.clearcache()

    def log_norm_constant(self):
        """Estimate the normalization constant (partition function) using
        the current sample matrix F.
        """
        # First see whether logZ has been precomputed
        if hasattr(self, "logZapprox"):
            return self.logZapprox

        # Compute log v = log [p_dot(s_j)/aux_dist(s_j)]   for
        # j=1,...,n=|sample| using a precomputed matrix of sample
        # features.
        logv = self._logv()

        # Good, we have our logv.  Now:
        n = len(logv)
        self.logZapprox = logsumexp(logv) - math.log(n)
        return self.logZapprox

    def feature_expectations(self):
        """
        Estimate the feature expectations (generalized "moments") E_p[f(X)]
        under the current model p = p_theta using the given sample feature
        matrix.

        If self.staticsample is True, uses the current feature matrix
        self.sample_F.  If self.staticsample is False or self.matrixtrials
        is > 1, draw one or more sample feature matrices F afresh using
        the generator function self.sampleFgen().
        """
        # See if already computed
        if hasattr(self, "mu"):
            return self.mu
        self.estimate()
        return self.mu

    def _logv(self):
        """This function helps with caching of interim computational
        results.  It is designed to be called internally, not by a user.

        Returns
        -------
        logv : 1d ndarray
               The array of unnormalized importance sampling weights
               corresponding to the sample x_j whose features are represented
               as the columns of self.sample_F.

               Defined as:

                   logv_j = p_dot(x_j) / q(x_j),

               where p_dot(x_j) = p_0(x_j) exp(theta . f(x_j)) is the
               unnormalized pdf value of the point x_j under the current model.
        """
        # First see whether logv has been precomputed
        if hasattr(self, "logv"):
            return self.logv

        # Compute log v = log [p_dot(s_j)/aux_dist(s_j)]   for
        # j=1,...,n=|sample| using a precomputed matrix of sample
        # features.
        if self.external is None:
            paramsdotF = self.sample_F.dot(self.params)
            logv = paramsdotF - self.sample_log_probs
            # Are we minimizing KL divergence between the model and a prior
            # density p_0?
            if self.priorlogprobs is not None:
                logv += self.priorlogprobs
        else:
            e = self.external
            paramsdotF = self.external_Fs[e].dot(self.params)
            logv = paramsdotF - self.external_logprobs[e]
            # Are we minimizing KL divergence between the model and a prior
            # density p_0?
            if self.external_priorlogprobs is not None:
                logv += self.external_priorlogprobs[e]

        # Good, we have our logv.  Now:
        self.logv = logv
        return logv

    def estimate(self):
        """
        Approximate both the feature expectation vector E_p f(X) and the log
        of the normalization term Z with importance sampling.

        This function also computes the sample variance of the component
        estimates of the feature expectations as: varE = var(E_1, ..., E_T)
        where T is self.matrixtrials and E_t is the estimate of E_p f(X)
        approximated using the 't'th auxiliary feature matrix.

        It doesn't return anything, but stores the member variables
        logZapprox, mu and varE.  (This is done because some optimization
        algorithms retrieve the dual fn and gradient fn in separate
        function calls, but we can compute them more efficiently
        together.)

        It uses a supplied generator whose __next__() method
        returns features of random observations s_j generated according
        to an auxiliary distribution aux_dist.  It uses these either in a
        matrix (with multiple runs) or with a sequential procedure, with
        more updating overhead but potentially stopping earlier (needing
        fewer samples).  In the matrix case, the features F={f_i(s_j)}
        and vector [log_aux_dist(s_j)] of log probabilities are generated
        by calling resample().

        We use [Rosenfeld01Wholesentence]'s estimate of E_p[f_i] as:
            {sum_j  p(s_j)/aux_dist(s_j) f_i(s_j) }
              / {sum_j p(s_j) / aux_dist(s_j)}.

        Note that this is consistent but biased.

        This equals:
            {sum_j  p_dot(s_j)/aux_dist(s_j) f_i(s_j) }
              / {sum_j p_dot(s_j) / aux_dist(s_j)}

        Compute the estimator E_p f_i(X) in log space as:
            num_i / denom,
        where
            num_i = exp(logsumexp(theta.f(s_j) - log aux_dist(s_j)
                        + log f_i(s_j)))
        and
            denom = [n * Zapprox]

        where Zapprox = exp(self.log_norm_constant()).

        We can compute the denominator n*Zapprox directly as:
            exp(logsumexp(log p_dot(s_j) - log aux_dist(s_j)))
          = exp(logsumexp(theta.f(s_j) - log aux_dist(s_j)))
        """

        if self.verbose >= 3:
            print("(estimating dual and gradient ...)")

        # Hereafter is the matrix code

        mus = []
        logZs = []

        for trial in range(self.matrixtrials):
            if self.verbose >= 2 and self.matrixtrials > 1:
                print("(trial " + str(trial) + " ...)")

            # Resample if necessary
            if (not self.staticsample) or self.matrixtrials > 1:
                self.resample()

            logv = self._logv()
            n = len(logv)
            logZ = self.log_norm_constant()
            logZs.append(logZ)

            # We don't need to handle negative values separately,
            # because we don't need to take the log of the feature
            # matrix sample_F. See Ed Schofield's PhD thesis, Section 4.4

            logu = logv - logZ
            if self.external is None:
                averages = self.sample_F.T.dot(np.exp(logu))
            else:
                averages = self.external_Fs[self.external].T.dot(np.exp(logu))
            averages /= n
            mus.append(averages)

        # Now we have T=trials vectors of the sample means.  If trials > 1,
        # estimate st dev of means and confidence intervals
        ttrials = len(mus)  # total number of trials performed
        if ttrials == 1:
            self.mu = mus[0]
            self.logZapprox = logZs[0]
            try:
                del self.varE  # make explicit that this has no meaning
            except AttributeError:
                pass
        else:
            # The log of the variance of logZ is:
            #     -log(n-1) + logsumexp(2*log|Z_k - meanZ|)

            self.logZapprox = logsumexp(logZs) - math.log(ttrials)
            # stdevlogZ = np.array(logZs).std()
            mus = np.array(mus)
            self.varE = mus.var(axis=0)  # column variances
            self.mu = mus.mean(axis=0)  # column means

    # def pdf_from_features(self, fx, *, log_prior_x=None):
    #     """Returns the estimated density p_theta(x) at the point x with
    #     feature statistic fx = f(x).  This is defined as
    #         p_theta(x) = exp(theta.f(x)) / Z(theta),
    #     where Z is the estimated value self.norm_constant() of the partition
    #     function.
    #     """
    #     return np.exp(self.log_pdf_from_features(fx, log_prior_x=log_prior_x))

    def pdf_function(self):
        """Returns the estimated density p_theta(x) as a function p(f)
        taking a vector f = f(x) of feature statistics at any point x.
        This is defined as:
            p_theta(x) = exp(theta.f(x)) / Z
        """
        log_Z_est = self.log_norm_constant()

        def p(fx, *, log_prior_x=None):
            if log_prior_x is not None:
                raise NotImplementedError("fix me!")
            return np.exp(fx.dot(self.params) - log_Z_est)

        return p

    # def log_pdf_from_features(self, fx, *, log_prior_x=None):
    #     """Returns the log of the estimated density p(x) = p_theta(x) at
    #     the point x.  This is defined as:
    #         log p(x) = log p0(x) + theta.f(x) - log Z
    #     where f(x) is given by the (m x 1) array fx.

    #     If, instead, fx is a 2-d (n x m) array, this function interprets
    #     each of its rows j=0,...,n-1 as a feature vector f(x_j), and
    #     returns an array containing the log pdf value of each point x_j
    #     under the current model.

    #     log Z is estimated using the auxiliary sampler provided.

    #     The optional argument log_prior_x is the log of the prior density
    #     p_0 at the point x (or at each point x_j if fx is 2-dimensional).
    #     The log pdf of the model is then defined as
    #         log p(x) = log p0(x) + theta.f(x) - log Z
    #     and p then represents the model of minimum KL divergence D(p||p0)
    #     instead of maximum entropy.
    #     """
    #     log_Z_est = self.log_norm_constant()
    #     if len(fx.shape) == 1:
    #         log_pdf = np.dot(self.params, fx) - log_Z_est
    #     else:
    #         log_pdf = fx.dot(self.params) - log_Z_est
    #     if self.prior_log_pdf is not None:
    #         # We expect log_prior_x to be passed in
    #         if log_prior_x is None:
    #             raise ValueError(
    #                 "It appears your model was fitted to minimize "
    #                 "KL divergence but no log_prior_x value is being passed in. "
    #                 "This would incorrectly calculate the pdf; it would not "
    #                 "integrate to 1."
    #             )
    #         log_pdf += log_prior_x
    #     return log_pdf

    def settestsamples(self, F_list, logprob_list, testevery=1, priorlogprob_list=None):
        """Requests that the model be tested every 'testevery' iterations
        during fitting using the provided list F_list of feature
        matrices, each representing a sample {x_j} from an auxiliary
        distribution q, together with the corresponding log probabiltiy
        mass or density values log {q(x_j)} in logprob_list.  This is
        useful as an external check on the fitting process with sample
        path optimization, which could otherwise reflect the vagaries of
        the single sample being used for optimization, rather than the
        population as a whole.

        If self.testevery > 1, only perform the test every self.testevery
        calls.

        If priorlogprob_list is not None, it should be a list of arrays
        of log(p0(x_j)) values, j = 0,. ..., n - 1, specifying the prior
        distribution p0 for the sample points x_j for each of the test
        samples.
        """
        # Sanity check
        assert len(F_list) == len(logprob_list)

        self.testevery = testevery
        self.external_Fs = F_list
        self.external_logprobs = logprob_list
        self.external_priorlogprobs = priorlogprob_list

        # Store the dual and mean square error based on the internal and
        # external (test) samples.  (The internal sample is used
        # statically for sample path optimization; the test samples are
        # used as a control for the process.)  The hash keys are the
        # number of function or gradient evaluations that have been made
        # before now.

        # The mean entropy dual and mean square error estimates among the
        # t external (test) samples, where t = len(F_list) =
        # len(logprob_list).
        self.external_duals = {}
        self.external_gradnorms = {}

    def test(self):
        """Estimate the dual and gradient on the external samples,
        keeping track of the parameters that yield the minimum such dual.
        The vector of desired (target) feature expectations is stored as
        self.K.
        """
        if self.verbose:
            print("  max(params**2)    = " + str((self.params**2).max()))

        if self.verbose:
            print("Now testing model on external sample(s) ...")

        # Estimate the entropy dual and gradient for each sample.  These
        # are not regularized (smoothed).
        dualapprox = []
        gradnorms = []
        for e in range(len(self.external_Fs)):
            self.external = e
            self.clearcache()
            if self.verbose >= 2:
                print("(testing with sample %d)" % e)
            dualapprox.append(self.dual(ignorepenalty=True, ignoretest=True))
            gradnorms.append(norm(self.grad(ignorepenalty=True)))

        # Reset to using the normal sample matrix sample_F
        self.external = None
        self.clearcache()

        meandual = np.average(dualapprox, axis=0)
        self.external_duals[self.n_iter_] = dualapprox
        self.external_gradnorms[self.n_iter_] = gradnorms

        if self.verbose:
            print(
                "** Mean (unregularized) dual estimate from the %d"
                " external samples is %f" % (len(self.external_Fs), meandual)
            )
            print(
                "** Mean mean square error of the (unregularized) feature"
                " expectation estimates from the external samples ="
                r" mean(|| \hat{\mu_e} - k ||,axis=0) =",
                np.average(gradnorms, axis=0),
            )
        # Track the parameter vector params with the lowest mean dual estimate
        # so far:
        if meandual < self.bestdual:
            self.bestdual = meandual
            self.bestparams = self.params
            if self.verbose:
                print("\n\t\t\tStored new minimum entropy dual: %f\n" % meandual)


class MinKLClassifier(ClassifierMixin, BaseEstimator):
    """
    Parameters
    ----------
        prior_log_proba_fn: function
            Pass a function that takes an (n, m) array X and returns a matrix of
            log probability density values of shape (n, d), where d is the
            number of classes. An example is the `predict_log_proba()` methods
            of scikit-learn classifiers. This will be evaluated on the samples
            produced by `auxiliary_sampler`.
    """

    def __init__(
        self,
        feature_functions,
        auxiliary_sampler,
        prior_log_proba_fn=None,
        *,
        vectorized=True,
        matrix_format="csc_matrix",
        algorithm="CG",
        max_iter=1000,
        warm_start=False,
        verbose=0,
    ):
        self.models = {}
        self.feature_functions = feature_functions
        self.auxiliary_sampler = auxiliary_sampler
        self.prior_log_proba_fn = prior_log_proba_fn
        self.vectorized = vectorized
        self.matrix_format = matrix_format
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the baseline classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, cast_to_ndarray=True, accept_sparse=["csr", "csc"])
        y = self._validate_data(y=y)

        check_classification_targets(y)
        # Handle non-contiguous output labels y:
        self.classes_, y = np.unique(y, return_inverse=True)

        prior_log_pdfs = {}

        for target_class in range(len(self.classes_)):
            if self.prior_log_proba_fn is None:
                prior_log_pdfs[target_class] = None
            else:
                prior_log_pdfs[target_class] = lambda X: (
                    self.prior_log_proba_fn(X)[:, target_class]
                )
            self.models[target_class] = SamplingMinKLDensity(
                self.feature_functions,
                self.auxiliary_sampler,
                prior_log_pdf=prior_log_pdfs[target_class],
                vectorized=self.vectorized,
                matrix_format=self.matrix_format,
                algorithm=self.algorithm,
                max_iter=self.max_iter,
                warm_start=self.warm_start,
                verbose=self.verbose,
            )

        for target, model in self.models.items():
            # Filter the rows of X to those whose corresponding y matches the target class:
            X_subset = X[y == target]
            F = evaluate_feature_matrix(
                model.feature_functions, X_subset, matrix_format=self.matrix_format
            )
            k = np.asarray(F.mean(axis=0))
            if self.verbose:
                print(f"Fitting model for target {target}")
            model.fit(k)

        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self

    def predict_log_proba(self, X):
        """
        The log probability of the true model being for each target class of
        those fitted.

        The probability vector for the m classes is defined as:

            log [ (p_1(x), ..., p_m(x)) / sum p_i(x) ]

            which can be reexpressed as:
            (log p_1(x), ..., log p_m(x)) - logsumexp p_i(x)

            where the p_i values are the values of the probability density (or
            mass) functions (not necessarily normalized) for the m component
            models.
        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        log_proba = np.array(
            [model.predict_log_proba(X) for model in self.models.values()]
        ).T
        # We want this logically but it gives a shape error under NumPy's
        # broadcasting rules:
        # log_proba -= logsumexp(log_proba, axis=1)
        log_proba = (log_proba.T - logsumexp(log_proba, axis=1)).T
        return log_proba

    def predict_proba(self, X):
        """
        The probability of the true model being for each target class of
        those fitted.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        predictions = self.classes_[np.argmax(log_proba, axis=1)]
        return predictions

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


__all__ = ["MinKLDensity", "SamplingMinKLDensity", "MinKLClassifier"]
