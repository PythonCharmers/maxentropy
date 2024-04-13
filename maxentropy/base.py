import pickle
from abc import ABCMeta, abstractmethod
from collections.abc import Iterator
import types

from sklearn.base import BaseEstimator, DensityMixin
import numpy as np
from scipy import optimize
from scipy.linalg import norm
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.metrics import mean_squared_error

from maxentropy.utils import (
    DivergenceError,
    evaluate_feature_matrix,
)


class BaseMinDivergenceDensity(DensityMixin, BaseEstimator, metaclass=ABCMeta):
    """A base class providing generic functionality for Minimum KL divergence
    models using either exact summation or sampling. Cannot be instantiated.

    Parameters
    ----------
    feature_functions : list of functions
        Each feature function f_i from i=1 to i=m must operate on a vector of
        samples xs = {x1,...,xn}, either real data or samples generated by an
        auxiliary sampler.

        Your feature functions are expected to be vectorized for good
        performance. If they are not, pass vectorized=False.

    prior_log_pdf : None (default) or function
        Do you seek to minimize the KL divergence between the model and a
        prior density p_0?  If not, set this to None; then we maximize
        the Shannon information entropy H(p).

        If so, set this to a function that can take an array of values X of
        shape (k x m) and return an array of the log probability densities
        p_0(x) under the prior p_0 for each (row vector) x in the sample space.

        (It may be a good idea to make prior_log_pdf squeeze its output, so we
        get e.g.  shape (k,) instead of (k, 1).)

        For models involving simulation, set this to a function
        that should return p_0(x) for each x in the random sample produced by
        the auxiliary distribution.

        In both cases the minimization / maximization are done subject to the
        same constraints on feature expectations.

    vectorized : bool (default True)
        If True, the feature functions f_i(xs) are assumed to be "vectorized",
        meaning that each is assumed to accept a sequence of values xs = (x_1,
        ..., x_n) at once and each return a vector of length n.

        If False, the feature functions f_i(x) take individual values x on the
        sample space and return real values. This is likely to be slow down
        computing the features significantly.

    array_format : string
        Currently 'csr_array', 'csc_array', and 'ndarray' are recognized.

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

    smoothing_factor : float or ndarray or None
        Set this to a positive scalar or vector value (of the same length as the
        parameter vector) for Gaussian regularization / smoothing on the
        parameters. Penalizes parameter values far from zero. Smoothing is as
        described in:

           Chen, Stanley F. and Rosenfeld, Ronald A Gaussian prior for smoothing
           maximum entropy models. (CMU-CS-99-108), Carnegie Mellon University
           (1999).

    """

    def __init__(
        self,
        feature_functions: list[types.FunctionType],
        *,
        prior_log_pdf=None,
        vectorized=True,
        array_format="csr_array",
        algorithm="CG",
        maxgtol=1e-7,
        avegtol=1e-7,
        tol=1e-8,
        max_iter=1000,
        warm_start=False,
        verbose=0,
        smoothing_factor=None,
    ):
        self.feature_functions = feature_functions
        self.prior_log_pdf = prior_log_pdf
        self.vectorized = vectorized
        self.array_format = array_format
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose
        self.maxgtol = maxgtol

        # Required tolerance of gradient on average (closeness to zero,axis=0)
        # for CG optimization:
        self.avegtol = avegtol

        # Default tolerance for the other optimization algorithms:
        self.tol = tol

        # Variances for a Gaussian prior on the parameters for smoothing.
        # Penalizes parameter values far from zero:
        self.smoothing_factor = smoothing_factor

    def _validate_and_setup(self):
        self.resetparams()

        self.features = lambda xs: evaluate_feature_matrix(
            self.feature_functions,
            xs,
            vectorized=self.vectorized,
            array_format=self.array_format,
        )

        # TODO: It would be nice to validate that prior_log_pdf is a
        # function. But a function passed into the numpy vectorize decorator
        # is no longer an instance of FunctionType.
        # Would this work?
        # assert isinstance(prior_log_pdf, (types.FunctionType,
        #                                   types.MethodType))
        # TODO: ensure it's vectorized

        if self.prior_log_pdf is None:
            self.priorlogprobs = None

        if self.array_format not in {
            "csr_array",
            "csc_array",
            "ndarray",
        }:
            raise ValueError("array format not understood")

        # Clear the stored duals and gradient norms
        self.duals = {}
        self.gradnorms = {}
        if hasattr(self, "external_duals"):
            self.external_duals = {}
        if hasattr(self, "external_gradnorms"):
            self.external_gradnorms = {}
        if hasattr(self, "external"):
            self.external = None

        self.callingback = False

        # Default tolerance for stochastic approximation: stop if
        # ||params_k - params_{k-1}|| < paramstol:
        self.paramstol = 1e-5

        self.maxfun = 1500
        self.mindual = -100.0  # The entropy dual must actually be
        # non-negative, but the estimate may be slightly
        # out with BigModel instances without implying
        # divergence to -inf
        self.callingback = False
        self.n_iter_ = 0  # the number of iterations so far of the
        # optimization algorithm
        self.fnevals = 0
        self.gradevals = 0

        # Store the duals for each fn evaluation during fitting?
        self.storeduals = False
        self.duals = {}
        self.storegradnorms = False
        self.gradnorms = {}

        # By default, use the sample matrix sampleF to estimate the entropy dual
        # and its gradient.  Otherwise, set self.external to the index of the
        # sample feature matrix in the list self.externalFs.  This applies to
        # 'BigModel' objects only, but setting this here simplifies the code in
        # dual() and grad().
        self.external = None
        self.external_priorlogprobs = None

    @abstractmethod
    def _setup_features(self, *args, **kwargs):
        """
        Set up samplers and create a 2d array of features
        """

    def fit_expectations(self, k):
        """Fit the model of minimum divergence / maximum entropy subject to
        constraints on the feature expectations <f_i(X)> = k[0].

        Parameters
        ----------
        k : ndarray (dense) of shape n_features or (1 x n_features)
            A row vector representing desired expectations of features.  The
            curious shape is deliberate: models of minimum divergence / maximum
            entropy depend on the data only through the feature expectations.

        Returns
        -------
        self
        """
        # TODO: simplify this!
        K = np.atleast_2d(k)
        K = check_array(K)
        n_samples = K.shape[0]
        if n_samples != 1:
            raise ValueError("k must have only one row")

        self._validate_and_setup()
        self._setup_features()

        # if not (hasattr(self, "F") or hasattr(self, "sample_F")):
        #     raise ValueError("Call _setup_features before calling fit_expectations")

        # Extract a 1d array of the feature expectations
        # K = np.asarray(X[0], float)
        K = K[0, :]
        assert K.ndim == 1

        # Store the desired feature expectations as a member variable
        self.K = K

        if (not self.warm_start) or not hasattr(self, "params"):
            self.resetparams()
        # Sanity check:
        if len(self.params) != len(K):
            raise ValueError(
                "the number of target expectations does not match the number of features. We need len(np.squeeze(X)) == len(features)."
            )

        # Make a copy of the parameters
        self.oldparams = self.params.copy()

        callback = self.log

        retval = optimize.minimize(
            self.dual,
            self.oldparams,
            args=(),
            method=self.algorithm,
            jac=self.grad,
            tol=self.tol,
            options={"maxiter": self.max_iter, "disp": self.verbose},
            callback=callback,
        )
        newparams = retval.x
        func_calls = retval.nfev

        if np.any(self.params != newparams):
            self.setparams(newparams)
        self.func_calls = func_calls

        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
        return self

    def fit(self, X, y=None):
        """Fit the model of minimum divergence / maximum entropy subject to
        constraints on the feature expectations <f_i(X)> = X[0].

        Parameters
        ----------
        X : ndarray (dense) of shape (n_observations, n_features)
            A matrix representing desired expectations of features.

        y : is not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        self

        """
        X = self._validate_data(X, cast_to_ndarray=False, accept_sparse=["csr", "csc"])

        # We require that auxiliary_sampler be a generator:
        if hasattr(self, "auxiliary_sampler"):
            if not isinstance(self.auxiliary_sampler, Iterator):
                raise ValueError("Pass a generator as your `auxiliary_sampler`.")

        self._setup_features()

        F = evaluate_feature_matrix(
            self.feature_functions, X, array_format=self.array_format
        )
        k = np.asarray(F.mean(axis=0))
        self.fit_expectations(k)

        return self

    def dual(self, params=None, ignorepenalty=False, ignoretest=False):
        """Computes the Lagrangian dual L(theta) of the entropy of the
        model, for the given vector theta=params.  Minimizing this
        function (without constraints) should fit the maximum entropy
        model subject to the given constraints.  These constraints are
        specified as the desired (target) values self.K for the
        expectations of the feature statistic.

        This function is computed as:
            L(theta) = log(Z) - theta^T . K

        For 'BigModel' objects, it estimates the entropy dual without
        actually computing p_theta.  This is important if the sample
        space is continuous or innumerable in practice.  We approximate
        the norm constant Z using importance sampling as in
        [Rosenfeld01whole].  This estimator is deterministic for any
        given sample.  Note that the gradient of this estimator is equal
        to the importance sampling *ratio estimator* of the gradient of
        the entropy dual [see my thesis], justifying the use of this
        estimator in conjunction with grad() in optimization methods that
        use both the function and gradient. Note, however, that
        convergence guarantees break down for most optimization
        algorithms in the presence of stochastic error.

        Note that, for 'BigModel' objects, the dual estimate is
        deterministic for any given sample.  It is given as:

            L_est = log Z_est - sum_i{theta_i K_i}

        where
            Z_est = 1/m sum_{x in sample S_0} p_dot(x) / aux_dist(x),

        and m = # observations in sample S_0, and K_i = the empirical
        expectation E_p_tilde f_i (X) = sum_x {p(x) f_i(x)}.
        """

        if self.external is None and not self.callingback:
            if self.verbose >= 2:
                print("Function eval #", self.fnevals)

        if params is not None:
            self.setparams(params)

        if not hasattr(self, "K"):
            raise ValueError(
                "the entropy dual is a function of "
                "the target feature expectations. "
                "Set these first by calling `fit`."
            )

        # Subsumes both small and large cases:
        L = self.log_norm_constant() - np.dot(self.params, self.K)

        if np.isnan(L):
            raise ValueError("Oops: the dual is nan! Debug me!")

        if self.verbose >= 2 and self.external is None:
            print("  dual is ", L)

        # Use a Gaussian prior for smoothing if requested.
        # This adds the penalty term \sum_{i=1}^m \params_i^2 / {2 \sigma_i^2}.
        # Define 0 / 0 = 0 here; this allows a variance term of
        # sigma_i^2==0 to indicate that feature i should be ignored.
        if self.smoothing_factor is not None and not ignorepenalty:
            ratios = np.nan_to_num(self.params**2 / self.smoothing_factor)
            # Why does the above convert inf to 1.79769e+308?

            L += 0.5 * ratios.sum()
            if self.verbose >= 2 and self.external is None:
                print("  regularized dual is ", L)

        if not self.callingback and self.external is None:
            if hasattr(self, "callback_dual") and self.callback_dual is not None:
                # Prevent infinite recursion if the callback function
                # calls dual():
                self.callingback = True
                self.callback_dual(self)
                self.callingback = False

        if self.external is None and not self.callingback:
            self.fnevals += 1

        # (We don't reset self.params to its prior value.)
        return L

    # An alias for the dual function:
    entropydual = dual

    def log(self, params):
        """This method is called every iteration during the optimization
        process.  It calls the user-supplied callback function (if any),
        logs the evolution of the entropy dual and gradient norm, and
        checks whether the process appears to be diverging, which would
        indicate inconsistent constraints (or, for BigModel instances,
        too large a variance in the estimates).
        """

        if self.external is None and not self.callingback:
            if self.verbose >= 2:
                print("Iteration #", self.n_iter_)

        # Store new dual and/or gradient norm
        if not self.callingback:
            if self.storeduals:
                self.duals[self.n_iter_] = self.dual()
            if self.storegradnorms:
                self.gradnorms[self.n_iter_] = norm(self.grad())

        if not self.callingback and self.external is None:
            if hasattr(self, "callback"):
                # Prevent infinite recursion if the callback function
                # calls dual():
                self.callingback = True
                self.callback(self)
                self.callingback = False

        # Do we perform a test on external sample(s) every iteration?
        # Only relevant to BigModel objects
        if hasattr(self, "testevery") and self.testevery > 0:
            if (self.n_iter_ + 1) % self.testevery != 0:
                if self.verbose:
                    print("Skipping test on external sample(s) ...")
            else:
                self.test()

        if not self.callingback and self.external is None:
            if self.mindual > -np.inf and self.dual() < self.mindual:
                raise DivergenceError(
                    "dual is below the threshold 'mindual'"
                    " and may be diverging to -inf.  Fix the constraints"
                    " or lower the threshold!"
                )

        self.n_iter_ += 1

    def grad(self, params=None, ignorepenalty=False):
        """Computes or estimates the gradient of the entropy dual."""

        if self.verbose >= 2 and self.external is None and not self.callingback:
            print("Grad eval #" + str(self.gradevals))

        if params is not None:
            self.setparams(params)

        if not hasattr(self, "K"):
            raise ValueError(
                "the gradient of the entropy dual is "
                "a function of the target feature "
                "expectations. Set these first by "
                "calling `fit`."
            )

        G = self.feature_expectations() - self.K

        if self.verbose >= 2 and self.external is None:
            print("  norm of gradient =", norm(G))

        # (We don't reset params to its prior value.)

        # Use a Gaussian prior for smoothing if requested.  The ith
        # partial derivative of the penalty term is \params_i /
        # \sigma_i^2.  Define 0 / 0 = 0 here; this allows a variance term
        # of sigma_i^2==0 to indicate that feature i should be ignored.
        if self.smoothing_factor is not None and not ignorepenalty:
            penalty = self.params / self.smoothing_factor
            G += penalty
            features_to_kill = np.where(np.isnan(penalty))[0]
            G[features_to_kill] = 0.0
            if self.verbose >= 2 and self.external is None:
                normG = norm(G)
                print("  norm of regularized gradient =", normG)

        if not self.callingback and self.external is None:
            if hasattr(self, "callback_grad") and self.callback_grad is not None:
                # Prevent infinite recursion if the callback function
                # calls grad():
                self.callingback = True
                self.callback_grad(self)
                self.callingback = False

        if self.external is None and not self.callingback:
            self.gradevals += 1

        return G

    def mse(self):
        """
        Return the mean squared error of the desired versus estimated feature expectations.
        """
        return mean_squared_error(self.feature_expectations(), self.K)

    def cross_entropy(self, fx, log_prior_x=None, base=np.e):
        r"""Returns the cross entropy H(q, p) of the empirical
        distribution q of the data (with the given feature matrix fx)
        with respect to the model p.  For discrete distributions this is
        defined as:

            H(q, p) = - n^{-1} \sum_{j=1}^n log p(x_j)

        where x_j are the data elements assumed drawn from q whose
        features are given by the matrix fx = {f(x_j)}, j=1,...,n.

        The 'base' argument specifies the base of the logarithm, which
        defaults to e.

        For continuous distributions this makes no sense!
        """
        H = -self.logpdf(fx, log_prior_x).mean()
        if base != np.e:
            # H' = H * log_{base} (e)
            return H / np.log(base)
        else:
            return H

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns an array indexed by integers representing the logarithms of the
        probability mass function (pmf) for each X_j in the (n x m) array X
        under the current model (with the current parameter vector self.params).

        p(x) is given by:

          p(x) = exp(params . f(x)) / sum_y[ exp params . f(y) ]
               = exp[log p_dot(x) - logsumexp{ log p_dot(y) }]

        so log p(x) is given by:

          log p(x) = log p_dot(x) - logsumexp{ log p_dot(y) }

        If self.log_pdf is defined, we use it to evaluate the log of the prior density
        p_0 at the point x (or at each point x_j if fx is 2-dimensional).
        The log pdf of the model is then defined as

            log p(x) = log p0(x) + theta.f(x) - log Z

        and this model (with density p) then represents the model of minimum KL
        divergence D(p||p0) instead of maximum entropy.

        """
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # if not hasattr(self, "logZ"):
        #     # Compute the norm constant (quickly!)
        #     self.logZ = logsumexp(log_p_dot)

        log_Z = self.log_norm_constant()

        # Calculate the dot product of the feature matrix of the samples X_j in X and the parameter vector:
        fx = self.features(X)
        log_px = fx @ self.params - log_Z

        # Do we have a prior distribution p_0?
        if self.prior_log_pdf is not None:
            log_px += self.prior_log_pdf(X).squeeze()

        return log_px

    def predict_proba(self, X) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    @abstractmethod
    def log_norm_constant(self):
        """
        Subclasses must implement this.

        Return the log of the normalization constant, or log partition function, for
        the current model.

        For Monte Carlo cases, this method should estimate the normalization term as
        Z = E_aux_dist [{exp (params.f(X))} / aux_dist(X)] using a sample from
        aux_dist.
        """

    def norm_constant(self):
        """Return the normalization constant, or partition function, for
        the current model.  Warning -- this may be too large to represent;
        if so, this will result in numerical overflow.  In this case use
        log_norm_constant() instead.

        For Monte Carlo cases, this method estimates the normalization term as
        Z = E_aux_dist [{exp (params.f(X))} / aux_dist(X)] using a sample from
        aux_dist.
        """
        return np.exp(self.log_norm_constant())

    def setsmooth(self, sigma):
        """Specifies that the entropy dual and gradient should be
        computed with a quadratic penalty term on magnitude of the
        parameters.  This 'smooths' the model to account for noise in the
        target expectation values or to improve robustness when using
        simulation to fit models and when the sampling distribution has
        high variance.  The smoothing mechanism is described in Chen and
        Rosenfeld, 'A Gaussian prior for smoothing maximum entropy
        models' (1999).

        The parameter 'sigma' will be squared and stored as
        self.smoothing_factor.
        """
        self.smoothing_factor = sigma**2

    def setparams(self, params):
        """Set the parameter vector to params, replacing the existing
        parameters.  params must be a list or numpy array of the same
        length as the model's feature vector f.
        """

        self.params = np.array(params, float)  # make a copy

        # Log the new params to disk
        self.logparams()

        # Delete params-specific stuff
        self.clearcache()

    def clearcache(self):
        """Clears the interim results of computations depending on the
        parameters and the sample.
        """
        for var in ["mu", "logZ", "logZapprox", "log_w_dot_"]:
            if hasattr(self, var):
                delattr(self, var)

    def resetparams(self):
        """Reset the parameters self.params to zero, clearing the
        cache variables dependent on them.  Also reset the number of
        function and gradient evaluations to zero.
        """
        m = len(self.feature_functions)

        # Set parameters, clearing cache variables
        self.setparams(np.zeros(m, float))

        # These bounds on the param values are only effective for the
        # L-BFGS-B optimizer:
        self.bounds = [(-100.0, 100.0)] * len(self.params)

        self.fnevals = 0
        self.gradevals = 0
        self.n_iter_ = 0

    def setcallback(self, callback=None, callback_dual=None, callback_grad=None):
        """Sets callback functions to be called every iteration, every
        function evaluation, or every gradient evaluation. All callback
        functions are passed one argument, the current model object.

        Note that line search algorithms in e.g. CG make potentially
        several function and gradient evaluations per iteration, some of
        which we expect to be poor.
        """
        self.callback = callback
        self.callback_dual = callback_dual
        self.callback_grad = callback_grad

    def logparams(self):
        """Saves the model parameters if logging has been
        enabled and the # of iterations since the last save has reached
        self.paramslogfreq.
        """
        if not hasattr(self, "paramslogcounter"):
            # Assume beginlogging() was never called
            return
        self.paramslogcounter += 1
        if not (self.paramslogcounter % self.paramslogfreq == 0):
            return

        # Check whether the params are NaN
        if not np.all(self.params == self.params):
            raise FloatingPointError("some of the parameters are NaN")

        if self.verbose:
            print("Saving parameters ...")
        paramsfile = open(
            self.paramslogfilename + "." + str(self.paramslogcounter) + ".pickle", "wb"
        )
        pickle.dump(self.params, paramsfile, pickle.HIGHEST_PROTOCOL)
        paramsfile.close()
        # self.paramslog += 1
        # self.paramslogcounter = 0
        if self.verbose:
            print("Done.")

    def beginlogging(self, filename, freq=10):
        """Enable logging params for each fn evaluation to files named
        'filename.freq.pickle', 'filename.(2*freq).pickle', ... each
        'freq' iterations.
        """
        if self.verbose:
            print("Logging to files " + filename + "*")
        self.paramslogcounter = 0
        self.paramslogfilename = filename
        self.paramslogfreq = freq
        # self.paramslog = 1

    def endlogging(self):
        """Stop logging param values whenever setparams() is called."""
        del self.paramslogcounter
        del self.paramslogfilename
        del self.paramslogfreq


def _test():
    import doctest

    doctest.testmod()


__all__ = ["BaseMinDivergenceDensity"]


if __name__ == "__main__":
    _test()
