from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pickle
from abc import ABCMeta, abstractmethod
import types

import six
import numpy as np
from scipy import optimize
from scipy.linalg import norm
from sklearn.utils import check_array

from maxentropy.utils import DivergenceError


class BaseModel(six.with_metaclass(ABCMeta)):
    """A base class providing generic functionality for both small and
    large maximum entropy models.  Cannot be instantiated.

    Parameters
    ----------

    matrix_format : string
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

    """

    def __init__(self,
                 *,
                 prior_log_pdf=None,
                 algorithm='CG',
                 matrix_format='csr_matrix',
                 verbose=0):

        self.prior_log_pdf = prior_log_pdf
        if prior_log_pdf is not None:
            # Ensure it's a function
            assert isinstance(prior_log_pdf, (types.FunctionType,
                                              types.MethodType))
            # TODO: ensure it's vectorized
            raise NotImplementedError('fix me!')
        else:
            self.priorlogprobs = None
        self.algorithm = algorithm
        if matrix_format in ('csr_matrix', 'csc_matrix', 'ndarray'):
            self.matrix_format = matrix_format
        else:
            raise ValueError('matrix format not understood')
        self.verbose = verbose

        self.maxgtol = 1e-7

        # Required tolerance of gradient on average (closeness to zero,axis=0)
        # for CG optimization:
        self.avegtol = 1e-7

        # Default tolerance for the other optimization algorithms:
        self.tol = 1e-8

        # Default tolerance for stochastic approximation: stop if
        # ||params_k - params_{k-1}|| < paramstol:
        self.paramstol = 1e-5

        self.maxiter = 1000
        self.maxfun = 1500
        self.mindual = -100.    # The entropy dual must actually be
                                # non-negative, but the estimate may be slightly
                                # out with BigModel instances without implying
                                # divergence to -inf
        self.callingback = False
        self.iters = 0          # the number of iterations so far of the
                                # optimization algorithm
        self.fnevals = 0
        self.gradevals = 0

        # Variances for a Gaussian prior on the parameters for smoothing
        self.sigma2 = None

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


    def fit(self, X, y=None):
        """Fit the model of minimum divergence / maximum entropy subject to
        constraints on the feature expectations <f_i(X)> = X[0].

        Parameters
        ----------
        X : ndarray (dense) of shape [1, n_features]
            A row vector (1 x n_features matrix) representing desired
            expectations of features.  The curious shape is deliberate: models
            of minimum divergence / maximum entropy depend on the data only
            through the feature expectations.

        y : is not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        self

        """

        X = np.atleast_2d(X)
        X = check_array(X)
        n_samples = X.shape[0]
        if n_samples != 1:
            raise ValueError('X must have only one row')

        # Extract a 1d array of the feature expectations
        # K = np.asarray(X[0], float)
        K = X[0]
        assert K.ndim == 1

        # Store the desired feature expectations as a member variable
        self.K = K

        self._check_features()

        # Sanity checks
        try:
            self.params
        except AttributeError:
            self.resetparams(len(K))
        else:
            assert len(self.params) == len(K)

        # Don't reset the number of function and gradient evaluations to zero
        # self.fnevals = 0
        # self.gradevals = 0

        # Make a copy of the parameters
        oldparams = np.array(self.params)

        callback = self.log

        retval = optimize.minimize(self.dual, oldparams, args=(), method=self.algorithm,
                                   jac=self.grad, tol=self.tol,
                                   options={'maxiter': self.maxiter, 'disp': self.verbose},
                                   callback=callback)
        newparams = retval.x
        func_calls = retval.nfev

        # if self.algorithm == 'CG':
        #     retval = optimize.fmin_cg(self.dual, oldparams, self.grad, (), self.avegtol, \
        #                               maxiter=self.maxiter, full_output=1, \
        #                               disp=self.verbose, retall=0,
        #                               callback=callback)
        #
        #     (newparams, fopt, func_calls, grad_calls, warnflag) = retval
        #
        # elif self.algorithm == 'LBFGSB':
        #     if callback is not None:
        #         raise NotImplementedError("L-BFGS-B optimization algorithm"
        #                 " does not yet support callback functions for"
        #                 " testing with an external sample")
        #     retval = optimize.fmin_l_bfgs_b(self.dual, oldparams, \
        #                 self.grad, args=(), bounds=self.bounds, pgtol=self.maxgtol,
        #                 maxfun=self.maxfun)
        #     (newparams, fopt, d) = retval
        #     warnflag, func_calls = d['warnflag'], d['funcalls']
        #     if self.verbose:
        #         print(self.algorithm + " optimization terminated successfully.")
        #         print("\tFunction calls: " + str(func_calls))
        #         # We don't have info on how many gradient calls the LBFGSB
        #         # algorithm makes
        #
        # elif self.algorithm == 'BFGS':
        #     retval = optimize.fmin_bfgs(self.dual, oldparams, \
        #                                 self.grad, (), self.tol, \
        #                                 maxiter=self.maxiter, full_output=1, \
        #                                 disp=self.verbose, retall=0, \
        #                                 callback=callback)
        #
        #     (newparams, fopt, gopt, Lopt, func_calls, grad_calls, warnflag) = retval
        #
        # elif self.algorithm == 'Powell':
        #     retval = optimize.fmin_powell(self.dual, oldparams, args=(), \
        #                            xtol=self.tol, ftol = self.tol, \
        #                            maxiter=self.maxiter, full_output=1, \
        #                            disp=self.verbose, retall=0, \
        #                            callback=callback)
        #
        #     (newparams, fopt, direc, numiter, func_calls, warnflag) = retval
        #     # fmin_powell seems to turn newparams into a 0d array
        #     newparams = np.atleast_1d(newparams)
        #
        # elif self.algorithm == 'Nelder-Mead':
        #     retval = optimize.fmin(self.dual, oldparams, args=(), \
        #                            xtol=self.tol, ftol = self.tol, \
        #                            maxiter=self.maxiter, full_output=1, \
        #                            disp=self.verbose, retall=0, \
        #                            callback=callback)
        #
        #     (newparams, fopt, numiter, func_calls, warnflag) = retval
        #
        # else:
        #     raise AttributeError("the specified algorithm '" + str(self.algorithm)
        #             + "' is unsupported.  Options are 'CG', 'LBFGSB', "
        #             "'Nelder-Mead', 'Powell', and 'BFGS'")

        if np.any(self.params != newparams):
            self.setparams(newparams)
        self.func_calls = func_calls
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
            if self.verbose:
                print("Function eval #", self.fnevals)

        if params is not None:
            self.setparams(params)

        if not hasattr(self, 'K'):
            raise ValueError('the entropy dual is a function of '
                             'the target feature expectations. '
                             'Set these first by calling `fit`.')

        # Subsumes both small and large cases:
        L = self.log_norm_constant() - np.dot(self.params, self.K)

        if np.isnan(L):
            raise ValueError('Oops: the dual is nan! Debug me!')

        if self.verbose and self.external is None:
            print("  dual is ", L)

        # Use a Gaussian prior for smoothing if requested.
        # This adds the penalty term \sum_{i=1}^m \params_i^2 / {2 \sigma_i^2}.
        # Define 0 / 0 = 0 here; this allows a variance term of
        # sigma_i^2==0 to indicate that feature i should be ignored.
        if self.sigma2 is not None and ignorepenalty==False:
            ratios = np.nan_to_num(self.params**2 / self.sigma2)
            # Why does the above convert inf to 1.79769e+308?

            L += 0.5 * ratios.sum()
            if self.verbose and self.external is None:
                print("  regularized dual is ", L)

        if not self.callingback and self.external is None:
            if hasattr(self, 'callback_dual') \
                               and self.callback_dual is not None:
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
            if self.verbose:
                print("Iteration #", self.iters)

        # Store new dual and/or gradient norm
        if not self.callingback:
            if self.storeduals:
                self.duals[self.iters] = self.dual()
            if self.storegradnorms:
                self.gradnorms[self.iters] = norm(self.grad())

        if not self.callingback and self.external is None:
            if hasattr(self, 'callback'):
                # Prevent infinite recursion if the callback function
                # calls dual():
                self.callingback = True
                self.callback(self)
                self.callingback = False

        # Do we perform a test on external sample(s) every iteration?
        # Only relevant to BigModel objects
        if hasattr(self, 'testevery') and self.testevery > 0:
            if (self.iters + 1) % self.testevery != 0:
                if self.verbose:
                    print("Skipping test on external sample(s) ...")
            else:
                self.test()

        if not self.callingback and self.external is None:
            if self.mindual > -np.inf and self.dual() < self.mindual:
                raise DivergenceError("dual is below the threshold 'mindual'"
                        " and may be diverging to -inf.  Fix the constraints"
                        " or lower the threshold!")

        self.iters += 1


    def grad(self, params=None, ignorepenalty=False):
        """Computes or estimates the gradient of the entropy dual.
        """

        if self.verbose and self.external is None and not self.callingback:
            print("Grad eval #" + str(self.gradevals))

        if params is not None:
            self.setparams(params)

        if not hasattr(self, 'K'):
            raise ValueError('the gradient of the entropy dual is '
                             'a function of the target feature '
                             'expectations. Set these first by '
                             'calling `fit`.')

        G = self.expectations() - self.K

        if self.verbose and self.external is None:
            print("  norm of gradient =",  norm(G))

        # (We don't reset params to its prior value.)

        # Use a Gaussian prior for smoothing if requested.  The ith
        # partial derivative of the penalty term is \params_i /
        # \sigma_i^2.  Define 0 / 0 = 0 here; this allows a variance term
        # of sigma_i^2==0 to indicate that feature i should be ignored.
        if self.sigma2 is not None and ignorepenalty==False:
            penalty = self.params / self.sigma2
            G += penalty
            features_to_kill = np.where(np.isnan(penalty))[0]
            G[features_to_kill] = 0.0
            if self.verbose and self.external is None:
                normG = norm(G)
                print("  norm of regularized gradient =", normG)

        if not self.callingback and self.external is None:
            if hasattr(self, 'callback_grad') \
                               and self.callback_grad is not None:
                # Prevent infinite recursion if the callback function
                # calls grad():
                self.callingback = True
                self.callback_grad(self)
                self.callingback = False

        if self.external is None and not self.callingback:
            self.gradevals += 1

        return G


    def cross_entropy(self, fx, log_prior_x=None, base=np.e):
        """Returns the cross entropy H(q, p) of the empirical
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

    @abstractmethod
    def log_norm_constant(self):
        """Subclasses must implement this.
        """

    def norm_constant(self):
        """Return the normalization constant, or partition function, for
        the current model.  Warning -- this may be too large to represent;
        if so, this will result in numerical overflow.  In this case use
        log_norm_constant() instead.

        For 'BigModel' instances, estimates the normalization term as
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

        The parameter 'sigma' will be squared and stored as self.sigma2.
        """
        self.sigma2 = sigma**2


    def setparams(self, params):
        """Set the parameter vector to params, replacing the existing
        parameters.  params must be a list or numpy array of the same
        length as the model's feature vector f.
        """

        self.params = np.array(params, float)        # make a copy

        # Log the new params to disk
        self.logparams()

        # Delete params-specific stuff
        self.clearcache()


    def clearcache(self):
        """Clears the interim results of computations depending on the
        parameters and the sample.
        """
        for var in ['mu', 'logZ', 'logZapprox', 'logv']:
            if hasattr(self, var):
                delattr(self, var)

    def resetparams(self, numfeatures=None):
        """Reset the parameters self.params to zero, clearing the
        cache variables dependent on them.  Also reset the number of
        function and gradient evaluations to zero.
        """

        if numfeatures:
            m = numfeatures
        else:
            # Try to infer the number of parameters from existing state
            if hasattr(self, 'params'):
                m = len(self.params)
            elif hasattr(self, 'F'):
                m = self.F.shape[0]
            elif hasattr(self, 'sampleF'):
                m = self.sampleF.shape[0]
            elif hasattr(self, 'K'):
                m = len(self.K)
            else:
                raise ValueError("specify the number of features / parameters")

        # Set parameters, clearing cache variables
        self.setparams(np.zeros(m, float))

        # These bounds on the param values are only effective for the
        # L-BFGS-B optimizer:
        self.bounds = [(-100., 100.)]*len(self.params)

        self.fnevals = 0
        self.gradevals = 0
        self.iters = 0
        self.callingback = False

        # Clear the stored duals and gradient norms
        self.duals = {}
        self.gradnorms = {}
        if hasattr(self, 'external_duals'):
            self.external_duals = {}
        if hasattr(self, 'external_gradnorms'):
            self.external_gradnorms = {}
        if hasattr(self, 'external'):
            self.external = None


    def setcallback(self, callback=None, callback_dual=None, \
                    callback_grad=None):
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
        if not hasattr(self, 'paramslogcounter'):
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
        paramsfile = open(self.paramslogfilename + '.' + \
                          str(self.paramslogcounter) + '.pickle', 'wb')
        pickle.dump(self.params, paramsfile, pickle.HIGHEST_PROTOCOL)
        paramsfile.close()
        #self.paramslog += 1
        #self.paramslogcounter = 0
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
        #self.paramslog = 1

    def endlogging(self):
        """Stop logging param values whenever setparams() is called.
        """
        del self.paramslogcounter
        del self.paramslogfilename
        del self.paramslogfreq

def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()
