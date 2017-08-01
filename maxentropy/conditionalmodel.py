from .model import Model


class ConditionalModel(Model):
    """
    A conditional maximum-entropy (exponential-form) model p(x|w) on a
    discrete sample space.

    This is useful for classification problems:
    given the context w, what is the probability of each class x?

    The form of such a model is::

        p(x | w) = exp(theta . f(w, x)) / Z(w; theta)

    where Z(w; theta) is a normalization term equal to::

        Z(w; theta) = sum_x exp(theta . f(w, x)).

    The sum is over all classes x in the set Y, which must be supplied to
    the constructor as the parameter 'samplespace'.

    Such a model form arises from maximizing the entropy of a conditional
    model p(x | w) subject to the constraints::

        K_i = E f_i(W, X)

    where the expectation is with respect to the distribution::

        q(w) p(x | w)

    where q(w) is the empirical probability mass function derived from
    observations of the context w in a training set.  Normally the vector
    K = {K_i} of expectations is set equal to the expectation of f_i(w,
    x) with respect to the empirical distribution.

    This method minimizes the Lagrangian dual L of the entropy, which is
    defined for conditional models as::

        L(theta) = sum_w q(w) log Z(w; theta)
                   - sum_{w,x} q(w,x) [theta . f(w,x)]

    Note that both sums are only over the training set {w,x}, not the
    entire sample space, since q(w,x) = 0 for all w,x not in the training
    set.

    The partial derivatives of L are::

        dL / dtheta_i = K_i - E f_i(X, Y)

    where the expectation is as defined above.

    """
    def __init__(self, F, counts, numcontexts):
        """The F parameter should be a (sparse) m x size matrix, where m
        is the number of features and size is |W| * |X|, where |W| is the
        number of contexts and |X| is the number of elements X in the
        sample space.

        The 'counts' parameter should be a row vector stored as a (1 x
        |W|*|X|) sparse matrix, whose element i*|W|+j is the number of
        occurrences of x_j in context w_i in the training set.

        This storage format allows efficient multiplication over all
        contexts in one operation.
        """
        # Ideally, the 'counts' parameter could be represented as a sparse
        # matrix of size C x X, whose ith row # vector contains all points x_j
        # in the sample space X in context c_i.  For example:
        #     N = sparse.lil_matrix((len(contexts), len(samplespace)))
        #     for (c, x) in corpus:
        #         N[c, x] += 1

        # This would be a nicer input format, but computations are more
        # efficient internally with one long row vector.  What we really need is
        # for sparse matrices to offer a .reshape method so this conversion
        # could be done internally and transparently.  Then the numcontexts
        # argument to the ConditionalModel constructor could also be inferred
        # from the matrix dimensions.

        super(ConditionalModel, self).__init__()
        self.F = F
        self.numcontexts = numcontexts

        S = F.shape[1] // numcontexts          # number of sample point
        assert isinstance(S, int)

        # Set the empirical pmf:  p_tilde(w, x) = N(w, x) / \sum_c \sum_y N(c, y).
        # This is always a rank-2 beast with only one row (to support either
        # arrays or dense/sparse matrices.
        if not hasattr(counts, 'shape'):
            # Not an array or dense/sparse matrix
            p_tilde = asarray(counts).reshape(1, len(counts))
        else:
            if counts.ndim == 1:
                p_tilde = counts.reshape(1, len(counts))
            elif counts.ndim == 2:
                # It needs to be flat (a row vector)
                if counts.shape[0] > 1:
                    try:
                        # Try converting to a row vector
                        p_tilde = counts.reshape((1, counts.size))
                    except AttributeError:
                        raise ValueError("the 'counts' object needs to be a"
                            " row vector (1 x n) rank-2 array/matrix) or have"
                            " a .reshape method to convert it into one")
                else:
                    p_tilde = counts
        # Make a copy -- don't modify 'counts'
        self.p_tilde = p_tilde / p_tilde.sum()

        # As an optimization, p_tilde need not be copied or stored at all, since
        # it is only used by this function.

        self.p_tilde_context = np.empty(numcontexts, float)
        for w in range(numcontexts):
            self.p_tilde_context[w] = self.p_tilde[0, w*S : (w+1)*S].sum()

        # Now compute the vector K = (K_i) of expectations of the
        # features with respect to the empirical distribution p_tilde(w, x).
        # This is given by:
        #
        #     K_i = \sum_{w, x} q(w, x) f_i(w, x)
        #
        # This is independent of the model parameters.
        self.K = flatten(innerprod(self.F, self.p_tilde.transpose()))
        self.numsamplepoints = S


    def log_norm_constant(self):
        """Compute the elementwise log of the normalization constant
        (partition function) Z(w)=sum_{y \in Y(w)} exp(theta . f(w, y)).
        The sample space must be discrete and finite.  This is a vector
        with one element for each context w.
        """
        # See if it's been precomputed
        if hasattr(self, 'logZ'):
            return self.logZ

        numcontexts = self.numcontexts
        S = self.numsamplepoints
        # Has F = {f_i(x_j)} been precomputed?
        if not hasattr(self, 'F'):
            raise AttributeError("first create a feature matrix F")

        # Good, assume F has been precomputed

        log_p_dot = innerprodtranspose(self.F, self.params)

        # Are we minimizing KL divergence?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs

        self.logZ = np.zeros(numcontexts, float)
        for w in range(numcontexts):
            self.logZ[w] = logsumexp(log_p_dot[w*S: (w+1)*S])
        return self.logZ


    def dual(self, params=None, ignorepenalty=False):
        """The entropy dual function is defined for conditional models as

            L(theta) = sum_w q(w) log Z(w; theta)
                       - sum_{w,x} q(w,x) [theta . f(w,x)]

        or equivalently as

            L(theta) = sum_w q(w) log Z(w; theta) - (theta . k)

        where K_i = \sum_{w, x} q(w, x) f_i(w, x), and where q(w) is the
        empirical probability mass function derived from observations of the
        context w in a training set.  Normally q(w, x) will be 1, unless the
        same class label is assigned to the same context more than once.

        Note that both sums are only over the training set {w,x}, not the
        entire sample space, since q(w,x) = 0 for all w,x not in the training
        set.

        The entropy dual function is proportional to the negative log
        likelihood.

        Compare to the entropy dual of an unconditional model:
            L(theta) = log(Z) - theta^T . K
        """
        if not self.callingback:
            if self.verbose:
                print("Function eval #", self.fnevals)

            if params is not None:
                self.setparams(params)

        logZs = self.log_norm_constant()

        L = np.dot(self.p_tilde_context, logZs) - np.dot(self.params, self.K)

        if self.verbose and self.external is None:
            print("  dual is ", L)

        # Use a Gaussian prior for smoothing if requested.
        # This adds the penalty term \sum_{i=1}^m \theta_i^2 / {2 \sigma_i^2}
        if self.sigma2 is not None and ignorepenalty==False:
            penalty = 0.5 * (self.params**2 / self.sigma2).sum()
            L += penalty
            if self.verbose and self.external is None:
                print("  regularized dual is ", L)

        if not self.callingback:
            if hasattr(self, 'callback_dual'):
                # Prevent infinite recursion if the callback function calls
                # dual():
                self.callingback = True
                self.callback_dual(self)
                self.callingback = False
            self.fnevals += 1

        # (We don't reset params to its prior value.)
        return L


    # These do not need to be overridden:
    #     grad
    #     pmf
    #     probdist


    def fit(self, algorithm='CG'):
        """Fits the conditional maximum entropy model subject to the
        constraints

            sum_{w, x} p_tilde(w) p(x | w) f_i(w, x) = k_i

        for i=1,...,m, where k_i is the empirical expectation
            k_i = sum_{w, x} p_tilde(w, x) f_i(w, x).
        """

        # Call base class method
        return model.fit(self, self.K, algorithm)


    def expectations(self):
        """The vector of expectations of the features with respect to the
        distribution p_tilde(w) p(x | w), where p_tilde(w) is the
        empirical probability mass function value stored as
        self.p_tilde_context[w].
        """
        if not hasattr(self, 'F'):
            raise AttributeError("need a pre-computed feature matrix F")

        # A pre-computed matrix of features exists

        numcontexts = self.numcontexts
        S = self.numsamplepoints
        p = self.pmf()
        # p is now an array representing p(x | w) for each class w.  Now we
        # multiply the appropriate elements by p_tilde(w) to get the hybrid pmf
        # required for conditional modelling:
        for w in range(numcontexts):
            p[w*S : (w+1)*S] *= self.p_tilde_context[w]

        # Use the representation E_p[f(X)] = p . F
        return flatten(innerprod(self.F, p))

        # # We only override to modify the documentation string.  The code
        # # is the same as for the model class.
        # return model.expectations(self)


    def logpmf(self):
        """Returns a (sparse) row vector of logarithms of the conditional
        probability mass function (pmf) values p(x | c) for all pairs (c,
        x), where c are contexts and x are points in the sample space.
        The order of these is log p(x | c) = logpmf()[c * numsamplepoints
        + x].
        """
        # Have the features already been computed and stored?
        if not hasattr(self, 'F'):
            raise AttributeError("first set the feature matrix F")

        # p(x | c) = exp(theta.f(x, c)) / sum_c[exp theta.f(x, c)]
        #      = exp[log p_dot(x) - logsumexp{log(p_dot(y))}]

        numcontexts = self.numcontexts
        S = self.numsamplepoints
        log_p_dot = flatten(innerprodtranspose(self.F, self.params))
        # Do we have a prior distribution p_0?
        if self.priorlogprobs is not None:
            log_p_dot += self.priorlogprobs
        if not hasattr(self, 'logZ'):
            # Compute the norm constant (quickly!)
            self.logZ = np.zeros(numcontexts, float)
            for w in range(numcontexts):
                self.logZ[w] = logsumexp(log_p_dot[w*S : (w+1)*S])
        # Renormalize
        for w in range(numcontexts):
            log_p_dot[w*S : (w+1)*S] -= self.logZ[w]
        return log_p_dot
