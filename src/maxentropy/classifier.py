"""
A generative classifier derived from another classifier and additional feature
constraints.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Optional, Iterable, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
    column_or_1d,
)
from sklearn.utils.multiclass import check_classification_targets
from scipy.special import logsumexp

from maxentropy.density import MinDivergenceDensity


# class MinDivergenceClassifier(ClassifierMixin, BaseEstimator):
#     """
#     TODO: reimplement this more simply in terms of the other components.
#
#     Parameters
#     ----------
#         prior_clf: sklearn classifier
#             This must have a method `.predict_log_proba()` that takes an (n, m)
#             array X and returns a 2d array of log class probabilities
#                 [log p(k | X)]
#             of shape (n, k), where k is the number of classes, giving log p(k |
#             X). The probabilities must sum to 1 across each row.
#
#             This will be evaluated on the samples produced by
#             `auxiliary_sampler` and the outputs will be extracted as column d
#             for each class k in turn.
#     """
#
#     def __init__(
#         self,
#         feature_functions: Sequence[Callable],
#         auxiliary_sampler: Iterator,
#         *,
#         prior_clf=None,
#         prior_class_probs=None,
#         vectorized=True,
#         array_format="csc_array",
#         algorithm="CG",
#         max_iter=1000,
#         warm_start=False,
#         verbose=0,
#         smoothing_factor=None,
#     ):
#         self.feature_functions = feature_functions
#         self.auxiliary_sampler = auxiliary_sampler
#         self.prior_clf = prior_clf
#         self.prior_class_probs = prior_class_probs
#         self.vectorized = vectorized
#         self.array_format = array_format
#         self.algorithm = algorithm
#         self.max_iter = max_iter
#         self.warm_start = warm_start
#         self.verbose = verbose
#         self.smoothing_factor = smoothing_factor
#
#     # @_fit_context(prefer_skip_nested_validation=True)
#     def fit(self, X, y, sample_weight=None):
#         """Fit the baseline classifier.
#
#         Parameters
#         ----------
#         X : array-like of shape (n_samples, n_features)
#             Training data.
#
#         y : array-like of shape (n_samples,) or (n_samples, n_outputs)
#             Target values.
#
#         sample_weight : array-like of shape (n_samples,), default=None
#             Sample weights.
#
#         Returns
#         -------
#         self : object
#             Returns the instance itself.
#         """
#         X = self._validate_data(X, cast_to_ndarray=False, accept_sparse=["csr", "csc"])
#         y = self._validate_data(y=y)
#         X, y = check_X_y(X, y)
#
#         check_classification_targets(y)
#
#         # Handle non-contiguous output labels y:
#         self.classes_, y = np.unique(y, return_inverse=True)
#
#         self.prior_class_probs = column_or_1d(self.prior_class_probs)
#
#         if not self.warm_start:
#             self.models = {}
#
#         @tz.curry
#         def prior_log_proba_x_given_k(
#             prior_clf: ClassifierMixin,
#             prior_class_probs: np.ndarray,
#             target_class,
#             X: np.ndarray,
#         ):
#             outputs = prior_clf.predict_log_proba(X) - np.log(prior_class_probs)
#             return outputs[:, target_class]
#
#         prior_log_pdfs = {}
#
#         for target_class in range(len(self.classes_)):
#
#             if self.prior_clf is None:
#                 prior_log_pdfs[target_class] = None
#             else:
#                 prior_log_pdfs[target_class] = prior_log_proba_x_given_k(
#                     self.prior_clf, self.prior_class_probs, target_class
#                 )
#
#             self.models[target_class] = MinDivergenceDensity(
#                 self.feature_functions,
#                 self.auxiliary_sampler,
#                 prior_log_pdf=prior_log_pdfs[target_class],
#                 vectorized=self.vectorized,
#                 array_format=self.array_format,
#                 algorithm=self.algorithm,
#                 max_iter=self.max_iter,
#                 warm_start=self.warm_start,
#                 verbose=self.verbose,
#                 smoothing_factor=self.smoothing_factor,
#             )
#
#         for target_class, model in self.models.items():
#             # Filter the rows of X to those whose corresponding y matches the target class:
#             X_subset = X[y == target_class]
#             if self.verbose:
#                 print(f"Fitting model for target {target_class}")
#             model.fit(X_subset)
#
#         # Custom attribute to track if the estimator is fitted
#         self._is_fitted = True
#         return self
#
#     def predict_log_proba(self, X):
#         """
#         The log probability of the true model being for each target class of
#         those fitted.
#
#         The probability vector for class k given X is defined as:
#
#             p(k | x) = p(x | k) p(k) / p(x)
#
#         So:
#
#             log p(k | x) = log p(x | k) + log p(k) - additive constant
#
#         We write p_1(x) = p(x | k=1) and normalize by re-expressing this:
#
#             log [ (p_1(x), ..., p_k(x)) / sum_i p_i(x) ]
#
#         as this:
#
#             (log p_1(x), ..., log p_k(x)) - logsumexp_i p_i(x)
#
#         where the p_i values are the values of the probability density (or
#         mass) functions (not necessarily normalized) for the m component
#         models.
#         """
#         # Check if fit has been called
#         check_is_fitted(self)
#
#         # Input validation
#         X = check_array(X)
#
#         """
#         Logic:
#
#         p(k | x) = p(x | k) p(k) / p(x)
#
#         and p(x) is constant in k. Now use:
#
#         \sum_k p(k | x) = 1
#
#         So we can calculate const by:
#         const = p(x | k=0) p(k=0) + p(x | k=1) p(k=1)
#
#         Finally, we have:
#         log p(k | x) = log p(x | k) + log p(k) - log const
#         """
#
#         log_scores = np.array(
#             [model.predict_log_proba(X) for model in self.models.values()]
#         ).T
#         # These represent pdf values p(x | k) under each component model (density) k.
#
#         unnormalized_log_proba = log_scores + np.log(self.prior_class_probs)
#         log_const = logsumexp(unnormalized_log_proba, axis=1)
#         log_proba = (unnormalized_log_proba.T - log_const).T
#         return log_proba
#
#     def predict_proba(self, X):
#         """
#         The probability of the true model being for each target class of
#         those fitted.
#         """
#         return np.exp(self.predict_log_proba(X))
#
#     def predict(self, X):
#         log_proba = self.predict_log_proba(X)
#         predictions = self.classes_[np.argmax(log_proba, axis=1)]
#         # pred = net._label_binarizer.inverse_transform(log_proba)
#         return predictions
#
#     def __sklearn_is_fitted__(self):
#         """
#         Check fitted status and return a Boolean value.
#         """
#         return hasattr(self, "_is_fitted") and self._is_fitted
#



class D2GClassifier(ClassifierMixin, BaseEstimator):
    """
    Construct a classifier from a fitted density p(x | k) using the Bayes
    decision rule:

        p(k | x) \propto p(x | k) p(k)

    normalizing so sum_k p(k | x) = 1 across the K classes k=1, ..., K for all
    observations x.

    We expect posterior_log_pdf to return a N x K array of log probabilities
    log p(x | k), with one column for each of the classes k=1, ..., K.

    prior_class_probs must be a 1d array of length k giving the proportions of
    each target class (unique y_train value) in the training set. You can estimate this from the proportions of each class k in the training data labels y_train using:

        freq = np.bincount(y)
        prior_class_probs = freq / np.sum(freq)

    """

    def __init__(
        self,
        # Maybe one day we'll have this instead: posterior_density: DensityMixin
        posterior_log_pdf: Callable,
        prior_class_probs: np.ndarray,
        *,
        vectorized: bool = True,
        array_format: str = "csc_array",
        algorithm: str = "CG",
        max_iter: int = 1000,
        warm_start: bool = False,
        verbose: int = 0,
        smoothing_factor: Optional[float] = None,
    ):
        self.posterior_log_pdf = posterior_log_pdf
        self.prior_class_probs = prior_class_probs
        self.vectorized = vectorized
        self.array_format = array_format
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose
        self.smoothing_factor = smoothing_factor

    def fit(self, X, y, sample_weight=None):
        """
        Does nothing. Provided for pipelines.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def predict_log_proba(self, X):
        """
        The log probability of the true model being for each target class of
        those fitted.

        The probability vector for class k given X is defined as:

            p(k | x) = p(x | k) p(k) / p(x)

        So:

            log p(k | x) = log p(x | k) + log p(k) - additive constant

        We write p_1(x) = p(x | k=1) and normalize by re-expressing this:

            log [ (p_1(x), ..., p_k(x)) / sum_i p_i(x) ]

        as this:

            (log p_1(x), ..., log p_k(x)) - logsumexp_i p_i(x)

        where the p_i values are the values of the probability density (or
        mass) functions (not necessarily normalized) for the m component
        models.
        """
        # Check if fit has been called
        # check_is_fitted(self.posterior_density)

        # Input validation
        X = check_array(X)

        """
        Logic:

        p(k | x) = p(x | k) p(k) / p(x)

        and p(x) is constant in k. Now use:

        \sum_k p(k | x) = 1

        So we can calculate const by:
        const = p(x | k=0) p(k=0) + p(x | k=1) p(k=1) + ... p(k | k=K) p(k=K)

        Finally, we have:
        log p(k | x) = log p(x | k) + log p(k) - log const
        """

        # Calculate the pdf values p(x | k) under each component model (density) k.
        # log_p_x_given_k = self.posterior_density.score_samples(X)
        log_p_x_given_k = self.posterior_log_pdf(X)

        unnormalized_log_proba = log_p_x_given_k + np.log(self.prior_class_probs)
        log_const = logsumexp(unnormalized_log_proba, axis=1)
        log_proba = (unnormalized_log_proba.T - log_const).T
        return log_proba

    def predict_proba(self, X):
        """
        The probability of the true model being for each target class of
        those fitted.
        """
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        # predictions = self.classes_[np.argmax(log_proba, axis=1)]
        # TODO: generalize this to allow classes other than 0, 1, ..., K-1.
        # (We currently assume classes are numbered contiguously, starting at 0.)
        predictions = np.argmax(log_proba, axis=1)
        return predictions



class GenerativeBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    A meta-estimator that turns a density estimator into a supervised classifier
    via the Bayes decision rule.

    This classifier implements the following decision function:

        c*(x) = argmax_c [ log p(x | c) + log p(c) ]

    where p(x | c) is given by the class-conditional density estimator and
    p(c) is the class prior. Logarithms are used for numerical stability.

    Parameters
    ----------
    density_estimator : BaseEstimator
        Instantiated (but unfitted) density estimator with .fit() and
        .score_samples() methods. We expect .score_samples() to return the log
        density p(X). The estimator will be cloned once for each unique class
        during fitting.

    priors : dict[str, float] or "empirical" or None, default=None
        Class priors.

        Optional dictionary mapping class label → prior probability p(c).

        If None or "empirical", class priors will be estimated from empirical
        frequencies y. Custom priors may be useful for various purposes, like
        cost-sensitive classification or with highly imbalanced datasets.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of sampling in `sample()`. Passed to
        `sklearn.utils.check_random_state`. Does not affect fitting.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        All seen class labels in sorted order (consistent with sklearn API).

    estimators_ : dict[str, BaseEstimator]
        Mapping from each class label to the fitted density estimator instance
        cloned from the input `density_estimator`.

    log_priors_ : np.ndarray of shape (n_classes,)
        Logarithm of class prior probabilities log p(c), ordered according to
        `classes_`.

    Notes
    -----
    - At prediction time, `score_samples` from each class-specific density
      estimator is evaluated on X.
    - Returned values are treated as log p(x | c).
    - Class scores are computed as log p(x | c) + log p(c).
    - The predicted label corresponds to the class index with maximal score.

    Future Extensions
    -----------------
    This basic API may later support:
    - A list of density estimators, one per class, passed at construction.
    - A list of already fitted density estimators (e.g., trained on separate data).

    See Also
    --------
    - sklearn.naive_bayes : Naive Bayes models are generative classifiers,
      but use fixed conditional likelihood families and are not wrappers over
      arbitrary density estimators.
    """

    def __init__(
        self,
        density_estimator: BaseEstimator,
        priors: dict[str, float] | str | None = None,
        random_state=None
    ) -> None:
        """
        Initialize the GenerativeBayesClassifier.
        """
        # The scikit-learn convention is that __init__ usually contains *only*
        # the assignments of passed hyperparameters and no additional
        # validation. Validation happens in `fit()`, where data and class
        # labels are available.
        self.density_estimator = density_estimator
        self.priors = priors
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Iterable[Any]) -> GenerativeBayesClassifier:
        """
        Fit a density estimator for each unique observed class.

        We discover the set of classes, clone one density estimator per class,
        and train each separate model on the corresponding subset of the
        training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target class labels. May be strings, integers, or any hashable type
            supported by numpy arrays. The set of unique labels in `y` defines
            `self.classes_`.

            If `priors` is a dict, it must provide a prior for every class observed
            in `y` (i.e. every element of `self.classes_`). Any extra keys in the
            `priors` dict that do not correspond to observed classes are ignored.

        Returns
        -------
        self : GenerativeBayesClassifier
            The fitted estimator.

        Notes
        -----
        - **One density estimator per class:**
          Conceptually, this implements a generative model of the form
              p(x | class = c) for each class c.
          This differs from discriminative classifiers like logistic regression,
          which model p(class | x) directly.

        - **Bayes priors:**
          Priors may be specified manually (e.g., domain knowledge, class costs),
          or learned empirically from sample frequencies. All priors are stored
          internally as log p(c).

        - **Cloning:**
          We clone the provided `density_estimator` for every class using
          sklearn.base.clone(). This ensures each class model is independent,
          parameter-isolated, and compatible with scikit-learn meta-estimators.

        Examples
        --------
        >>> from sklearn.neighbors import KernelDensity
        >>> clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.5))
        >>> clf.fit(X_train, y_train)
        >>> clf.predict(X_test)


        Returns
        -------
        self : GenerativeBayesClassifier
            Fitted estimator.
        """
        # Validate input: enforce 2D numeric X, and y length consistency
        X, y = check_X_y(X, y, ensure_2d=True, dtype="numeric")
    
        # Determine unique labels
        self.classes_ = np.unique(y)
    
        # Create fresh density estimators for each class
        self.estimators_ = self._clone_density_for_classes()
    
        # Safety: estimator must expose score_samples
        # This prevents mysterious runtime failures later.
        for est in self.estimators_.values():
            if not hasattr(est, "score_samples"):
                raise TypeError(
                    f"Density estimator {type(est).__name__} must implement .score_samples(X)."
                )
    
        # Fit each class-specific estimator on its own subset of samples
        for c in self.classes_:
            est = self.estimators_[c]
            est.fit(X[y == c])
    
        # Compute log priors (empirical or user-provided)
        self.log_priors_ = self._compute_priors(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely class for each input sample (each row in X)
        using the Bayes decision rule.

        The classifier evaluates the log-density for each class‐specific model and adds
        the log prior of that class. The predicted label is the class with the highest
        resulting score:

            y*(x) = argmax_c [ log p(x | c) + log p(c) ]

        This rule corresponds to the Bayes optimal classifier for 0–1 loss, assuming
        the density models approximate the true class-conditional distributions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples. These should match the representation expected by the
            density estimator used at construction (continuous features, embeddings,
            etc.). No validation is performed here; the density estimator is assumed
            to behave consistently with its own `.fit()` and `.score_samples()` logic.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.

        Notes
        -----
        **Why not directly compare p(x|c)?**
        - Densities may be extremely small; multiplying them can cause numerical
          underflow.
        - Working in log space converts multiplication to addition and dramatically
          improves stability.

        **Difference from discriminative classifiers:**
        - This classifier does not learn decision boundaries directly.
        - It separately models each class distribution and lets inference perform
          the comparison.

        **Robustness to imbalance:**
        - Adding log p(c) means rare classes are penalized.
        - Setting custom priors can counteract imbalance or incorporate external
          domain knowledge.

        See Also
        --------
        - `decision_function`: returns the raw score matrix.
        - `predict_proba`: returns normalized class posteriors via softmax.
        """
        # Compute raw scores (log p(x|c) + log p(c)) for every sample / class
        # We use decision_function() for code reuse and clarity.
        scores = self.decision_function(X)

        # Take the class with the greatest score for each sample
        # Argmax returns indices; map them back to class labels
        class_indices = np.argmax(scores, axis=1)
        return self.classes_[class_indices]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the raw (unnormalized log) discriminant scores for each class.

            score_c(x) = log p(x | c) + log p(c)

        where:
        - log p(x|c) is obtained from each class‐specific density estimator via
          `.score_samples(X)`,
        - log p(c) is the log prior.

        These scores may be used for ranking or thresholding. They are not
        normalized probabilities; for that, use .predict_proba(). These scores
        live in a log-density space and are only meaningful up to an additive
        constant.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples, n_classes)
            Raw discriminant scores for each class. Each column corresponds to
            the class at the same index in `self.classes_`.

        Notes
        -----
        This is the canonical quantity for density-based Bayes classifiers. Unlike
        discriminative models (e.g., logistic regression), the model does not learn
        a boundary directly; instead, inference compares these class-conditional
        likelihoods plus priors.
        """
        # Input validation is performed here once, so that all downstream
        # methods (predict, predict_log_proba, predict_proba) benefit.
        check_is_fitted(self, ["classes_", "estimators_", "log_priors_"])
        X = check_array(X, ensure_2d=True, dtype="numeric")

        
        # Evaluate log densities from each class-specific estimator
        all_logdens = [
            # score_samples returns log-density for each sample under p(x|c)
            self.estimators_[c].score_samples(X)
            for c in self.classes_
        ]

        # Stack into matrix (n_samples, n_classes)
        scores = np.column_stack(all_logdens)

        # Add log priors to each column
        scores = scores + self.log_priors_

        return scores


    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities log p(c | x) for each sample
        (each row x in X).

        Returns the same probabilities as `predict_proba`, but in log space:

            log p(c | x) = [log p(x|c) + log p(c)] - logsumexp over classes

        This form is often preferable for:
        - downstream loss functions,
        - numerical stability in pipelines,
        - avoiding precision loss when probabilities are extremely small.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_proba : np.ndarray of shape (n_samples, n_classes)
            Log posterior probabilities. Rows sum to log(1)=0 in log-space.
            Columns correspond to `self.classes_`.

        Notes
        -----
        - log probabilities preserve ordering and are safe for use with argmax.
        """
        scores = self.decision_function(X)
        # Convert to probabilities using log-sum-exp for numeric stability
        log_norm = logsumexp(scores, axis=1, keepdims=True)  # normalization constant
        return scores - log_norm


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute normalized posterior class probabilities p(c | x).

        Posterior for each class is computed using Bayes' rule:

            p(c | x) = softmax_c [ log p(x|c) + log p(c) ]

        Equivalently:

            p(c | x) = exp(log p(x|c) + log p(c)) / Z(x)

        where Z(x) normalizes across all classes for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Posterior probability for each sample and class.
            Columns correspond to `self.classes_` in the same order.

        Notes
        -----
        - These are proper probabilities: each row sums to 1.
        - Predictions may be sharply peaked if class-conditional densities are
          extremely confident.
        - Calibration quality depends entirely on the quality of the underlying
          density estimators, not on this wrapper.
        - This requires normalizing across all classes via the log-sum-exp
          trick for numerical stability.
        """
        return np.exp(self.predict_log_proba(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return the log density log p(x) under the generative model.

        Uses the Bayes mixture form:

            p(x) = Σ_c p(c) * p(x|c)
            log p(x) = logsumexp( log p(x|c) + log p(c) )

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples to evaluate. Must be compatible with the underlying
            density estimators.

        Returns
        -------
        log_density : np.ndarray of shape (n_samples,)
            Log density of each sample under the mixture model.

        Notes
        -----
        - This method *does not* return per-class scores; use `decision_function()`
          instead if you need log p(x|c) + log p(c) in matrix form.

        Examples
        --------
        >>> clf.score_samples(X_test)   # returns log-density, not class posteriors
        """
        scores = self.decision_function(X)
        return logsumexp(scores, axis=1)

    def _compute_priors(self, y: np.ndarray) -> np.ndarray:
        """
        Compute and return the logarithm of class priors.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target labels seen during fitting. Used only if priors="empirical" or None.

        Returns
        -------
        log_priors : np.ndarray of shape (n_classes,)
            Logarithm of prior probabilities p(c) for each class in the order
            of `self.classes_`.

        Notes on priors
        ---------------

        This method implements two approaches:

        1. **Empirical priors** (default behavior)
           Priors are computed from class frequencies:
               p(c) = count(c) / N
           This has the advantage of being data-driven and usually reasonable when
           training data reflects the real deployment distribution.

        2. **User-specified priors**
           The user can pass a mapping {class_label: prior_probability}. This allows:
               - correcting for class imbalance,
               - reflecting domain knowledge,
               - integrating business constraints (e.g., false positives are expensive),
               - using priors learned from external corpora or deployed population.
          If `self.priors` is a dict, its domain must include all classes.

        Other Notes
        -----------
        - The returned quantity is log p(c), matching the scoring convention used in
          the Bayes decision:
              score = log p(x|c) + log p(c)
        - No smoothing or Bayesian hyperprior is introduced here. This keeps the
          meta-estimator honest: it delegates density estimation entirely to the
          underlying estimator(s).
        - We *only validate* the numeric correctness of user priors. We do not attempt
          to infer priors from continuous labels, hierarchies, or structured outputs.

        """
        # Case 1: Use empirical class frequencies
        if self.priors is None or self.priors == "empirical":
            # Count occurrences for each class
            counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)

            # Convert to probabilities
            priors = counts / counts.sum()

            # Convert to log
            return np.log(priors)

        # Case 2: User supplied dictionary
        if isinstance(self.priors, dict):
            # Validate: every class must be present
            missing = [c for c in self.classes_ if c not in self.priors]
            if missing:
                raise ValueError(
                    f"priors dict missing classes: {missing}. "
                    "All observed classes must be assigned a prior."
                )

            # Extract and validate non-negativity
            pvals = np.array([self.priors[c] for c in self.classes_], dtype=float)
            if np.any(pvals < 0):
                raise ValueError("Class priors must be non-negative.")

            # Normalize to sum to 1, so users don't have to be perfectly precise
            total = pvals.sum()
            if total <= 0:
                raise ValueError("Sum of priors must be > 0.")
            pvals = pvals / total

            return np.log(pvals)

        # Any other type is unsupported
        raise ValueError(
            "Invalid value for `priors`. Expected None, 'empirical', or dict[str, float]. "
            f"Got: {self.priors!r}"
        )

    def _clone_density_for_classes(self) -> dict[Any, BaseEstimator]:
        """
        Clone the base density estimator once per class.

        Returns
        -------
        estimators : dict[class_label, BaseEstimator]
            A dictionary mapping each class label in `self.classes_` to a
            **fresh, unfitted** estimator instance cloned from
            `self.density_estimator`.

        Notes
        -----
        - This mirrors how sklearn meta-estimators behave internally
          (e.g. OneVsRestClassifier, StackingClassifier).
        - It ensures independence between models and safe parallel training.
        - It prevents subtle bugs during cross-validation or refitting.
        """
        # Following sklearn meta-estimator conventions:
        # - Parameters are copied via `.get_params()`
        #   (so hyperparameters are preserved faithfully).
        # - Learned attributes are absent until `.fit()` is called.

        estimators: dict[Any, BaseEstimator] = {}
        for c in self.classes_:
            # clone() copies hyperparameters but not learned attributes
            estimators[c] = clone(self.density_estimator)
        return estimators

    def sample(self, n_samples: int = 1):
        """
        Generate random samples from the fitted generative model.

        The classifier defines a mixture model over features:

            p(x) = Σ_c p(c) · p(x | c)

        This method samples from that mixture distribution using the standard
        two-stage generative procedure:

           1. Sample class labels from the prior p(c)
           2. For each class, draw samples from its density estimator

        Requirements
        ------------
        Every per-class density model must implement `.sample(n_samples)`.
        Examples that support `.sample()`:
            * GaussianMixture
            * BayesianGaussianMixture

        KernelDensity does **not** implement sampling. If any class estimator
        lacks `.sample()`, a RuntimeError is raised.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to draw.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Generated feature samples.

        y : ndarray of shape (n_samples,)
            Corresponding sampled class labels

        Raises
        ------
        RuntimeError
            If any fitted density estimator does not implement `.sample()`.
        ValueError
            If n_samples < 1.

        Notes
        -----
        This is a generative operation and does not respect decision boundaries.
        It samples from the full joint distribution implied by the model.

        Examples
        --------
        >>> from sklearn.mixture import GaussianMixture
        >>> clf = GenerativeBayesClassifier(GaussianMixture(n_components=3))
        >>> clf.fit(X_train, y_train)
        >>> X_synth, y_synth = clf.sample(200)
        """
        # Ensure fitted
        check_is_fitted(self, ["estimators_", "classes_", "log_priors_"])

        if n_samples < 1:
            raise ValueError(
                f"Invalid value for 'n_samples': {n_samples}. "
                "Sampling requires at least one sample."
            )

        # Convert log priors to actual mixture weights
        priors = np.exp(self.log_priors_)

        # RNG compatibility with sklearn
        rng = check_random_state(self.random_state)

        # Number of samples to draw from each class
        n_samples_by_class = rng.multinomial(n_samples, priors)

        # Validate that every model supports sampling
        for c, est in self.estimators_.items():
            if not hasattr(est, "sample"):
                raise RuntimeError(
                    f"Density estimator for class {c!r} does not implement .sample(). "
                    "Sampling requires all sub-estimators to support .sample()."
                )

        X_chunks = []
        y_chunks = []
        for c, count in zip(self.classes_, n_samples_by_class):
            if count == 0:
                continue

            result = self.estimators_[c].sample(count)

            # GaussianMixture / BayesianGaussianMixture: (X, labels)
            if isinstance(result, tuple):
                X_c = result[0]
            else:
                # KernelDensity: pure ndarray
                X_c = result
            
            X_chunks.append(X_c)
            # Label samples with class
            y_chunks.append(np.full(count, c, dtype=object))

        X = np.vstack(X_chunks)
        y = np.concatenate(y_chunks)

        return X, y


__all__ = ["D2GClassifier", "GenerativeBayesClassifier"]
