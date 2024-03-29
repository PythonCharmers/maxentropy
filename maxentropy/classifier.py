"""
A generative classifier derived from another classifier and additional feature
constraints.
"""

from collections.abc import Callable, Iterator, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    column_or_1d,
)
from sklearn.utils.multiclass import check_classification_targets
from scipy.special import logsumexp

from maxentropy.utils import (
    prior_log_proba_x_given_k,
)
from maxentropy.density import MinDivergenceDensity


class MinDivergenceClassifier(ClassifierMixin, BaseEstimator):
    """
    Parameters
    ----------
        prior_clf: sklearn classifier
            This must have a method `.predict_log_proba()` that takes an (n, m)
            array X and returns a matrix of log class probabilities
                [log p(k | X)]
            of shape (n, k), where k is the number of classes, giving log p(k |
            X). The probabilities must sum to 1 across each row.

            This will be evaluated on the samples produced by
            `auxiliary_sampler` and the outputs will be extracted as column d
            for each class k in turn.
    """

    def __init__(
        self,
        feature_functions: Sequence[Callable],
        auxiliary_sampler: Iterator,
        *,
        prior_clf=None,
        prior_class_probs=None,
        vectorized=True,
        matrix_format="csc_matrix",
        algorithm="CG",
        max_iter=1000,
        warm_start=False,
        verbose=0,
    ):
        self.feature_functions = feature_functions
        self.auxiliary_sampler = auxiliary_sampler
        self.prior_clf = prior_clf
        self.prior_class_probs = prior_class_probs
        self.vectorized = vectorized
        self.matrix_format = matrix_format
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.verbose = verbose

    def _validate_and_setup(self):
        """
        Various checks and setup stuff
        """
        self.prior_class_probs = column_or_1d(self.prior_class_probs)

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
        X, y = check_X_y(X, y)

        check_classification_targets(y)

        # Handle non-contiguous output labels y:
        self.classes_, y = np.unique(y, return_inverse=True)

        self._validate_and_setup()

        if not self.warm_start:
            self.models = {}

        prior_log_pdfs = {}

        for target_class in range(len(self.classes_)):

            if self.prior_clf is None:
                prior_log_pdfs[target_class] = None
            else:
                prior_log_pdfs[target_class] = prior_log_proba_x_given_k(
                    self.prior_clf, self.prior_class_probs, target_class
                )

            self.models[target_class] = MinDivergenceDensity(
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

        for target_class, model in self.models.items():
            # Filter the rows of X to those whose corresponding y matches the target class:
            X_subset = X[y == target_class]
            # F = evaluate_feature_matrix(
            #     model.feature_functions, X_subset, matrix_format=self.matrix_format
            # )
            if self.verbose:
                print(f"Fitting model for target {target_class}")
            model.fit(X_subset)

        # Custom attribute to track if the estimator is fitted
        self._is_fitted = True
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
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        """
        Logic:

        p(k | x) = p(x | k) p(k) / p(x)

        and p(x) is constant in k. Now use:

        \sum_k p(k | x) = 1

        So we can calculate const by:
        const = p(x | k=0) p(k=0) + p(x | k=1) p(k=1)

        Finally, we have:
        log p(k | x) = log p(x | k) + log p(k) - log const
        """

        log_scores = np.array(
            [model.predict_log_proba(X) for model in self.models.values()]
        ).T
        # These represent pdf values p(x | k) under each component model (density) k.

        unnormalized_log_proba = log_scores + np.log(self.prior_class_probs)
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
        predictions = self.classes_[np.argmax(log_proba, axis=1)]
        # pred = net._label_binarizer.inverse_transform(log_proba)
        return predictions

    def __sklearn_is_fitted__(self):
        """
        Check fitted status and return a Boolean value.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted


__all__ = ["MinDivergenceClassifier"]
