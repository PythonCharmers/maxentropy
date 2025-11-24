import numpy as np
import pytest
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from maxentropy.classifier import GenerativeBayesClassifier


@pytest.fixture
def simple_data():
    """
    Generate a simple binary 1-D dataset with clearly separable Gaussian clusters.

    Returns
    -------
    (X, y) : tuple
        X : ndarray of shape (400, 1)
            Two Gaussian blobs drawn from N(-2, 0.5) and N(+2, 0.5).
        y : ndarray of shape (400,)
            Corresponding class labels ("red" and "green").
    """
    rng = np.random.default_rng(42)

    X1 = rng.normal(loc=-2.0, scale=0.5, size=(200, 1))
    X2 = rng.normal(loc=+2.0, scale=0.5, size=(200, 1))
    X = np.vstack([X1, X2])
    y = np.array(["red"] * 200 + ["green"] * 200)
    return X, y


def test_fit_predict_string_labels(simple_data):
    X, y = simple_data

    clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.8))
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.dtype.kind in ("U", "O")  # unicode/string or object
    assert set(preds) == {"red", "green"}


def test_predict_proba_rows_sum_to_one(simple_data):
    X, y = simple_data

    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2))
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    row_sums = proba.sum(axis=1)

    # Numerical tolerance
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_classes_order_and_column_alignment(simple_data):
    """
    predict_proba() columns must be aligned with classifier.classes_.
    The i-th column corresponds to the i-th class listed in clf.classes_.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    proba = clf.predict_proba(X)

    # basic invariants
    assert proba.shape == (X.shape[0], clf.classes_.shape[0])

    # each row probability distribution sums to ~1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # check monotonic alignment: argmax of proba must equal predict()
    preds = clf.predict(X)
    pred_idx = np.argmax(proba, axis=1)
    mapped_preds = clf.classes_[pred_idx]

    assert np.all(mapped_preds == preds)


def test_priors_override(simple_data):
    """
    Priors should influence posterior probabilities even if hard predictions
    do not change in a fully separable dataset.
    """
    X, y = simple_data
    priors = {"red": 0.9, "green": 0.1}

    clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.8), priors=priors)
    clf.fit(X, y)

    proba = clf.predict_proba(X)
    red_idx = clf.classes_.tolist().index("red")
    green_idx = clf.classes_.tolist().index("green")

    assert proba[:, red_idx].mean() > proba[:, green_idx].mean()


def test_invalid_missing_prior(simple_data):
    X, y = simple_data

    # Missing "green"
    priors = {"red": 0.8}

    clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.8), priors=priors)
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_score_samples_log_density(simple_data):
    X, y = simple_data

    clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.8))
    clf.fit(X, y)

    logp = clf.score_samples(X)
    assert logp.shape == (X.shape[0],)

    # log-density values should be finite (no inf or nan)
    assert np.isfinite(logp).all()


def test_score_samples_reflects_density_quality(simple_data):
    """
    Model should assign higher likelihood to samples drawn
    from its own training distribution than to distant values.
    """
    X, y = simple_data

    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2))
    clf.fit(X, y)

    # These samples are near training data
    logp_train = clf.score_samples(X)

    # Outliers far from both classes
    X_far = np.array([[50.0], [-50.0], [100.0]])
    logp_far = clf.score_samples(X_far)

    assert logp_train.mean() > logp_far.mean()


def test_score_method(simple_data):
    """
    The classifier inherits .score() from ClassifierMixin,
    so .score(X, y) must return classification accuracy — not likelihood.

    Since this dataset is separable into two Gaussian clusters,
    the generative classifier should achieve reasonably high accuracy
    when predicting on the data it was trained on.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    acc = clf.score(X, y)

    # Score should be a float in [0, 1]
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0

    # The synthetic dataset is easy — expect very high accuracy
    assert acc > 0.9


# ============================================================
# Tests for GenerativeBayesClassifier.sample()
# ============================================================


def test_sample_shapes(simple_data):
    """
    Test that sample() returns arrays with correct shape:

    * X has shape (n_samples, n_features)
    * y has shape (n_samples,)

    This validates that the model returns proper batched samples
    consistent with sklearn's GaussianMixture.sample().
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    Xs, ys = clf.sample(100)

    assert Xs.shape == (100, X.shape[1])
    assert ys.shape == (100,)


def test_sample_label_space(simple_data):
    """
    Ensure that sampled labels always belong to the original training label space.

    This checks that sampling does not invent labels and that
    class-conditional generation allocates class membership correctly.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=3, random_state=0))
    clf.fit(X, y)

    _, ys = clf.sample(250)
    assert set(ys) <= set(clf.classes_)


def test_sampling_reflects_priors(simple_data):
    """
    Verify that sampling respects the fitted class priors in a probabilistic sense.

    If priors are strongly skewed (e.g. 0.9 vs 0.1),
    then samples drawn from the generative model should reflect that bias.
    """
    X, y = simple_data
    priors = {"red": 0.9, "green": 0.1}

    clf = GenerativeBayesClassifier(
        GaussianMixture(n_components=2, random_state=0),
        priors=priors
    )
    clf.fit(X, y)

    _, ys = clf.sample(500)
    red_frac = np.mean(ys == "red")
    green_frac = np.mean(ys == "green")

    assert red_frac > green_frac
    assert red_frac > 0.5


def test_sample_distribution_location(simple_data):
    """
    Generated samples should cluster near the training distributions.

    Since the data is made of two clusters at −2 and +2,
    the sample mean should be reasonably near the midpoint,
    and not arbitrarily dispersed.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    Xs, _ = clf.sample(500)

    assert np.abs(np.mean(Xs)) < 2.0


def test_sample_for_KDE(simple_data):
    """
    KernelDensity.sample() is implemented and only returns X,
    not (X, component_ids) like GaussianMixture.

    Verify that the generative classifier correctly handles this case
    and returns (X, y), where y are the sampled class labels.
    """
    X, y = simple_data

    # KDE supports sample() so the classifier should not fail.
    clf = GenerativeBayesClassifier(KernelDensity(bandwidth=0.5))
    clf.fit(X, y)

    Xs, ys = clf.sample(20)

    # Xs should be a numeric array of correct dimensionality
    assert isinstance(Xs, np.ndarray)
    assert Xs.shape == (20, X.shape[1])

    # y should be a label array of length 20
    assert ys.shape == (20,)
    assert set(ys) <= set(clf.classes_)

def test_invalid_n_samples(simple_data):
    """
    Sampling requires n_samples >= 1.

    Values of 0 or negative n_samples should raise ValueError,
    consistent with sklearn GaussianMixture.sample().
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    with pytest.raises(ValueError):
        clf.sample(0)
    with pytest.raises(ValueError):
        clf.sample(-5)


def test_sample_label_dtype_preserved(simple_data):
    """
    The dtype of class labels returned by sample() should preserve
    the original label type. For string labels, dtype should be
    unicode or object, not coerced into numeric types.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    _, ys = clf.sample(100)

    assert ys.dtype.kind in ("O", "U")


def test_sample_counts_respect_multinomial(simple_data):
    """
    Verify that class counts allocated internally via multinomial sampling
    sum to the total requested number of samples.

    This checks population consistency rather than distributional correctness.
    """
    X, y = simple_data
    clf = GenerativeBayesClassifier(GaussianMixture(n_components=2, random_state=0))
    clf.fit(X, y)

    _, ys = clf.sample(500)
    counts = np.array([(ys == c).sum() for c in clf.classes_])
    assert counts.sum() == 500

