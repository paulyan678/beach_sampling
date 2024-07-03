import numpy as np

from beach_rl.belief import GaussianSpatialBelief


def test_information_gain_equals_entropy_reduction() -> None:
    belief = GaussianSpatialBelief.from_prior(np.zeros(4), 1.3, 0.2)
    feature = np.array([1.0, -0.5, 0.2, 0.8])
    predicted = belief.information_gain(feature)
    before = belief.logdet()
    realised = belief.update(feature, measurement=0.7)
    after = belief.logdet()
    assert np.isclose(predicted, realised, atol=1e-12)
    assert np.isclose(realised, 0.5 * (before - after), atol=1e-10)


def test_posterior_is_psd_and_uncertainty_decreases() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 6))
    belief = GaussianSpatialBelief.from_prior(np.zeros(6), 1.0, 0.04)
    _, before = belief.predict(features)
    for index in [2, 7, 2, 14]:
        belief.update(features[index], float(rng.normal()))
    _, after = belief.predict(features)
    assert np.linalg.eigvalsh(belief.covariance).min() >= -1e-10
    assert np.all(after <= before + 1e-10)


def test_profile_specific_prior_covariance_is_preserved() -> None:
    mean = np.zeros(3)
    covariance = np.diag([1.0e-4, 2.0, 7.0])
    belief = GaussianSpatialBelief.from_covariance(mean, covariance, 0.04)
    np.testing.assert_array_equal(belief.covariance, covariance)
