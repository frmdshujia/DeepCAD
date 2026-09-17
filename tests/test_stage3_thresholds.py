import numpy as np

from train_stage3_fusion import operating_threshold


def _metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    sensitivity = (prediction & (labels == 1)).sum() / (labels == 1).sum()
    specificity = ((~prediction) & (labels == 0)).sum() / (labels == 0).sum()
    return sensitivity, specificity


def test_validation_thresholds_satisfy_constraints():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.asarray([0.05, 0.10, 0.20, 0.35, 0.30, 0.60, 0.80, 0.95])
    low_cutoff = operating_threshold(
        labels, probabilities, "sensitivity", 0.75)
    high_cutoff = operating_threshold(
        labels, probabilities, "specificity", 0.75)
    low_sensitivity, _ = _metrics(labels, probabilities, low_cutoff)
    _, high_specificity = _metrics(labels, probabilities, high_cutoff)
    assert low_sensitivity >= 0.75
    assert high_specificity >= 0.75


def test_specificity_threshold_can_use_all_negative_boundary():
    labels = np.asarray([1, 0])
    probabilities = np.asarray([0.1, 0.9])
    cutoff = operating_threshold(labels, probabilities, "specificity", 1.0)
    sensitivity, specificity = _metrics(labels, probabilities, cutoff)
    assert specificity == 1.0
    assert sensitivity == 0.0
    assert cutoff > probabilities.max()
