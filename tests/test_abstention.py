import unittest

import numpy as np

from vision_baseline.abstention import (
    AbstentionPolicy,
    abstention_metrics,
    apply_abstention,
    fit_abstention_policy,
    policy_record,
)


class AbstentionTests(unittest.TestCase):
    def test_policy_is_fit_on_validation_and_frozen_for_test(self):
        validation = np.array([[.9, .1], [.7, .3], [.55, .45], [.4, .6]])
        labels = np.array([0, 0, 1, 1])
        policy = fit_abstention_policy(validation, labels, target_coverage=.5)
        self.assertAlmostEqual(policy.min_margin, .4)
        self.assertEqual(policy.validation_retained, 2)
        self.assertEqual(policy.validation_coverage, .5)
        self.assertEqual(policy.validation_risk, 0.0)

        test = np.array([[.75, .25], [.65, .35], [.51, .49]])
        report = abstention_metrics(test, np.array([0, 1, 0]), policy)
        self.assertEqual(report, {
            "coverage": 1 / 3,
            "risk": 0.0,
            "retained": 1,
            "samples": 3,
        })
        np.testing.assert_array_equal(apply_abstention(test, policy),
                                      np.array([True, False, False]))

    def test_boundary_ties_are_retained_together(self):
        probabilities = np.array([[.8, .2], [.7, .3], [.3, .7], [.55, .45]])
        policy = fit_abstention_policy(probabilities, np.array([0, 0, 1, 1]), .5)
        self.assertAlmostEqual(policy.min_margin, .4)
        self.assertEqual(policy.validation_retained, 3)
        self.assertEqual(policy.validation_coverage, .75)

    def test_zero_test_acceptance_has_defined_empty_risk(self):
        policy = AbstentionPolicy(.9, .5, .5, 0.0, 1, 2)
        report = abstention_metrics(np.array([[.6, .4]]), np.array([0]), policy)
        self.assertEqual(report["retained"], 0)
        self.assertEqual(report["coverage"], 0.0)
        self.assertIsNone(report["risk"])

    def test_policy_record_is_json_compatible(self):
        policy = fit_abstention_policy(np.array([[.8, .2], [.6, .4]]),
                                       np.array([0, 1]), 1.0)
        self.assertEqual(policy_record(policy)["validation_samples"], 2)

    def test_invalid_inputs_are_rejected(self):
        probabilities = np.array([[.8, .2], [.6, .4]])
        labels = np.array([0, 1])
        for target in (0.0, 1.1, float("nan"), True):
            with self.subTest(target=target), self.assertRaises(ValueError):
                fit_abstention_policy(probabilities, labels, target)
        with self.assertRaises(ValueError):
            fit_abstention_policy(probabilities, np.array([0]), .5)
        with self.assertRaises(TypeError):
            apply_abstention(probabilities, object())


if __name__ == "__main__":
    unittest.main()
