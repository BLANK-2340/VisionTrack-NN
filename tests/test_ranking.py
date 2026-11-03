import unittest

import numpy as np

from vision_baseline import TinyConvNet, predict_top_k, rank_probabilities


class RankingTests(unittest.TestCase):
    def test_known_order_margin_and_full_ranking(self):
        probabilities = np.array([[0.1, 0.7, 0.2], [0.4, 0.35, 0.25]])
        ranking = rank_probabilities(probabilities, k=3)
        np.testing.assert_array_equal(ranking.indices, [[1, 2, 0], [0, 1, 2]])
        np.testing.assert_array_equal(
            ranking.probabilities, [[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
        np.testing.assert_allclose(ranking.confidence_margin, [0.5, 0.05])

    def test_exact_ties_use_ascending_class_index(self):
        probabilities = np.array([[0.4, 0.4, 0.2, 0.0],
                                  [0.25, 0.25, 0.25, 0.25]])
        first = rank_probabilities(probabilities, k=2)
        second = rank_probabilities(probabilities.copy(), k=2)
        np.testing.assert_array_equal(first.indices, [[0, 1], [0, 1]])
        np.testing.assert_array_equal(first.indices, second.indices)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)
        np.testing.assert_array_equal(first.confidence_margin, [0.0, 0.0])

    def test_k_does_not_change_confidence_margin(self):
        probabilities = np.array([[0.05, 0.15, 0.8]])
        top_one = rank_probabilities(probabilities, k=1)
        top_three = rank_probabilities(probabilities, k=3)
        np.testing.assert_array_equal(top_one.indices, [[2]])
        np.testing.assert_array_equal(top_one.confidence_margin,
                                      top_three.confidence_margin)

    def test_invalid_probability_matrices_are_rejected(self):
        invalid = [
            np.array([0.5, 0.5]),
            np.empty((0, 2)),
            np.ones((1, 1)),
            np.array([[0.6, 0.6]]),
            np.array([[-0.1, 1.1]]),
            np.array([[np.nan, np.nan]]),
            np.array([[np.inf, -np.inf]]),
            np.array([[0, 1]], dtype=np.int64),
            np.array([[0.5 + 0.0j, 0.5 + 0.0j]]),
        ]
        for probabilities in invalid:
            with self.subTest(shape=probabilities.shape):
                with self.assertRaises(ValueError):
                    rank_probabilities(probabilities)

    def test_invalid_k_values_are_rejected(self):
        probabilities = np.array([[0.2, 0.3, 0.5]])
        for k in (0, 4, -1, True, 1.0, "1"):
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    rank_probabilities(probabilities, k=k)

    def test_model_integration_matches_probabilities_and_argmax(self):
        model = TinyConvNet(channels=3, classes=4, seed=19)
        images = np.random.default_rng(7).random((5, 8, 9))
        probabilities = model.predict_proba(images, temperature=0.75)
        ranking = predict_top_k(model, images, k=3, temperature=0.75)
        np.testing.assert_array_equal(ranking.indices[:, 0], model.predict(images))
        np.testing.assert_array_equal(
            ranking.probabilities, np.take_along_axis(probabilities, ranking.indices, axis=1))
        self.assertEqual(ranking.indices.shape, (5, 3))
        self.assertEqual(ranking.confidence_margin.shape, (5,))


if __name__ == "__main__":
    unittest.main()
