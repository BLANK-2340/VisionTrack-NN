import unittest

import numpy as np

from vision_baseline import TinyConvNet, make_dataset


class BaselineTests(unittest.TestCase):
    def test_dataset_is_deterministic_and_balanced(self):
        images_a, labels_a = make_dataset(4, seed=5)
        images_b, labels_b = make_dataset(4, seed=5)
        np.testing.assert_array_equal(images_a, images_b)
        np.testing.assert_array_equal(labels_a, labels_b)
        np.testing.assert_array_equal(np.bincount(labels_a), [4, 4, 4])
        self.assertTrue(np.all((images_a >= 0.0) & (images_a <= 1.0)))

    def test_selected_gradients_match_central_differences(self):
        images, labels = make_dataset(2, seed=17, size=8)
        model = TinyConvNet(channels=3, seed=19)
        # Move the check away from ReLU's nondifferentiable zero (blank windows
        # can otherwise produce an exact zero with the default zero bias).
        model.conv_bias[:] = [0.03, -0.02, 0.05]
        _, analytical = model.loss_and_gradients(images, labels)
        epsilon = 1e-6
        for name, index in [("kernel", (1, 0, 2)), ("conv_bias", (2,)),
                            ("weight", (1, 2)), ("bias", (0,))]:
            parameter = model.parameters()[name]
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss_and_gradients(images, labels)[0]
            parameter[index] = original - epsilon
            minus = model.loss_and_gradients(images, labels)[0]
            parameter[index] = original
            numeric = (plus - minus) / (2.0 * epsilon)
            self.assertAlmostEqual(numeric, analytical[name][index], delta=2e-7)

    def test_training_generalizes_to_held_out_noise(self):
        train_images, train_labels = make_dataset(60, seed=11)
        test_images, test_labels = make_dataset(30, seed=29)
        model = TinyConvNet(seed=7)
        initial = model.loss_and_gradients(train_images, train_labels)[0]
        history = model.train(train_images, train_labels, seed=13)
        self.assertLess(history[-1], initial * 0.08)
        self.assertGreaterEqual(model.accuracy(train_images, train_labels), 0.98)
        self.assertGreaterEqual(model.accuracy(test_images, test_labels), 0.95)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            make_dataset(0, seed=1)
        model = TinyConvNet(seed=1)
        with self.assertRaises(ValueError):
            model.predict(np.zeros((1, 2, 2)))
        images, _ = make_dataset(1, seed=2)
        with self.assertRaises(ValueError):
            model.loss_and_gradients(images, np.array([0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
