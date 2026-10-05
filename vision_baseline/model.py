"""A deterministic NumPy CNN and synthetic orientation dataset."""

from __future__ import annotations

import numpy as np


def make_dataset(samples_per_class: int, seed: int, size: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Create noisy vertical, horizontal, and diagonal-line images."""
    if samples_per_class < 1 or size < 8:
        raise ValueError("samples_per_class must be positive and size must be at least 8")
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in range(3):
        for _ in range(samples_per_class):
            image = rng.normal(0.0, 0.08, (size, size))
            if label == 0:
                column = int(rng.integers(2, size - 2))
                image[:, column - 1 : column + 1] += 1.0
            elif label == 1:
                row = int(rng.integers(2, size - 2))
                image[row - 1 : row + 1, :] += 1.0
            else:
                offset = int(rng.integers(-2, 3))
                for row in range(size):
                    column = row + offset
                    if 0 <= column < size:
                        image[row, column] += 1.0
                    if 0 <= column + 1 < size:
                        image[row, column + 1] += 1.0
            images.append(np.clip(image, 0.0, 1.0))
            labels.append(label)
    order = rng.permutation(len(labels))
    return np.asarray(images, dtype=np.float64)[order], np.asarray(labels)[order]


class TinyConvNet:
    """3x3 convolution -> ReLU -> global average pool -> linear classifier."""

    def __init__(self, channels: int = 8, classes: int = 3, seed: int = 0) -> None:
        if channels < 1 or classes < 2:
            raise ValueError("channels must be positive and classes must be at least two")
        rng = np.random.default_rng(seed)
        self.kernel = rng.normal(0.0, np.sqrt(2.0 / 9.0), (channels, 3, 3))
        self.conv_bias = np.zeros(channels)
        self.weight = rng.normal(0.0, np.sqrt(2.0 / channels), (channels, classes))
        self.bias = np.zeros(classes)

    def parameters(self) -> dict[str, np.ndarray]:
        return {"kernel": self.kernel, "conv_bias": self.conv_bias,
                "weight": self.weight, "bias": self.bias}

    def _forward(self, images: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        if images.ndim != 3 or images.shape[1] < 3 or images.shape[2] < 3:
            raise ValueError("images must have shape [batch, height>=3, width>=3]")
        windows = np.lib.stride_tricks.sliding_window_view(images, (3, 3), axis=(1, 2))
        pre = np.einsum("bhwij,kij->bkhw", windows, self.kernel) + self.conv_bias[None, :, None, None]
        activation = np.maximum(pre, 0.0)
        features = activation.mean(axis=(2, 3))
        logits = features @ self.weight + self.bias
        return logits, (windows, pre, features)

    def loss_and_gradients(self, images: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, (windows, pre, features) = self._forward(images)
        if labels.shape != (images.shape[0],) or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("labels must contain one valid class per image")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(len(labels)), labels]).mean()
        dlogits = probabilities
        dlogits[np.arange(len(labels)), labels] -= 1.0
        dlogits /= len(labels)
        gradients = {
            "weight": features.T @ dlogits,
            "bias": dlogits.sum(axis=0),
        }
        dfeatures = dlogits @ self.weight.T
        spatial = pre.shape[2] * pre.shape[3]
        dpre = dfeatures[:, :, None, None] * (pre > 0.0) / spatial
        gradients["kernel"] = np.einsum("bkhw,bhwij->kij", dpre, windows)
        gradients["conv_bias"] = dpre.sum(axis=(0, 2, 3))
        return float(loss), gradients

    def train(self, images: np.ndarray, labels: np.ndarray, epochs: int = 120,
              learning_rate: float = 0.35, batch_size: int = 30, seed: int = 0) -> list[float]:
        if epochs < 1 or learning_rate <= 0 or batch_size < 1:
            raise ValueError("epochs, learning_rate, and batch_size must be positive")
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            for indices in np.array_split(rng.permutation(len(labels)), max(1, int(np.ceil(len(labels) / batch_size)))):
                _, gradients = self.loss_and_gradients(images[indices], labels[indices])
                for name, parameter in self.parameters().items():
                    parameter -= learning_rate * gradients[name]
            history.append(self.loss_and_gradients(images, labels)[0])
        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self._forward(images)[0].argmax(axis=1)

    def accuracy(self, images: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.predict(images) == labels))
