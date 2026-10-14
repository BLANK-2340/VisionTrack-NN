"""A small NumPy CNN that retains coarse spatial layout before classification."""

from __future__ import annotations

import numpy as np


class SpatialConvNet:
    """3x3 convolution -> ReLU -> grid average pool -> linear classifier."""

    def __init__(self, channels: int = 8, classes: int = 10,
                 grid_size: int = 4, seed: int = 0) -> None:
        if channels < 1 or classes < 2 or grid_size < 1:
            raise ValueError(
                "channels and grid_size must be positive; classes must be at least two"
            )
        rng = np.random.default_rng(seed)
        features = channels * grid_size * grid_size
        self.grid_size = grid_size
        self.kernel = rng.normal(0.0, np.sqrt(2.0 / 9.0), (channels, 3, 3))
        self.conv_bias = np.zeros(channels)
        self.weight = rng.normal(0.0, np.sqrt(2.0 / features), (features, classes))
        self.bias = np.zeros(classes)

    def parameters(self) -> dict[str, np.ndarray]:
        return {"kernel": self.kernel, "conv_bias": self.conv_bias,
                "weight": self.weight, "bias": self.bias}

    def _pool(self, activation: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        height, width = activation.shape[2:]
        if height < self.grid_size or width < self.grid_size:
            raise ValueError("convolution output must be at least as large as the pooling grid")
        row_edges = np.linspace(0, height, self.grid_size + 1, dtype=np.int64)
        column_edges = np.linspace(0, width, self.grid_size + 1, dtype=np.int64)
        pooled = np.empty((activation.shape[0], activation.shape[1],
                           self.grid_size, self.grid_size), dtype=np.float64)
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                region = activation[:, :, row_edges[row]:row_edges[row + 1],
                                    column_edges[column]:column_edges[column + 1]]
                pooled[:, :, row, column] = region.mean(axis=(2, 3))
        return pooled, (row_edges, column_edges)

    def _forward(self, images: np.ndarray) -> tuple[np.ndarray, tuple[object, ...]]:
        values = np.asarray(images, dtype=np.float64)
        if (values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 3 or
                values.shape[2] < 3 or not np.all(np.isfinite(values))):
            raise ValueError("images must be a non-empty finite [batch, height>=3, width>=3] array")
        windows = np.lib.stride_tricks.sliding_window_view(values, (3, 3), axis=(1, 2))
        pre = np.einsum("bhwij,kij->bkhw", windows, self.kernel)
        pre += self.conv_bias[None, :, None, None]
        activation = np.maximum(pre, 0.0)
        pooled, edges = self._pool(activation)
        features = pooled.reshape(len(values), -1)
        logits = features @ self.weight + self.bias
        return logits, (windows, pre, features, edges)

    def loss_and_gradients(self, images: np.ndarray,
                           labels: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        logits, (windows, pre, features, edges) = self._forward(images)
        targets = np.asarray(labels)
        if (targets.shape != (len(logits),) or targets.dtype.kind not in "iu" or
                np.any(targets < 0) or np.any(targets >= logits.shape[1])):
            raise ValueError("labels must contain one valid integer class per image")
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.log(probabilities[np.arange(len(targets)), targets]).mean()
        dlogits = probabilities
        dlogits[np.arange(len(targets)), targets] -= 1.0
        dlogits /= len(targets)
        gradients = {"weight": features.T @ dlogits, "bias": dlogits.sum(axis=0)}
        dpooled = (dlogits @ self.weight.T).reshape(
            len(targets), len(self.kernel), self.grid_size, self.grid_size
        )
        dactivation = np.zeros_like(pre)
        row_edges, column_edges = edges
        for row in range(self.grid_size):
            for column in range(self.grid_size):
                row_slice = slice(row_edges[row], row_edges[row + 1])
                column_slice = slice(column_edges[column], column_edges[column + 1])
                area = ((row_edges[row + 1] - row_edges[row]) *
                        (column_edges[column + 1] - column_edges[column]))
                dactivation[:, :, row_slice, column_slice] = (
                    dpooled[:, :, row, column, None, None] / area
                )
        dpre = dactivation * (pre > 0.0)
        gradients["kernel"] = np.einsum("bkhw,bhwij->kij", dpre, windows)
        gradients["conv_bias"] = dpre.sum(axis=(0, 2, 3))
        return float(loss), gradients

    def train(self, images: np.ndarray, labels: np.ndarray, *, epochs: int = 100,
              learning_rate: float = 0.15, batch_size: int = 100,
              seed: int = 0) -> list[float]:
        if (epochs < 1 or not np.isfinite(learning_rate) or
                learning_rate <= 0.0 or batch_size < 1):
            raise ValueError("epochs, learning_rate, and batch_size must be positive")
        values = np.asarray(images, dtype=np.float64)
        targets = np.asarray(labels)
        rng = np.random.default_rng(seed)
        history = []
        batches = max(1, int(np.ceil(len(targets) / batch_size)))
        for _ in range(epochs):
            for indices in np.array_split(rng.permutation(len(targets)), batches):
                _, gradients = self.loss_and_gradients(values[indices], targets[indices])
                for name, parameter in self.parameters().items():
                    parameter -= learning_rate * gradients[name]
            history.append(self.loss_and_gradients(values, targets)[0])
        return history

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        return self._forward(images)[0]

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_logits(images).argmax(axis=1)
