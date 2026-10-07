"""Command-line inference for a saved TinyConvNet and a P2/P5 PGM image."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_model
from .pgm import load_pgm


def predict_file(checkpoint: str | Path, image: str | Path) -> tuple[int, tuple[int, int]]:
    model = load_model(checkpoint)
    pixels = load_pgm(image)
    prediction = int(model.predict(pixels[None, :, :])[0])
    return prediction, pixels.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="VTNN checkpoint produced by the baseline")
    parser.add_argument("image", help="P2 or P5 grayscale PGM image")
    parser.add_argument("--labels", nargs="+", help="optional class labels in model order")
    arguments = parser.parse_args()
    model = load_model(arguments.checkpoint)
    pixels = load_pgm(arguments.image)
    prediction = int(model.predict(pixels[None, :, :])[0])
    labels = arguments.labels or (
        ["vertical", "horizontal", "diagonal"] if model.bias.size == 3
        else [f"class_{index}" for index in range(model.bias.size)]
    )
    if len(labels) != model.bias.size:
        parser.error(f"--labels requires exactly {model.bias.size} values")
    print(f"class_index={prediction} label={labels[prediction]} "
          f"height={pixels.shape[0]} width={pixels.shape[1]}")


if __name__ == "__main__":
    main()
