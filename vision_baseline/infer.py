"""Command-line inference for a saved TinyConvNet and a P2/P5 PGM image."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_model
from .pgm import load_pgm
from .ranking import predict_top_k


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
    parser.add_argument("--top-k", type=int, default=1,
                        help="number of ranked classes to print (default: 1)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="positive softmax temperature (default: 1.0)")
    arguments = parser.parse_args()
    model = load_model(arguments.checkpoint)
    pixels = load_pgm(arguments.image)
    labels = arguments.labels or (
        ["vertical", "horizontal", "diagonal"] if model.bias.size == 3
        else [f"class_{index}" for index in range(model.bias.size)]
    )
    if len(labels) != model.bias.size:
        parser.error(f"--labels requires exactly {model.bias.size} values")
    try:
        ranking = predict_top_k(model, pixels[None, :, :], k=arguments.top_k,
                                temperature=arguments.temperature)
    except ValueError as error:
        parser.error(str(error))
    if arguments.top_k == 1 and arguments.temperature == 1.0:
        prediction = int(ranking.indices[0, 0])
        print(f"class_index={prediction} label={labels[prediction]} "
              f"height={pixels.shape[0]} width={pixels.shape[1]}")
        return
    for rank, (prediction, probability) in enumerate(
            zip(ranking.indices[0], ranking.probabilities[0]), start=1):
        prediction = int(prediction)
        print(f"rank={rank} class_index={prediction} label={labels[prediction]} "
              f"probability={float(probability):.12f}")
    print(f"confidence_margin={float(ranking.confidence_margin[0]):.12f} "
          f"height={pixels.shape[0]} width={pixels.shape[1]} "
          f"temperature={arguments.temperature:.12g}")


if __name__ == "__main__":
    main()
