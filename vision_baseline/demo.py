"""Train and evaluate the reproducible synthetic orientation baseline."""

import argparse

from .checkpoint import save_model
from .model import TinyConvNet, make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="optional path for a deterministic VTNN checkpoint")
    arguments = parser.parse_args()
    train_images, train_labels = make_dataset(60, seed=11)
    test_images, test_labels = make_dataset(30, seed=29)
    model = TinyConvNet(seed=7)
    initial_loss = model.loss_and_gradients(train_images, train_labels)[0]
    history = model.train(train_images, train_labels, seed=13)
    print(f"train_samples={len(train_labels)} test_samples={len(test_labels)}")
    print(f"initial_loss={initial_loss:.6f} final_loss={history[-1]:.6f}")
    print(f"train_accuracy={model.accuracy(train_images, train_labels):.4f}")
    print(f"test_accuracy={model.accuracy(test_images, test_labels):.4f}")
    if arguments.checkpoint:
        save_model(model, arguments.checkpoint)
        print(f"checkpoint={arguments.checkpoint}")


if __name__ == "__main__":
    main()
