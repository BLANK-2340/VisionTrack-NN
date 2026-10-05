# Self-contained vision baseline

This task-owned extension leaves the repository's original `main.py` untouched.
It implements a small convolutional classifier directly in NumPy and trains on a
deterministic generated orientation task: noisy vertical, horizontal and diagonal
lines. The generated data is original test data, needs no download or license, and
is intended to prove the training and evaluation path—not to claim real-world
vision performance.

## Run

```sh
python -m pip install -r requirements-baseline.txt
python -m unittest discover -s tests -v
python -m vision_baseline.demo
```

The model is `3x3 convolution -> ReLU -> global average pooling -> linear
classifier`. Cross-entropy gradients for kernels, convolution bias, classifier
weights and classifier bias are implemented explicitly. Mini-batch SGD uses fixed
seeds, and train/test images use independent noise and placement draws.

Tests cover deterministic balanced data, pixel bounds, central finite-difference
checks for every parameter group, invalid inputs, loss reduction, and held-out
accuracy. The accuracy threshold is a regression check on this narrow synthetic
task and must not be presented as accuracy on a public or real-world dataset.

## Verified result — October 5, 2026

Python 3.12.14 and NumPy 2.3.5 passed all four test cases. The reproducible demo
used 180 training and 90 held-out examples: loss decreased from 1.108148 to
0.011034, with 1.0000 train accuracy and 1.0000 test accuracy. These deterministic
synthetic-task values are regression evidence only, not a real-data result.

Limitations: grayscale 12x12 inputs, three artificial classes, one convolutional
layer, no augmentation beyond generated placement/noise, no checkpoint format and
no external dataset. This is an educational CPU baseline, not a benchmark or paper
reproduction.

Next vision milestone: add model serialization and a minimal PGM image inference
CLI, then evaluate a separately documented baseline on a small appropriately
licensed public dataset. Portfolio rotation: CUDA reduction remains next in
`cuda-kernel-lab`; numeric MLP save/load remains next in `cpp-autograd-engine`.
