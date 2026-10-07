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

## Checkpoint and PGM inference

Train, save a deterministic checkpoint, and classify the included PGM image:

```sh
python -m vision_baseline.demo --checkpoint /tmp/orientation.vtnn
python -m vision_baseline.infer /tmp/orientation.vtnn examples/vertical_12x12.pgm
```

Expected final inference line:

```text
class_index=0 label=vertical height=12 width=12
```

The version-1 `VTNNCHK` format stores canonical JSON metadata followed by fixed
little-endian float64 arrays in `kernel`, `conv_bias`, `weight`, `bias`
order and a SHA-256 checksum. Identical models produce identical bytes. Loading
uses no pickle or executable object deserialization and validates the magic,
version, architecture, dtype, array table, shapes, parameter/file limits, finite
values, exact payload length and checksum before returning a model.

`vision_baseline.pgm` supports grayscale P2 (ASCII) and P5 (binary) portable
graymaps with comments in the header, maxima from 1 through 65535, and big-endian
16-bit P5 samples. It rejects excessive dimensions, truncated/extra samples and
values above the declared maximum. Pixels are normalized to float64 in `[0, 1]`.
Inference performs no resize or normalization beyond that scaling; images should
match the training domain (12x12 for this demo).

## Verified results

On October 5, 2026, Python 3.12.14 and NumPy 2.3.5 passed the original four test
cases. The reproducible demo used 180 training and 90 held-out examples: loss
decreased from 1.108148 to 0.011034, with 1.0000 train accuracy and 1.0000 test
accuracy. These deterministic synthetic-task values are regression evidence only,
not a real-data result.

On October 7, 2026, the expanded ten-test suite passed on Python 3.12.14 and
NumPy 2.3.5. It additionally covers byte-deterministic and exact model round trips,
prediction preservation, checksum/corruption/version/size validation, P2/P5 and
16-bit PGM parsing, malformed images and checkpoint-to-PGM inference. The complete
train-save-infer demo classified the included vertical sample as class 0.

Limitations: grayscale inputs, three artificial demo classes, one convolutional
layer, no augmentation beyond generated placement/noise, no optimizer state in
checkpoints and no external dataset. SHA-256 detects accidental corruption but is
not an authenticity signature. This is an educational CPU baseline, not a
benchmark or paper reproduction.

Next vision milestone: add a controlled baseline using a small appropriately
licensed public dataset and report measured CPU runtime and accuracy with explicit
dataset attribution. Portfolio rotation: implement CPU-oracle boundary tests and
a tiled CUDA matrix-multiplication benchmark; GPU results remain pending until
actual GPU execution.

