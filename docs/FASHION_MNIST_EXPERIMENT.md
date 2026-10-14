# Controlled Fashion-MNIST experiment

This experiment connects the verified IDX adapter to the repository's original
dependency-light TinyConvNet, a deliberately small
`3x3 convolution -> ReLU -> global average pool -> linear` model. It produces a
canonical JSON report and an optional checksum-protected checkpoint.

The published run used the official files and a manifest containing 100
training, 50 validation and 50 test images per class. The seed was fixed before
training. Validation was used only to fit one scalar temperature; the official
test subset was evaluated once afterward.

```bash
python -m vision_baseline.fashion_mnist /path/to/fashion-mnist \
  --output /tmp/fashion-manifest.json \
  --train-per-class 100 --validation-per-class 50 --test-per-class 50 \
  --seed 20261014

python -m vision_baseline.fashion_experiment \
  /tmp/fashion-manifest.json /path/to/fashion-mnist \
  --output /tmp/fashion-experiment.json \
  --checkpoint /tmp/fashion-experiment.vtnn \
  --channels 16 --epochs 100 --learning-rate 0.15 \
  --batch-size 100 --seed 20261014
```

## Measured result

The local CPU run used Python 3.12.14 and NumPy 2.3.5. All four source MD5
values matched the official upstream values. On the balanced 500-image test
subset, the 330-parameter model reached **28.8% accuracy** and **0.272427 macro
F1**. Training cross-entropy moved from `2.331954` to `1.846882`.

Validation temperature fitting selected `0.788463`. On test data, calibration
changed NLL from `1.869807` to `1.858109` and 10-bin ECE from `0.066507` to
`0.057498`; accuracy is unchanged by positive scalar temperature scaling. The
machine-readable summary in `results/fashion_mnist_tinyconvnet_summary.json`
includes the complete true-row/predicted-column confusion matrix.

## Interpretation and limitations

- This is a controlled educational baseline, not a competitive Fashion-MNIST
  result. Its global average pool discards most spatial layout, and the measured
  accuracy exposes that limitation rather than hiding it.
- The balanced subsets are much smaller than the full 60,000/10,000 dataset.
- No hyperparameter sweep, augmentation, early stopping or test-set tuning was
  performed. Only one seeded run is reported, so variance is unknown.
- CPU wall-clock timing is intentionally omitted: the run did not use a
  controlled benchmarking environment.
- Dataset files and the generated checkpoint are not committed.

The next model milestone should preserve spatial structure (for example,
pooling into a small grid) and compare it under the same manifest and evaluation
protocol.
