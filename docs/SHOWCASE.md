# VisionTrack-NN showcase

This repository contains a dependency-light NumPy vision stack: an original
synthetic orientation baseline, strict checkpoint and image parsing, calibrated
evaluation, a verified Fashion-MNIST adapter, and controlled spatial-model
comparisons. The repository's original `main.py` remains unchanged.

## Setup and clean-checkout verification

Python 3.12 is used in CI. NumPy is the only runtime dependency.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-baseline.txt
./scripts/verify_showcase.sh
```

The verifier exports committed `HEAD` with `git archive`, then runs the full
test suite, result-artifact consistency validation, bytecode compilation,
checkpointed training, PGM inference, and byte-compared environment and
calibrated-evaluation reports. It therefore detects missing tracked files that
an in-place working-tree test can overlook.

## Runnable targets

| Target | Command | Scope |
| --- | --- | --- |
| Synthetic training | `python -m vision_baseline.demo` | Original generated orientation task; no download |
| Save and infer | `python -m vision_baseline.demo --checkpoint /tmp/orientation.vtnn` then `python -m vision_baseline.infer /tmp/orientation.vtnn examples/vertical_12x12.pgm` | Deterministic validated checkpoint and P2/P5 input path |
| Calibration report | `python -m vision_baseline.evaluate --output /tmp/evaluation.json` | Held-out synthetic corruption, calibration and selective risk |
| Full checks | `python -W error -m unittest discover -s tests -v` | Model gradients, I/O, data manifests, reports and failure paths |
| Result validation | `python -W error -m vision_baseline.validate_results results` | Schemas, manifest identity, confusion totals, aggregates and cross-file provenance |
| Environment fingerprint | `python -W error -m vision_baseline.environment_report --output /tmp/environment.json` | Canonical Python, platform, NumPy, BLAS/LAPACK compiler and SIMD identity without timestamps, hostnames or build paths |
| Fashion-MNIST subset experiments | See [adapter](FASHION_MNIST_ADAPTER.md), [multi-seed](FASHION_MNIST_MULTISEED.md) and [equal-budget](FASHION_MNIST_POOLING_BUDGET.md) guides | Requires separately downloaded official files |

## Recorded results

These values are checked-in measurements, not fresh results from the verifier.
Dataset files and trained Fashion-MNIST checkpoints are not redistributed.

| Experiment | Data and protocol | Recorded result |
| --- | --- | --- |
| Synthetic orientation demo | 180 generated train and 90 independent generated test samples | Loss `1.108148 -> 0.011034`; train/test accuracy `1.0000` |
| Tiny global-pool control | Official verified balanced 1,000/500/500 Fashion-MNIST subsets; one seed; 330 parameters | Test accuracy `28.8%`; macro F1 `0.272427` |
| Spatial CNN seed suite | Same fixed manifest; three prescribed seeds; 1,370 parameters | Mean accuracy `72.67%`; sample stddev `0.70` percentage points; mean macro F1 `0.723750` |
| Equal-parameter pooling comparison | Three seeds; grid and global arms each 1,370 parameters | Mean accuracy `72.67%` vs `36.00%`; mean paired difference `+36.67` percentage points |

The complete machine-readable reports are in
[`results/`](../results/). Their interpretation belongs with the experiment
documents: the spatial suite varies initialization and shuffle only on one fixed
subset, and the equal-parameter comparison also changes convolution width and
compute. It is not a pooling-only causal ablation.

## Architecture and evidence map

- [Baseline architecture, checkpoint format and synthetic evidence](VISION_BASELINE.md)
- [Calibration protocol and recorded synthetic-corruption metrics](CALIBRATION_EVALUATION.md)
- [Fashion-MNIST source verification, licensing and manifest rules](FASHION_MNIST_ADAPTER.md)
- [Single-seed control](FASHION_MNIST_EXPERIMENT.md) and
  [spatial comparison](FASHION_MNIST_SPATIAL_COMPARISON.md)
- [Multi-seed variability](FASHION_MNIST_MULTISEED.md) and
  [matched-parameter comparison](FASHION_MNIST_POOLING_BUDGET.md)
- [Implementation continuity and exact historical checks](PORTFOLIO_PROGRESS.md)

## Known limitations

- Fashion-MNIST measurements use fixed balanced subsets, not the full official
  train and test sets, and the test subset has been evaluated repeatedly.
- Three seeds describe initialization/shuffle variability, not split uncertainty
  or a confidence interval. No best seed is selected.
- The matched-parameter comparison reallocates capacity and is not compute
  matched; it does not identify pooling as the sole cause of the difference.
- Exact repeatability is demonstrated in the recorded Python/NumPy environment;
  other BLAS and math libraries can differ in floating-point rounding.
- There is no GPU, latency, throughput, production-robustness or competitive
  state-of-the-art claim.

The next robustness milestone is to validate a full official Fashion-MNIST run
under a separately fixed protocol, if compute permits, while retaining subset
results and avoiding test-set tuning. Future reports can now retain the portable
environment fingerprint alongside the fixed protocol and data manifest.
