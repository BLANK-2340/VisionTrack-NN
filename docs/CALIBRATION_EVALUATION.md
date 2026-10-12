# Confidence calibration evaluation

This is a deterministic evaluation layer for the task-owned NumPy
`TinyConvNet`. It measures confidence quality separately from classification
accuracy and writes a machine-readable JSON artifact.

## Reproduce

```sh
python -m vision_baseline.evaluate --output evaluation.json
```

The command uses three disjoint generated splits:

- 180 clean training images from data seed 11;
- 90 calibration images from data seed 23 and corruption seed 101;
- 90 test images from data seed 29 and corruption seed 103.

Calibration and test inputs receive clipped zero-mean Gaussian pixel noise with
standard deviation 0.4. This is a controlled synthetic corruption protocol, not
a natural-image or camera-noise benchmark. The model is trained only on the clean
training split.

A single positive temperature is fitted on calibration negative log-likelihood
with a deterministic bounded search over log-temperature `[-4, 4]`. That
temperature is then frozen before test evaluation. The artifact contains:

- clean and corrupted test accuracy;
- negative log-likelihood and multiclass Brier score;
- 10-bin expected calibration error and full reliability-bin counts;
- selective accuracy/risk at requested 50%, 75%, 90%, and 100% coverage;
- every data, corruption, and training seed needed to reproduce the protocol.

## Recorded result

Measured locally on October 12, 2026 with Python 3.12.14 and NumPy 2.3.5:

| metric | uncalibrated | temperature-scaled |
|---|---:|---:|
| Corrupted accuracy | 0.977778 | 0.977778 |
| Negative log-likelihood | 0.259316 | 0.057906 |
| Multiclass Brier score | 0.118332 | 0.035628 |
| 10-bin ECE | 0.186439 | 0.031575 |

The fitted temperature was `0.210235`. Calibration-split NLL decreased from
`0.310740` to `0.123000`. Clean test accuracy remained `1.000000`. On the
corrupted test split, the most-confident 81 of 90 examples (90% coverage) were
all correct; full-coverage risk was `0.022222`.

Two independent CLI executions produced byte-identical JSON locally with SHA-256
`1549f29a97bcbedde72c25d488d19de0f3546dfe83c703ab68ddd6294d7c4152`.
CI checks byte identity again within its own environment rather than assuming
cross-version floating-point serialization is identical.

## Interpretation and limitations

Temperature scaling changes confidence, not predicted class, so accuracy should
remain constant. Lower NLL, Brier score, and ECE here show improvement only for
this fixed synthetic corruption and these fixed splits. ECE depends on binning,
the calibration set is small, and the generated line classes are much simpler
than real visual recognition. The selected temperature must not be reused as a
claim of real-world calibration. A public-data baseline remains future work.
