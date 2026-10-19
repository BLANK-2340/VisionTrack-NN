# Multi-seed spatial Fashion-MNIST evaluation

The seed suite continues the [spatial comparison](FASHION_MNIST_SPATIAL_COMPARISON.md)
without changing its model, preprocessing or manifest. It varies initialization
and minibatch-shuffle seeds, not dataset membership. It is not a parameter-matched
pooling ablation, a full-dataset benchmark, or an estimate of split uncertainty.

## Reproduce

Obtain the four official files following [the adapter guide](FASHION_MNIST_ADAPTER.md).
The adapter verifies their upstream MD5 values and records SHA-256 identities;
the files themselves are not redistributed here.

```sh
python -m pip install -r requirements-baseline.txt
python -m vision_baseline.fashion_mnist /path/to/fashion-mnist \
  --train-per-class 100 --validation-per-class 50 --test-per-class 50 \
  --seed 20261014 --output /tmp/fashion-manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -W error \
  -m vision_baseline.fashion_multiseed /tmp/fashion-manifest.json \
  /path/to/fashion-mnist --seeds 20261014 20261015 20261016 \
  --channels 8 --grid-size 4 --epochs 100 --learning-rate 0.15 \
  --batch-size 100 --bins 10 --output /tmp/fashion-multiseed.json
python -W error -m unittest discover -s tests -v
```

Each seed trains independently, fits its own scalar temperature using validation
NLL only, and then evaluates test data once. All prespecified seeds are retained;
the suite does not choose a best model or ensemble predictions. A changed manifest
aborts the suite, and inconsistent dataset/protocol identities cannot be averaged.
The output contains full per-seed training histories, temperatures, confusion
matrices, class metrics and calibration bins, plus a compact aggregate.

The aggregate reports min, max, mean and **sample** standard deviation (`n-1`)
for accuracy, macro F1, NLL and ECE, retaining calibrated and uncalibrated values.
Positive paired calibration changes mean a regression; these are counted, not
discarded. Standard deviation across three runs is descriptive, not a confidence
interval. The test split was used by the earlier educational comparison, so this
continuation is not a fresh independent benchmark evaluation.

## Measured result — October 19, 2026

Python 3.12.14, NumPy 2.3.5, CPU float64, with OpenBLAS and OMP threads set to
one. All four official files passed MD5 checks. The manifest SHA-256 is
`4b36c62d1043ad2f6ef48953eea5e82aba1f048b1788c20fab7e5a4dc83a5ffc`,
identical to the original 1,000/500/500 comparison. Each run has 1,370 parameters.

| Seed | Test accuracy | Macro F1 | Calibrated NLL | Calibrated ECE |
| --- | ---: | ---: | ---: | ---: |
| 20261014 | 72.0% | 0.721262 | 0.788264 | 0.031309 |
| 20261015 | 72.6% | 0.718273 | 0.778593 | 0.036713 |
| 20261016 | 73.4% | 0.731716 | 0.774786 | 0.036441 |
| Mean | 72.67% | 0.723750 | 0.780547 | 0.034821 |
| Sample stddev | 0.70 percentage points | 0.007058 | 0.006948 | 0.003045 |

Temperature fitting **increased test NLL in 2/3 runs and ECE in 2/3 runs**.
Mean uncalibrated NLL was 0.779751 versus calibrated 0.780547; mean ECE was
0.030998 versus calibrated 0.034821. A validation NLL improvement does not
guarantee an improvement on this small test subset. No seed or unfavorable metric
was removed. The full report is in `results/fashion_mnist_spatial_multiseed.json`.
Two independent complete suite executions produced byte-identical JSON (SHA-256
`045947a56404f37af4d178d1343c41f002ea2fe16676cfd3ab87832c4c9a1d84`).

## Verification and limitations

- Synthetic gzip/IDX fixtures exercise the real training/calibration/CLI path
  for two distinct seeds and compare canonical JSON bytes from repeated runs.
- Hand-calculated statistics verify sample standard deviation and regression
  counts independently of the CNN implementation.
- Invalid plans, duplicate seeds, non-finite metrics, mismatched settings and
  mid-suite manifest changes are rejected.
- Fixture manifests explicitly report `official_source_verified: false`.
- Exact repeatability is tested in one environment; cross-platform BLAS/math
  libraries can differ in floating-point rounding. No GPU or throughput claim
  is made.
