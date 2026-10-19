# Equal-budget pooling comparison

This follows the [multi-seed spatial evaluation](FASHION_MNIST_MULTISEED.md).
It compares two CNNs with equal **total trainable parameter counts**, not equal
convolution width or compute cost. It does not isolate pooling alone.

## Architecture budgets

For `C` channels, a `g` by `g` pooling grid and ten classes, the parameter count
is `C * (10 + 10*g*g) + 10`: nine kernel weights and one bias per channel,
`10*C*g*g` classifier weights and ten classifier biases.

| Arm | Channels | Pooling | Conv parameters | Classifier parameters | Total |
| --- | ---: | --- | ---: | ---: | ---: |
| Grid | 8 | 4×4 | 80 | 1,290 | 1,370 |
| Global | 68 | 1×1 | 680 | 690 | 1,370 |

All counted parameters participate in the forward pass and backpropagation;
there is no unused padding. For a requested grid model, the CLI derives the
global width `C*(1+g*g)/2` and rejects nonintegral matches. Both arms use the
same `SpatialConvNet` implementation. An independent test checks that its 1×1
case matches `TinyConvNet` global pooling in logits, loss and every gradient.

## Reproduce

Use the official, checksum-verified files and the same 1,000/500/500 manifest as
the prior comparisons. See [dataset setup](FASHION_MNIST_ADAPTER.md).

```sh
python -m vision_baseline.fashion_mnist /path/to/fashion-mnist \
  --train-per-class 100 --validation-per-class 50 --test-per-class 50 \
  --seed 20261014 --output /tmp/fashion-manifest.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -W error \
  -m vision_baseline.fashion_pooling_comparison \
  /tmp/fashion-manifest.json /path/to/fashion-mnist \
  --channels 8 --grid-size 4 --seeds 20261014 20261015 20261016 \
  --epochs 100 --learning-rate 0.15 --batch-size 100 --bins 10 \
  --output /tmp/fashion-pooling-comparison.json
python -W error -m unittest discover -s tests -v
```

Each arm trains all prespecified seeds, fits per-run temperature on validation
NLL only and evaluates test data after fitting. The output retains all full
reports, calibration regressions and paired grid-minus-global differences with
mean/sample-standard-deviation/min/max. Positive accuracy differences favor grid;
negative NLL/ECE differences favor grid. The CLI rejects mismatched budgets,
manifests, ordered seeds or training/evaluation settings.

## Measured subset result — October 19, 2026

Python 3.12.14 / NumPy 2.3.5, CPU float64, OpenBLAS/OMP thread counts set to
one. All four official files passed MD5 checks. Manifest SHA-256:
`4b36c62d1043ad2f6ef48953eea5e82aba1f048b1788c20fab7e5a4dc83a5ffc`.

| Seed | Grid accuracy | Global accuracy | Grid minus global |
| --- | ---: | ---: | ---: |
| 20261014 | 72.0% | 34.8% | +37.2 percentage points |
| 20261015 | 72.6% | 37.4% | +35.2 percentage points |
| 20261016 | 73.4% | 35.8% | +37.6 percentage points |
| Mean | 72.67% | 36.00% | +36.67 percentage points |
| Sample stddev | 0.70 percentage points | 1.31 percentage points | 1.29 percentage points |

Mean macro F1 was 0.723750 (grid) versus 0.323571 (global). Mean calibrated
NLL was 0.780547 versus 1.793511, and calibrated ECE was 0.034821 versus
0.089707. Temperature scaling improved global-pool test NLL and ECE in all
three seeds, while the grid arm retained its two NLL and two ECE regressions.

The full six-run report, histories, confusion matrices and paired statistics are
in `results/fashion_mnist_pooling_budget_comparison.json`. The grid arm exactly
reproduced the earlier committed three-seed report. The completed result was
reconstructed with the final comparison code, and all parameter budgets and
500-sample confusion totals passed consistency checks. Fixture CLI repeats are
byte-identical; this full six-run official comparison was executed once, not
independently repeated in its entirety.

The result supports the grid model **under this prescribed budget and training
protocol**. It does not establish a causal pooling-only effect or show that the
wide global model cannot improve with a separately selected training protocol.

## Interpretation limits

- This controls parameter count, but reallocates capacity between convolution
  and classifier. A result cannot prove that pooling alone caused a difference.
- Same seed labels give matched experiment labels, not identical initial weights
  for differently shaped arrays. Both arms receive the same minibatch shuffle.
- It is not compute matched, and no timing or GPU claim is made.
- The fixed test subset has been evaluated previously. It is not fresh independent
  generalization evidence, and neither learning rate nor epoch count is tuned
  against the new test results.
- Three-run sample standard deviation is descriptive, not a confidence interval
  or a significance test. No best seed is selected.
- Fixture CLI runs are byte-identical across repeated executions; official-data
  results are recorded only after actual runs finish.
