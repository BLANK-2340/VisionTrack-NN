# Spatial Fashion-MNIST comparison

The original TinyConvNet averages each feature map to one number. That makes it
small, but discards where visual evidence occurred. This follow-up preserves a
coarse 4×4 activation grid before the linear classifier while leaving the
convolution, optimizer, preprocessing, manifest and evaluation policy simple
and inspectable.

```bash
python -m vision_baseline.fashion_spatial_experiment \
  /tmp/fashion-manifest.json /path/to/fashion-mnist \
  --output /tmp/fashion-spatial.json \
  --channels 8 --grid-size 4 --epochs 100 \
  --learning-rate 0.15 --batch-size 100 --seed 20261014
```

## Controlled result

Both rows used manifest SHA-256
`4b36c62d1043ad2f6ef48953eea5e82aba1f048b1788c20fab7e5a4dc83a5ffc`:
1,000 balanced training images, 500 validation images and 500 untouched test
images from the official checksum-verified files.

| Model | Parameters | Test accuracy | Macro F1 | Calibrated NLL |
|---|---:|---:|---:|---:|
| Global average pool control | 330 | 28.8% | 0.272427 | 1.858109 |
| 4×4 spatial grid | 1,370 | 72.0% | 0.721262 | 0.788264 |

Retaining coarse location improved accuracy by **43.2 percentage points**. Two
independent executions produced byte-identical full JSON reports. Selected
analytical gradients for convolution, convolution bias, classifier weights and
classifier bias matched central finite differences.

Temperature scaling was selected solely on validation NLL. It changed test NLL
from `0.788757` to `0.788264`, but test ECE moved from `0.021998` to `0.031309`.
That ECE regression is retained in the result rather than selectively omitted.

## Scope

- This remains a small educational model and subset result, not a state-of-the-art
  Fashion-MNIST claim.
- The comparison isolates the pooling representation, but channel count and
  resulting parameter count differ. It is not a parameter-matched ablation.
- Only one seed is measured; run-to-run variance is not estimated.
- No augmentation, hyperparameter sweep, test tuning or controlled timing was used.
- The committed comparison artifact contains the complete true-row/predicted-
  column confusion matrix. Dataset files are not redistributed.
