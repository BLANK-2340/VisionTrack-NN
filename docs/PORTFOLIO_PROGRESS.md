# Portfolio continuity

## October 7, 2026 (Asia/Kolkata)

Source of truth before this session: `main` at
`1a66a6beef5381f9cd17aae4525442c27bffad2a`. No open pull requests or
October 7 commits existed in the authorized repositories when inspected.

### Completed

- Added a deterministic, versioned, checksum-protected TinyConvNet checkpoint
  format with path and bytes APIs, exact array restoration and no pickle.
- Added strict architecture, shape, finiteness, payload, checksum and resource
  validation, including overflow-safe handling of untrusted shape metadata.
- Added P2/P5 PGM parsing for 8-bit and 16-bit images, plus a command-line
  checkpoint-to-image inference path and original 12x12 demo image.
- Extended the training demo with optional checkpoint output.
- Added six persistence/inference tests alongside the four existing baseline
  tests, and expanded architecture, limitations and reproducibility documentation.
- Preserved the repository's original `main.py` and all pre-delegation files.

### Actual verification

- `python -m unittest discover -s tests -v`: all 10 tests passed under Python
  3.12.14 and NumPy 2.3.5.
- `python -m vision_baseline.demo`: reproduced loss `1.108148 -> 0.011034`
  and train/test accuracy `1.0000` on the narrow synthetic orientation task.
- Train-save-infer pipeline produced a checkpoint and classified
  `examples/vertical_12x12.pgm` as class 0 (`vertical`).
- No real-world dataset result or GPU measurement is claimed.

### Next concrete milestone

Rotate to `cuda-kernel-lab` for CPU-oracle matrix-multiplication boundary tests
and a bounds-safe tiled CUDA kernel/benchmark. Keep NVCC, compute-sanitizer and GPU
performance validation explicitly pending until actual GPU access is available.
After that, add a small attributed public-data baseline to this vision project.

## October 12, 2026 (Asia/Kolkata)

### Completed

- Added stable probability prediction, deterministic scalar temperature fitting,
  NLL, multiclass Brier score, equal-width ECE reliability bins, and selective
  risk/coverage evaluation.
- Added independent calibration/test corruptions and a canonical JSON artifact
  containing the full reproducibility protocol and seeds.
- Added six evaluation tests, invalid-input coverage, two-run artifact comparison
  in CI, and a measured-results document.
- Preserved the original `main.py` and all pre-delegation files.

### Actual local verification

- Python 3.12.14 and NumPy 2.3.5 passed all 16 tests and bytecode compilation.
- Two CLI runs produced identical JSON; calibrated corrupted-test NLL was
  `0.057906`, Brier score `0.035628`, ECE `0.031575`, and accuracy `0.977778`.
- These are controlled synthetic-corruption results, not real-world performance.

### Next concrete milestone

Add reproducible CPU benchmark JSON and cross-kernel metadata to
`cuda-kernel-lab`, while keeping GPU measurements explicitly pending until a
real CUDA runtime is available.

## October 13, 2026 (Asia/Kolkata)

### Completed

- Added a deterministic image-folder evaluator for saved TinyConvNet checkpoints
  and strict class-directory PGM datasets.
- Added checkpoint identity, sorted relative sample paths, a true-row/predicted-
  column confusion matrix, accuracy, and macro/per-class precision, recall and F1
  to a canonical JSON artifact.
- Rejects ambiguous labels, missing or unknown classes, symlinks, nested content,
  non-PGM files, empty classes and invalid inference temperatures.
- Added four focused tests covering exact metrics, deterministic CLI bytes,
  portable paths and invalid layouts, while preserving the original `main.py`.

### Verification scope

- Full unit-test, bytecode and CLI checks are run
  before publishing; the resulting exact counts belong in the commit report.
- No external dataset accuracy, GPU result or production robustness is claimed.

### Next concrete milestone

Add a small attributed public-data adapter with an explicit preprocessing and
split manifest, while keeping the synthetic and user-folder results clearly
separated.

## October 14, 2026 (Asia/Kolkata)

### Completed

- Added an original dependency-light Fashion-MNIST IDX/gzip adapter with strict
  magic, shape, count, label, payload and resource-limit validation.
- Added upstream MD5 enforcement, recorded SHA-256 source identities and
  re-verification when materializing a split.
- Added deterministic, class-balanced train/validation/test manifests with exact
  sorted indices and explicit preprocessing; validation never consumes the
  official test source.
- Added five synthetic-fixture tests and attribution, license, usage and
  limitation documentation without redistributing dataset files.
- Preserved the repository's original `main.py` and prior task-owned features.

### Actual verification

- All 25 tests passed with warnings promoted to errors; bytecode compilation
  also passed.
- The upstream repository's four official files passed every published MD5
  check and produced a canonical 105,526-byte manifest.
- Materialized train/validation/test arrays had shapes `(5000, 28, 28)`,
  `(1000, 28, 28)` and `(1000, 28, 28)`, with all ten labels represented.
- No Fashion-MNIST model accuracy is claimed by this data-adapter milestone.

### Next concrete milestone

Run a controlled Fashion-MNIST subset experiment with a fixed training protocol,
then report accuracy, confusion and calibration artifacts separately from the
synthetic orientation benchmark.

### Controlled experiment completed

- Added a deterministic train/validation/test runner, optional checkpoint output,
  canonical report, confusion matrix and per-class/macro metrics.
- Ran one fixed 330-parameter TinyConvNet protocol on official, balanced
  1,000/500/500 subsets. Test accuracy was `0.288000`, macro F1 `0.272427`,
  calibrated NLL `1.858109` and 10-bin ECE `0.057498`.
- Added four focused tests covering matrix orientation, split accounting,
  deterministic report/checkpoint bytes and invalid hyperparameters.
- Recorded the weak result explicitly: global average pooling loses spatial
  layout, so this is an educational baseline rather than a competitive claim.

### Next concrete milestone

Add a spatially preserving model variant and compare it using the same manifest
and evaluation protocol. Keep the existing low-capacity result as the honest
control rather than replacing it.


