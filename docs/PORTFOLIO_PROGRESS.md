# Portfolio continuity

For the concise runnable-project index, measured-result boundaries and current
limitations, see [SHOWCASE.md](SHOWCASE.md).

## October 23, 2026 — showcase verification (Asia/Kolkata)

Started from `344498caba8ca377fa90d6b2fe3b367316a84275` with no open
pull requests and no commits yet today across the accessible allowlisted
repositories. The original `main.py`, prior models, reports and result artifacts
remain unchanged.

### Completed

- Added a committed-tree verifier that exports `HEAD` before running the complete
  suite, compilation, checkpointed demo, PGM inference and byte-repeatable
  calibrated evaluation.
- Added a concise showcase index of setup commands, runnable targets, evidence,
  recorded results and explicit interpretation limits.
- Clearly separates the synthetic regression evidence and measured official
  1,000/500/500 Fashion-MNIST subset results from a full-dataset benchmark claim.

### Verification

- Python 3.12 / NumPy 2.3.5 passed all 42 tests with warnings promoted to
  errors, plus bytecode compilation.
- The checkpointed synthetic demo and included PGM inference path passed; the
  image was classified as class 0 (`vertical`).
- Two calibrated-evaluation runs produced byte-identical JSON. Final
  committed-tree archive verification is run immediately before publication.

### Next concrete milestone

Add schema and cross-artifact consistency validation for committed result JSON,
then consider a separately fixed full official Fashion-MNIST protocol only if
available CPU time permits. Do not tune against the repeatedly used test subset.

## October 19, 2026 — matched-budget continuation (Asia/Kolkata)

Started from main `1d8fbea8ac6a192ea3beadce397993d6b76f32d8`. One
portfolio commit was already present in today's Kolkata window; no open PRs
were present. The original `main.py`, prior implementations and results remain
unchanged.

### Completed

- Added an exactly matched-total-parameter comparison using the established CNN:
  8 channels with 4×4 grid pooling versus 68 channels with global pooling, both
  1,370 active trainable parameters.
- Added architecture-budget derivation, paired seed differences, complete per-arm
  reports and guards against mismatched protocols, manifests, seeds and budgets.
- Added four tests for actual parameter counts, independent global-pooling
  equivalence in forward/gradient calculations, known paired statistics, rejected
  comparisons and two-run fixture CLI byte reproducibility.
- Explicitly documented width/capacity allocation and compute confounders: this
  is not a pure pooling-only causal ablation.

### Verification

- All 42 tests passed under Python 3.12.14 / NumPy 2.3.5 with warnings treated as
  errors. Bytecode compilation and diff checks passed.
- Official data passed upstream MD5 checks and rebuilt the exact previous
  1,000/500/500 manifest. All six prescribed model/seed runs completed.
- At 1,370 parameters per model, mean accuracy was 72.67% (grid) versus 36.00%
  (global); mean paired difference was 36.67 percentage points with sample
  standard deviation 1.29 points. This is a matched-budget comparison, not a
  pooling-only causal claim or independently held-out benchmark result.
- The grid arm exactly reproduced the earlier committed multi-seed report.
  Final-source reconstruction, all six budgets and all confusion totals passed.
  All per-arm reports and calibration regressions are preserved in the result.
- Original training/checkpoint/inference smoke checks passed. The full official
  comparison was run once; only fixture comparisons were independently repeated.
  GitHub Actions results require separate confirmation.

### Next concrete milestone

Rotate to `cuda-kernel-lab` for robustness/reproducibility checks around the
existing CPU benchmark report and available compiler paths. Keep actual GPU,
NVCC and compute-sanitizer measurements pending if the required runtime is absent.

## October 19, 2026 (Asia/Kolkata)

Started from `65c8b6f62415343cbdd564060270dd84f5692b83`, with no open PRs
and no commits yet in today's Kolkata date window across the six accessible
allowlisted repositories. Preserved the original `main.py` and all previous work.

### Completed

- Added a fixed-protocol multi-seed runner around the established spatial CNN,
  retaining every seed's complete report and validation-only calibration.
- Added min/max/mean/sample-standard-deviation summaries for accuracy, F1, NLL
  and ECE, plus paired calibration changes and explicit regression counts.
- Added distinct-seed, identical-protocol and manifest-mutation guards; synthetic
  source manifests remain explicitly distinguished from official files.
- Added five tests for independent known statistics, rejected plans and mismatched
  runs, manifest changes, and repeated real fixture training/CLI artifact bytes.

### Verification

- Python 3.12.14 / NumPy 2.3.5 passed all 38 tests with warnings as errors and
  bytecode compilation. No full-data or GPU result is implied by these tests.
- Official files passed all four MD5 checks; the rebuilt 1,000/500/500 manifest
  exactly matched the earlier comparison's SHA-256.
- Seeds 20261014/15/16 yielded 72.0%, 72.6% and 73.4% test accuracy; mean
  72.67%, sample standard deviation 0.70 percentage points. Mean macro F1 was
  0.723750. Full per-seed reports and aggregation are retained in the result JSON.
- Temperature calibration increased test NLL in two runs and ECE in two runs;
  these regressions are explicitly preserved, not filtered out.
- Two independent full official-data suites produced byte-identical JSON.
  Artifact consistency checks, original train-save-infer smoke checks and
  `git diff --check` passed. Remote CI results require separate confirmation.

### Next concrete milestone

Add a parameter-matched pooling comparison under the same fixed manifest and
prespecified training/seed protocol. Keep architecture capacity differences and
calibration regressions explicit; do not tune against test results.

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

### Spatial comparison completed

- Added a deterministic `conv3x3 -> ReLU -> 4x4 grid average pool -> linear`
  variant with explicit backpropagation and central finite-difference checks.
- On the identical official 1,000/500/500 manifest, test accuracy increased
  from `0.288000` to `0.720000` and macro F1 from `0.272427` to `0.721262`.
- Two complete real-data runs produced byte-identical JSON reports. The recorded
  calibration result includes the slight ECE regression after temperature fitting.
- Added a machine-readable comparison, exact manifest identity, confusion matrix,
  reproducibility notes and limitations. The original control remains unchanged.

### Next concrete milestone

Rotate to another project track. When returning to vision, add multi-seed
evaluation and a parameter-matched pooling ablation before expanding model depth.


