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


