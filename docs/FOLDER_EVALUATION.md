# Image-folder evaluation

This workflow evaluates a saved TinyConvNet against a local PGM dataset and
writes a canonical JSON artifact. It is intended for small, auditable datasets;
it does not download data or claim real-world benchmark quality.

## Dataset contract

The root must contain exactly one directory for each model class. Each class
directory must contain one or more lowercase `.pgm` files and no other entries:

```text
dataset/
  vertical/*.pgm
  horizontal/*.pgm
  diagonal/*.pgm
```

P2, 8-bit P5 and 16-bit P5 images are supported. Symlinks, nested directories,
unknown classes and non-PGM files are rejected so an accidental file cannot
silently change the evaluated sample set. Images may have different dimensions,
because inference is performed one image at a time.

## Run

```bash
python -m vision_baseline.demo --checkpoint /tmp/orientation.vtnn
python -m vision_baseline.evaluate_folder /tmp/orientation.vtnn dataset \
  --output /tmp/folder-evaluation.json
```

Use `--labels cat dog other` for another three-class checkpoint. The label order
must match training. `--temperature` can apply a previously selected positive
temperature, but this command deliberately does not fit one on the test folder.

The artifact records a SHA-256 digest of the exact checkpoint, class order,
sample count, true-row/predicted-column confusion matrix, accuracy, macro and
per-class precision/recall/F1, and sorted per-image predictions. Stored paths are
relative POSIX paths, so identical inputs produce identical bytes regardless of
the machine's absolute dataset location.

## Limitations

- The checkpoint format does not embed semantic class names or a calibrated
  temperature; callers must supply the training-time order and chosen value.
- PGM is used for a dependency-light, inspectable path. PNG/JPEG decoding and
  resize/crop policies are intentionally out of scope.
- Accuracy on a user-provided folder is only meaningful when collection,
  labeling and split provenance are controlled independently.
