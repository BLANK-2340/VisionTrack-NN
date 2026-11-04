# Policy-aware folder inference

The folder-inference command applies a validation-fixed abstention policy to a
flat directory of P2/P5 PGM images. It records every file's top prediction,
probability, top-two confidence margin and accepted/abstained decision.

```bash
python -m vision_baseline.infer_folder model.vtnn images/ evaluation.json \
  --output folder-inference.json
```

`evaluation.json` may be the calibration report produced by
`python -m vision_baseline.evaluate`; the command reads its
`abstention_policy` record. A standalone policy record is also accepted. Policy
JSON rejects duplicate keys, non-finite values, inconsistent counts and files
larger than 64 KiB. Image paths are relative and deterministically sorted.

The command never fits or adjusts the threshold against the inference folder.
Coverage can therefore differ from validation coverage. The folder is unlabeled,
so the report cannot claim accuracy or selective risk. Confidence, margin and an
accepted decision are not guarantees of correctness.
