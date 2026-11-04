# Labeled-folder abstention audit

The policy-audit command applies a previously fitted policy to the strict
`dataset/<label>/*.pgm` layout used by folder evaluation:

```bash
python -m vision_baseline.audit_policy model.vtnn labeled-images/ evaluation.json \
  --output policy-audit.json
```

It reports aggregate and per-class sample counts, accepted counts, coverage,
accepted errors and selective risk. Every prediction retains a relative path,
target, prediction, top probability, confidence margin and acceptance decision.
Classes with no accepted samples remain in the report with `accepted: 0`,
`errors: 0`, and `risk: null`; they are never silently omitted or assigned an
artificial zero risk.

The threshold and its validation evidence are loaded from the existing policy
record and are not adjusted using the audit dataset. As a result, the audit is
valid evidence about one frozen operating point, not a procedure for selecting
that point. Per-class results on very small supports are descriptive only.
