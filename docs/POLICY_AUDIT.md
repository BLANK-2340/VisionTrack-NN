# Labeled-folder abstention audit

The policy-audit command applies a previously fitted policy to the strict
`dataset/<label>/*.pgm` layout used by folder evaluation:

```bash
python -m vision_baseline.audit_policy model.vtnn labeled-images/ evaluation.json \
  --output policy-audit.json
```

It reports aggregate and per-class sample counts, accepted counts, coverage,
accepted errors and selective risk. Every prediction retains a relative path,
target, prediction, top probability, confidence margin, acceptance decision and
input SHA-256. The report also binds the canonical frozen policy and the ordered
labeled-input manifest to SHA-256 identities, alongside the checkpoint hash.
Classes with no accepted samples remain in the report with `accepted: 0`,
`errors: 0`, and `risk: null`; they are never silently omitted or assigned an
artificial zero risk.

Validate a saved report against the exact local checkpoint, policy and labeled
inputs with:

```bash
python -m vision_baseline.validate_policy_audit policy-audit.json \
  model.vtnn labeled-images/ evaluation.json --temperature 1.0
```

The validator accepts only bounded, duplicate-free, finite, canonical JSON and
regenerates the entire report from the supplied sources. A changed checkpoint,
policy value, label mapping, input path, input byte or prediction therefore
causes validation to fail. SHA-256 here provides deterministic content identity,
not authenticity against a malicious party; trusted distribution would require
a separately authenticated signature.

The threshold and its validation evidence are loaded from the existing policy
record and are not adjusted using the audit dataset. As a result, the audit is
valid evidence about one frozen operating point, not a procedure for selecting
that point. Per-class results on very small supports are descriptive only.
