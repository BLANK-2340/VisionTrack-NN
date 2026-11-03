# Validation-fixed abstention

`vision_baseline.abstention` turns the existing top-two probability margin into
an optional accept/abstain decision. It does not change default inference.

Fit once on a validation split, then reuse the returned policy unchanged:

```python
from vision_baseline.abstention import abstention_metrics, fit_abstention_policy

policy = fit_abstention_policy(validation_probabilities, validation_labels, 0.9)
test_report = abstention_metrics(test_probabilities, test_labels, policy)
```

The requested coverage selects a validation margin rank. Exact ties at that
boundary are all accepted, so recorded validation coverage can be higher than
requested coverage. Test coverage is allowed to differ: the threshold is never
refit on test data. Selective risk is the error fraction among accepted samples;
it is `null` when no test sample is accepted.

The policy is a reproducible operating point, not a correctness guarantee.
Coverage and risk can shift under distribution change, and any choice of target
coverage must be made before inspecting final test outcomes.
