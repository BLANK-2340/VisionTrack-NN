# Deterministic inference ranking

`vision_baseline.ranking` turns validated class probabilities into deterministic
top-k predictions. Rows are ordered by descending probability; exact ties use
ascending class index, so results do not depend on an unstable sorting choice.

```python
from vision_baseline import predict_top_k

ranking = predict_top_k(model, images, k=3, temperature=1.0)
print(ranking.indices)
print(ranking.probabilities)
print(ranking.confidence_margin)
```

The confidence margin is always the top probability minus the runner-up. It is
computed even when `k=1`, making thresholding independent of presentation depth.
The API rejects empty batches, fewer than two classes, non-floating, non-finite,
out-of-range or non-normalized inputs and invalid `k` values.

This utility reports model confidence; it does not make confidence a correctness
guarantee. Temperature scaling and margin thresholds must be fixed on validation
data rather than tuned on the test set.
