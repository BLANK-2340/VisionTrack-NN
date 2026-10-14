# Fashion-MNIST data adapter

This adapter adds an attributed real-data path without vendoring or silently
downloading the dataset. Fashion-MNIST contains 60,000 training and 10,000 test
28×28 grayscale fashion images in ten balanced classes.

Sources:

- [Official repository, checksums, labels and MIT license][fashion-mnist]
- [Xiao, Rasul and Vollgraf, 2017](https://arxiv.org/abs/1708.07747)

[fashion-mnist]: https://github.com/zalandoresearch/fashion-mnist

Download the four official gzip files listed in the upstream README into one
directory. Then create an explicit stratified manifest:

```bash
python -m vision_baseline.fashion_mnist /path/to/fashion-mnist \
  --output /tmp/fashion-mnist-manifest.json
```

The default manifest selects 5,000 training, 1,000 validation and 1,000 test
examples. Validation is drawn only from the official training source; the
official test source remains test-only. Selection is deterministic within each
class and records the exact sorted IDX indices.

The adapter validates gzip/IDX structure, 28×28 shape, sample counts, labels,
payload sizes and the four MD5 values published upstream. The manifest also
records SHA-256 for each exact compressed input. `load_split` verifies those
SHA-256 identities again before returning `float64(pixel) / 255.0` images. No
resize, crop, inversion or augmentation is applied.

`--allow-nonofficial-files` exists for controlled synthetic tests and trusted
mirrors. Such a manifest records failed official-MD5 flags and must not be
presented as an official-data result.

## Scope and limitations

- This commit adds ingestion and split provenance, not an accuracy claim.
- The small TinyConvNet architecture was designed for the orientation demo;
  training/evaluation settings for Fashion-MNIST require a separate controlled
  experiment.
- Near-duplicate and label-quality concerns are not corrected by this adapter.
- Dataset files are not redistributed by this repository.
