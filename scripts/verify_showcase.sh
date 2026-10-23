#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
archive_root="$(mktemp -d "${TMPDIR:-/tmp}/visiontrack-showcase.XXXXXX")"
trap 'rm -rf "$archive_root"' EXIT

git -C "$repo_root" archive --format=tar HEAD | tar -xf - -C "$archive_root"
cd "$archive_root"

python -W error -m unittest discover -s tests -v
python -m compileall -q vision_baseline tests
python -W error -m vision_baseline.validate_results results

python -W error -m vision_baseline.environment_report \
  --output "$archive_root/environment-a.json"
python -W error -m vision_baseline.environment_report \
  --output "$archive_root/environment-b.json"
cmp "$archive_root/environment-a.json" "$archive_root/environment-b.json"

python -W error -m vision_baseline.demo --checkpoint "$archive_root/orientation.vtnn" \
  > "$archive_root/demo.txt"
python -W error -m vision_baseline.infer \
  "$archive_root/orientation.vtnn" examples/vertical_12x12.pgm \
  > "$archive_root/inference.txt"
grep -Fx \
  "class_index=0 label=vertical height=12 width=12" \
  "$archive_root/inference.txt"

python -W error -m vision_baseline.evaluate \
  --output "$archive_root/evaluation-a.json"
python -W error -m vision_baseline.evaluate \
  --output "$archive_root/evaluation-b.json"
cmp "$archive_root/evaluation-a.json" "$archive_root/evaluation-b.json"

echo "showcase verification passed for $(git -C "$repo_root" rev-parse HEAD)"
