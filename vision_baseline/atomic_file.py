"""Same-directory atomic publication for task-owned artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def atomic_write_bytes(
    path: str | Path,
    data: bytes,
    *,
    _replace: Callable[[str | Path, str | Path], None] = os.replace,
) -> None:
    """Publish *data* only after a sibling temporary file is flushed and synced.

    The fixed ``.tmp`` sibling is opened exclusively, so concurrent or stale
    writers are rejected instead of overwritten. ``_replace`` exists for
    deterministic failure-injection tests; callers should use the default.
    """
    destination = Path(path)
    if not destination.name:
        raise ValueError("atomic destination must name a file")
    temporary = destination.with_name(destination.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as output:
            created = True
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        _replace(temporary, destination)
    except Exception:
        if created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
